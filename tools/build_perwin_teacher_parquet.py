"""Build a row-aligned per-window TMR teacher cache for a parquet feature cache.

For every row of a feature cache from :mod:`tools.build_varlen_cache_parquet`
(``varlen`` clip or ``fixed`` window), encode that row's motion span with the
frozen TMR encoder into one unit-norm ``[256]`` embedding. Output ``<out>.npy``
is ``[N_rows, 256]`` in the SAME row order as the feature cache, guaranteed by
re-deriving the identical :func:`~tools.build_varlen_cache_parquet.plan_rows`
plan and asserting the ``rec_ids`` match the cache's ``.rec_ids.json``.

Multi-GPU: ``--create-only`` once, then one ``--num-shards K --shard-index i``
worker per GPU::

    python -m tools.build_perwin_teacher_parquet --feature-cache <cache>.npy \
        --parquet-dir .../train --out <cache>.tmr_perwin.npy
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tools.build_varlen_cache_parquet import plan_rows
from tools.tmr_adapter_smoke import DEFAULT_TEACHER_DIR

DSEM = 256
TMR_FPS = 30.0
TMR_MAX_FRAMES = 300
UMR_FPS = 50.0


def _shard_range(n: int, num_shards: int, shard_index: int) -> tuple[int, int]:
    per = (n + num_shards - 1) // num_shards
    lo = shard_index * per
    hi = min(n, lo + per)
    return lo, hi


def _read_joints_flat(sh: str):
    ca = pq.read_table(sh, columns=["joints77_pos"]).column("joints77_pos").combine_chunks()
    flat = np.asarray(ca.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    off = ca.offsets.to_numpy()
    return flat, off


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--feature-cache", required=True, type=Path, help="the .npy whose meta defines the plan")
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out", required=True, type=Path, help="teacher .npy [N_rows, 256]")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--create-only", action="store_true", help="allocate the [N,256] file and exit")
    ap.add_argument("--resume", action="store_true",
                    help="skip rows already filled (unit-norm != 0) -- node-death safe restart")
    ap.add_argument("--resample", choices=["poly", "linear"], default="poly",
                    help="50->30 resampler: 'poly' (anti-aliased polyphase, default) or 'linear'")
    ap.add_argument("--per-token", action="store_true",
                    help="emit ONE TMR embedding per AR token. fixed cache -> dense "
                         "[N_rows, T_token, 256] (T_token = window // token-stride); varlen cache -> "
                         "ragged [sum(len_i//stride), 256] + <out>.index.npy [N_rows,2]=(tok_off,n_tok). "
                         "Each token's TMR receptive field is a tmr-window (50fps) segment centred on "
                         "the token; overlap = tmr-window - token-stride.")
    ap.add_argument("--token-stride", type=int, default=4,
                    help="UMR (50fps) frames per AR token; MUST equal the codec's temporal_stride "
                         "so T_token matches the codec token count (default 4).")
    ap.add_argument("--tmr-window", type=int, default=None,
                    help="per-token TMR receptive field in 50fps frames (default = token-stride, "
                         "non-overlapping). W > stride gives overlap = W - stride.")
    ap.add_argument("--tmr-block", action="store_true",
                    help="NO-OVERLAP block teacher (requires --per-token): token g's target is the "
                         "non-overlapping block [blk*W, blk*W+W) with blk=(g*stride)//W and "
                         "W=tmr-window; all tokens in a block SHARE one embedding.")
    ap.add_argument("--group-tokens", type=int, default=1,
                    help="sliding per-token mode: number of CONSECUTIVE tokens that SHARE one "
                         "embedding (G). Semantic hop = G*token-stride frames; overlap between "
                         "groups = tmr-window - G*token-stride. G=1 (default) = one target per "
                         "token. Ignored when --tmr-block is set.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=2000)
    ap.add_argument("--teacher-dir", default=str(DEFAULT_TEACHER_DIR),
                    help="teacher checkpoint dir containing config.yaml (default: %(default)s)")
    args = ap.parse_args(argv)

    meta = json.loads(args.feature_cache.with_suffix(args.feature_cache.suffix + ".meta.json").read_text())
    mode = str(meta.get("mode", "varlen" if meta.get("varlen") else "fixed"))
    shards = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
    rows, rec_ids = plan_rows(
        shards, mode=mode,
        min_frames=int(meta.get("min_frames", 48)), max_frames=int(meta.get("max_frames", 300)),
        window=int(meta.get("window", 64)), step=int(meta.get("step", 32)),
        max_windows_per_clip=int(meta.get("max_windows_per_clip", 64)),
        token_stride=int(meta.get("token_stride", 4)),
    )
    n_rows = len(rows)
    # alignment guard against the feature cache's own rec_ids
    rec_json = args.feature_cache.with_suffix(args.feature_cache.suffix + ".rec_ids.json")
    if rec_json.is_file():
        cache_recs = json.loads(rec_json.read_text())
        if cache_recs != rec_ids:
            raise RuntimeError(
                f"plan rec_ids ({len(rec_ids)}) do not match feature cache rec_ids "
                f"({len(cache_recs)}); teacher would be misaligned"
            )
    per_token = bool(args.per_token)
    block_mode = bool(args.tmr_block)
    if block_mode and not per_token:
        raise ValueError("--tmr-block requires --per-token")
    group_tokens = max(1, int(args.group_tokens))
    if group_tokens > 1 and block_mode:
        raise ValueError("--group-tokens is for sliding per-token mode; not compatible with --tmr-block")
    window = int(meta.get("window", 64))
    token_stride = int(args.token_stride)
    tmr_window = int(args.tmr_window) if args.tmr_window else token_stride

    # per-token storage: fixed -> dense [n_rows, T_token, DSEM]; varlen -> ragged
    # [sum(len_i//stride), DSEM] + <out>.index.npy, token rows in feature-cache clip order.
    tok_index: np.ndarray | None = None  # varlen per-token only
    if per_token and mode == "varlen":
        n_tok_arr = np.array([int(r[3]) // token_stride for r in rows], dtype=np.int64)
        _off = np.zeros(n_rows + 1, dtype=np.int64)
        np.cumsum(n_tok_arr, out=_off[1:])
        total_tokens = int(_off[-1])
        tok_index = np.stack([_off[:-1], n_tok_arr], axis=1)
        t_token = 0
        cache_shape: tuple[int, ...] = (total_tokens, DSEM)
    elif per_token:
        if window % token_stride != 0:
            raise ValueError(f"window={window} not divisible by token-stride={token_stride}")
        t_token = window // token_stride
        cache_shape = (n_rows, t_token, DSEM)
    else:
        t_token = 0
        cache_shape = (n_rows, DSEM)
    print(f"[teacher] mode={mode} n_rows={n_rows} per_token={per_token} "
          f"t_token={t_token} tmr_window={tmr_window} stride={token_stride} "
          f"block={block_mode} group_tokens={group_tokens} "
          f"shard {args.shard_index}/{args.num_shards}")

    idx_path = args.out.with_suffix(args.out.suffix + ".index.npy")
    if args.create_only or not args.out.is_file():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        mm = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float32, shape=cache_shape)
        mm.flush(); del mm
        if tok_index is not None:
            np.save(idx_path, tok_index)
        meta_out = {
            "teacher": ("per_token_block" if (per_token and block_mode)
                        else "per_token" if per_token else "perwin"),
            "mode": mode, "dsem": DSEM, "n_rows": n_rows,
            "tmr_fps": TMR_FPS, "resample": f"{args.resample}_50to30", "source": "umr499_joints77_pos",
            "feature_cache": str(args.feature_cache.name),
            "per_token": per_token,
            "varlen_per_token": bool(per_token and mode == "varlen"),
            "t_token": (None if (per_token and mode == "varlen") else (t_token if per_token else None)),
            "total_tokens": (int(cache_shape[0]) if (per_token and mode == "varlen") else None),
            "index_path": (idx_path.name if tok_index is not None else None),
            "token_stride": token_stride if per_token else None,
            "tmr_window_frames": tmr_window if per_token else window,
            "tmr_block": block_mode if per_token else None,
            "group_tokens": group_tokens if per_token else None,
            "tmr_block_overlap": (0 if block_mode else (tmr_window - group_tokens * token_stride))
                                 if per_token else None,
        }
        args.out.with_suffix(args.out.suffix + ".meta.json").write_text(json.dumps(meta_out, indent=2))
        print(f"[teacher] allocated {args.out} shape={cache_shape}")
        if args.create_only:
            return 0
    if tok_index is None and idx_path.is_file():
        tok_index = np.load(idx_path)

    lo, hi = _shard_range(n_rows, args.num_shards, args.shard_index)
    print(f"[teacher] shard {args.shard_index}: rows [{lo},{hi})")

    import torch
    from tools.tmr_adapter_smoke import (
        load_soma_skeleton77,
        load_tmr,
        resample_linear_time,
    )

    if args.resample == "poly":
        from math import gcd
        from scipy.signal import resample_poly

        def _resample(seg: np.ndarray, src: float, dst: float) -> np.ndarray:
            si, di = int(round(src)), int(round(dst))
            g = gcd(si, di)
            # anti-aliased polyphase: up=dst/gcd, down=src/gcd (50->30 == up 3 / down 5)
            return resample_poly(seg, up=di // g, down=si // g, axis=0).astype(np.float32)
    else:
        _resample = resample_linear_time

    device = args.device
    tmr = load_tmr(args.teacher_dir, device=device)
    skel77 = load_soma_skeleton77()

    def _encode(seg30: np.ndarray) -> np.ndarray:
        posed = torch.from_numpy(np.ascontiguousarray(seg30)).float().to(device)[None]
        with torch.no_grad():
            e = tmr.encode_motion(posed_joints=posed, original_skeleton=skel77)
        return e[0].detach().cpu().numpy().astype(np.float32)

    _ZERO = np.zeros(DSEM, np.float32)

    def _encode_range(win_j50: np.ndarray, a: int, b: int) -> np.ndarray:
        """TMR-encode window-local 50fps frame span [a,b) -> unit-norm [256] (0 if <2f)."""
        a = max(0, a); b = min(win_j50.shape[0], b)
        if b - a < 2:
            return _ZERO
        s30 = _resample(win_j50[a:b], UMR_FPS, TMR_FPS)[:TMR_MAX_FRAMES]
        return _encode(s30) if s30.shape[0] >= 2 else _ZERO

    def _pertoken_embs(win_j50: np.ndarray, n_tokens: int) -> np.ndarray:
        """Per-token TMR targets [n_tokens, DSEM] for one window/clip.

        block_mode: tokens share one embedding per non-overlapping block of W=tmr_window frames.
        sliding: group_tokens consecutive tokens share one embedding centred on their combined span.
        """
        arr = np.zeros((n_tokens, DSEM), np.float32)
        if block_mode:
            blk_cache: dict[int, np.ndarray] = {}
            for lt in range(n_tokens):
                blk = (lt * token_stride) // tmr_window
                if blk not in blk_cache:
                    a = blk * tmr_window
                    blk_cache[blk] = _encode_range(win_j50, a, a + tmr_window)
                arr[lt] = blk_cache[blk]
        else:
            half = tmr_window / 2.0
            gspan = group_tokens * token_stride
            grp_cache: dict[int, np.ndarray] = {}
            for lt in range(n_tokens):
                gp = lt // group_tokens
                if gp not in grp_cache:
                    c = gp * gspan + gspan / 2.0
                    grp_cache[gp] = _encode_range(win_j50, int(round(c - half)), int(round(c + half)))
                arr[lt] = grp_cache[gp]
        return arr

    emb = np.load(args.out, mmap_mode="r+")
    varlen_pt = per_token and tok_index is not None

    def _clip_done(gi: int) -> bool:
        # resume marker: perwin/fixed use row gi; varlen per-token uses the clip's first token row.
        probe = emb[int(tok_index[gi, 0])] if varlen_pt else emb[gi]
        return float(np.linalg.norm(probe)) > 1e-6

    by_shard: dict[str, list[tuple[int, int, int]]] = {}
    for gi in range(lo, hi):
        sh, row, start, length = rows[gi]
        by_shard.setdefault(sh, []).append((row, start, length, gi))

    done = 0
    n_bad = 0
    n_skip = 0
    for sh in sorted(by_shard):
        # resume: skip shards whose rows are all already filled
        if args.resume:
            pending = [t for t in by_shard[sh] if not _clip_done(t[3])]
            if not pending:
                n_skip += len(by_shard[sh]); done += len(by_shard[sh]); continue
        else:
            pending = by_shard[sh]
        flat, off = _read_joints_flat(sh)
        for row, start, length, gi in pending:
            s = int(off[row]); e = int(off[row + 1])
            j50 = flat[s:e].reshape(-1, 77, 3)
            win = j50[start:start + length]
            if win.shape[0] < 2:
                n_bad += 1; done += 1; continue
            if not per_token:
                s30 = _resample(win, UMR_FPS, TMR_FPS)[:TMR_MAX_FRAMES]
                emb[gi] = _encode(s30) if s30.shape[0] >= 2 else 0.0
            elif varlen_pt:
                base = int(tok_index[gi, 0]); n_tok = int(tok_index[gi, 1])
                if n_tok > 0:
                    emb[base:base + n_tok] = _pertoken_embs(win, n_tok)
            else:
                emb[gi, :] = _pertoken_embs(win, t_token)
            done += 1
            if done % args.log_every == 0:
                print(f"[teacher] shard {args.shard_index}: {done}/{hi-lo} (bad={n_bad})", flush=True)
        del flat
    emb.flush()
    print(f"[teacher] shard {args.shard_index}: DONE {done} rows (bad={n_bad} skip={n_skip})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
