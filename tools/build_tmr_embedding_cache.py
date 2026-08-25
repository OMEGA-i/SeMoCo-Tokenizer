"""Build a row-aligned [N_windows, 256] TMR teacher-embedding cache for semantic distillation.

Row-alignment contract: the sibling ``.meta.json`` of the input training cache
(``umr499_train.npy``) supplies per-window provenance ``samples[i] =
{clip_id, source_path, start_frame}``; emitted row ``i`` of ``tmr_emb_*.npy``
supervises window ``i``. A sibling ``.meta.json`` is written alongside.

Usage (multi-GPU: ``--create-only`` once, one ``--num-shards K --shard-index i``
worker per GPU, then ``--write-meta``; full commands in README)::

    python -m tools.build_tmr_embedding_cache \
        --umr-cache local://cache/umr499_train.npy --out local://cache/tmr_emb_train.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from data.local_uri import default_data_root, resolve_local_uri
from data.umr_cached_dataset import cache_meta_path, load_cache_meta
from data.umr_schema import UMR_FPS
from tools.tmr_adapter_smoke import DEFAULT_TEACHER_DIR, load_soma_skeleton77, load_tmr

TMR_FPS = 30.0
TMR_MAX_FRAMES = 300
DSEM = 256
SOMA77_NPZ_NAME = "soma77.npz"

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--umr-cache", required=True,
                   help="umr499 training cache .npy (local:// ok). Its .meta.json supplies per-window provenance.")
    p.add_argument("--out", required=True,
                   help="Output tmr_emb .npy (local:// ok). Sibling .meta.json written alongside.")
    p.add_argument("--data-root", default=None,
                   help="Resolve local:// URIs (default: $MOTIONVERSE_DATA_ROOT or ../omega-MotionVerse).")
    p.add_argument("--teacher-dir", default=str(DEFAULT_TEACHER_DIR),
                   help="Directory with the frozen retrieval-teacher checkpoint and its config.yaml "
                        "(default: %(default)s; see README for how to obtain it).")
    p.add_argument("--max-windows", type=int, default=None,
                   help="Only process the first N windows (validation subset).")
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available).")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--num-shards", type=int, default=1, help="Total shards (one GPU worker each).")
    p.add_argument("--shard-index", type=int, default=0, help="This worker's shard index in [0, num-shards).")
    p.add_argument("--create-only", action="store_true",
                   help="Allocate the zero-filled memmap and exit (run once before workers).")
    p.add_argument("--write-meta", action="store_true",
                   help="Write the sibling .meta.json and exit (run once after all workers finish).")
    p.add_argument("--per-token", action="store_true",
                   help="Emit ONE TMR embedding per AR token instead of one per window. "
                        "Output shape becomes [N_windows, T_token, 256]; each token gets a "
                        "short sliding TMR window (see --tmr-window) so its q0 semantic "
                        "target varies over time (fixes q0-copy collapse).")
    p.add_argument("--token-stride", type=int, default=4,
                   help="UMR (50fps) frames per AR token = umr_fps / token_rate (default 4 = 50/12.5).")
    p.add_argument("--tmr-window", type=int, default=None,
                   help="Per-token TMR receptive field in UMR (50fps) frames (default = token-stride, "
                        "i.e. non-overlapping). W>stride gives overlap = W - stride. Larger W = more "
                        "temporal context per token (TMR is weak on <~0.5s segments).")
    p.add_argument("--tmr-block", action="store_true",
                   help="NO-OVERLAP block teacher (requires --per-token, teacher=tmr_clip). Instead of a "
                        "per-token sliding window centred on each token, token g's TMR target is the "
                        "NON-OVERLAPPING block [blk*W, blk*W+W) with blk=(g*token_stride)//W. All tokens "
                        "in a block SHARE one embedding => adjacent q0 targets are identical within a "
                        "block, overlap between blocks = 0 ('super-short perwin', W frames per block). "
                        "Default (unset) keeps the centred per-token behaviour unchanged.")
    p.add_argument("--event-aligned", action="store_true",
                   help="EVENT-ALIGNED teacher (requires --per-token, teacher=tmr_clip; mutually "
                        "exclusive with --tmr-block). Token g's TMR target is the pooled clip-level "
                        "embedding of the ANNOTATED EVENT that g's centre time falls in. All tokens "
                        "inside one event SHARE one embedding => q0 is piecewise-constant per "
                        "semantic event with a crisp change at every event boundary "
                        "(variable-length, semantically meaningful 'blocks'). Needs --events-jsonl.")
    p.add_argument("--events-jsonl", default=None,
                   help="Path to an event-annotation JSONL file, one line per recording: "
                        "{\"rec_id\": str, \"events\": [{\"start_time\": sec, \"end_time\": sec}, ...]}. "
                        "Required with --event-aligned.")
    p.add_argument("--joints-source", choices=("soma77", "umr499"), default="soma77",
                   help="Teacher joint source. 'soma77' (default): native soma77.npz -> SOMA-X FK -> "
                        "polyphase 120->30 (highest fidelity, needs soma77.npz). 'umr499': the "
                        "umr499.npz joints77_pos @50fps -> linear resample 50->30 (self-contained).")
    p.add_argument("--teacher", choices=("tmr_clip", "tmr_frame"), default="tmr_clip",
                   help="tmr_clip (default): pooled clip-level TMR embedding per segment "
                        "(perwin / ptw* per-token via short slices). tmr_frame: encode the "
                        "FULL window ONCE and read TMR's per-frame contextualized states "
                        "(final[:, nbtokens:]), then mean-pool a tmr-window (e.g. 160ms) "
                        "around each token -> in-distribution per-frame teacher. "
                        "Requires --per-token; emits [N, T_token, 256].")
    return p.parse_args(argv)

def _shard_range(n: int, num_shards: int, shard_index: int) -> tuple[int, int]:
    """Contiguous [lo, hi) split. Contiguous keeps each clip's windows in one shard."""
    per = (n + num_shards - 1) // num_shards
    lo = shard_index * per
    hi = min(n, lo + per)
    return lo, max(lo, hi)

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_root = Path(args.data_root) if args.data_root else default_data_root()
    umr_cache = resolve_local_uri(args.umr_cache, data_root)
    out_path = resolve_local_uri(args.out, data_root)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    meta = load_cache_meta(umr_cache)
    window = int(meta["window"])
    umr_fps = float(meta.get("fps", UMR_FPS))
    samples = meta["samples"]
    n_total = len(samples)
    n = n_total if args.max_windows is None else min(int(args.max_windows), n_total)

    ratio = TMR_FPS / umr_fps
    len30 = max(2, int(round(window * ratio)))

    per_token = bool(args.per_token)
    teacher = str(args.teacher)
    if teacher == "tmr_frame" and not per_token:
        raise ValueError("--teacher tmr_frame requires --per-token (it emits per-token targets).")
    block_mode = bool(args.tmr_block)
    if block_mode and not per_token:
        raise ValueError("--tmr-block requires --per-token (it emits per-token block-shared targets).")
    if block_mode and teacher != "tmr_clip":
        raise ValueError("--tmr-block only supports --teacher tmr_clip (the pooled clip-level path).")
    event_aligned = bool(args.event_aligned)
    if event_aligned:
        if not per_token:
            raise ValueError("--event-aligned requires --per-token.")
        if teacher != "tmr_clip":
            raise ValueError("--event-aligned only supports --teacher tmr_clip.")
        if block_mode:
            raise ValueError("--event-aligned is mutually exclusive with --tmr-block.")
        if not args.events_jsonl:
            raise ValueError("--event-aligned requires --events-jsonl.")
    token_stride = int(args.token_stride)
    tmr_window = int(args.tmr_window) if args.tmr_window else token_stride
    if per_token:
        if window % token_stride != 0:
            raise ValueError(f"window={window} not divisible by token-stride={token_stride}")
        t_token = window // token_stride
        wlen30 = max(2, int(round(tmr_window * ratio)))
        cache_shape: tuple[int, ...] = (n, t_token, DSEM)
    else:
        t_token = 0
        wlen30 = 0
        cache_shape = (n, DSEM)

    if args.create_only:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=cache_shape)
        mm[:] = 0.0
        mm.flush()
        del mm
        print(f"[tmr-cache] allocated {out_path} shape={cache_shape} (zero-filled)")
        return 0

    if args.write_meta:
        emb = np.asarray(np.load(out_path, mmap_mode="r"))
        reduce_axes = tuple(range(1, emb.ndim))
        n_failed = int((np.abs(emb).sum(axis=reduce_axes) == 0).sum())
        n_zero_tokens = (
            int((np.abs(emb).sum(axis=-1) == 0).sum()) if emb.ndim == 3 else n_failed
        )
        out_meta = {
            "schema_version": 1,
            "dsem": DSEM,
            "window": window,
            "umr_fps": umr_fps,
            "tmr_fps": TMR_FPS,
            "tmr_max_frames": TMR_MAX_FRAMES,
            "num_windows": int(n),
            "num_windows_total_in_umr": int(n_total),
            "n_failed_or_zero_rows": n_failed,
            "umr_manifest_hash": meta.get("manifest_hash"),
            "umr_cache": str(umr_cache),
            "granularity": "per_token" if per_token else "per_window_segment",
            "per_token": per_token,
            "event_aligned": bool(args.event_aligned),
            "events_jsonl": str(args.events_jsonl) if args.event_aligned else None,
            "tmr_block": block_mode if per_token else None,
            "tmr_block_overlap": 0 if (per_token and block_mode) else (
                (tmr_window - token_stride) if per_token else None),
            "token_stride": token_stride if per_token else None,
            "t_token": t_token if per_token else None,
            "tmr_window_frames": tmr_window if per_token else window,
            "tmr_window_frames_30fps": wlen30 if per_token else len30,
            "n_zero_tokens": n_zero_tokens if per_token else None,
            "joints_source": str(args.joints_source),
            "teacher": str(args.teacher),
            "source": "soma77_native_fk" if args.joints_source == "soma77" else "umr499_joints77_pos",
            "resample": ("polyphase_120to30_single" if args.joints_source == "soma77"
                         else "linear_50to30"),
            "canonicalization": "soma77_to_umr499_target_fps_30",
        }
        meta_out = cache_meta_path(out_path)
        tmp = meta_out.with_suffix(meta_out.suffix + ".tmp")
        tmp.write_text(json.dumps(out_meta, indent=2))
        tmp.replace(meta_out)
        print(f"[tmr-cache] wrote meta {meta_out}  zero_rows={n_failed}/{n}")
        return 0 if n_failed < n else 2

    lo, hi = _shard_range(n, max(1, int(args.num_shards)), int(args.shard_index))
    print(f"[tmr-cache] shard {args.shard_index}/{args.num_shards} range=[{lo}, {hi}) "
          f"of {n} windows (window={window}@{umr_fps}fps -> {len30}f@{TMR_FPS}fps), device={device}")

    by_clip: dict[str, list[int]] = {}
    for i in range(lo, hi):
        by_clip.setdefault(str(samples[i]["source_path"]), []).append(i)
    print(f"[tmr-cache] shard {args.shard_index}: unique clips = {len(by_clip)}")

    joints_source = str(args.joints_source)
    if joints_source == "soma77":
        from data.soma77_fk import soma77_joints_world_xyz
        from data.soma77_schema import Soma77Canonical
        from data.soma77_to_umr import soma77_to_umr499
    from tools.tmr_adapter_smoke import resample_linear_time

    tmr = load_tmr(args.teacher_dir, device=device)
    skel77 = load_soma_skeleton77()
    print(f"[tmr-cache] shard {args.shard_index}: joints_source={joints_source} "
          f"per_token={per_token} tmr_window={tmr_window} stride={token_stride} "
          f"event_aligned={event_aligned}")

    events_by_rec: dict[str, list[tuple[float, float]]] = {}
    if event_aligned:
        ev_path = resolve_local_uri(args.events_jsonl, data_root)
        with open(ev_path) as f:
            for line in f:
                d = json.loads(line)
                events_by_rec[str(d["rec_id"])] = [
                    (float(e["start_time"]), float(e["end_time"])) for e in d.get("events", [])
                ]
        print(f"[tmr-cache] shard {args.shard_index}: loaded {len(events_by_rec)} event tracks "
              f"from {ev_path}")

    if args.num_shards > 1 and not out_path.is_file():
        raise FileNotFoundError(
            f"{out_path} not allocated; run --create-only once before launching shard workers."
        )
    mode = "r+" if out_path.is_file() else "w+"
    if mode == "w+":
        out_path.parent.mkdir(parents=True, exist_ok=True)
    emb_cache = np.lib.format.open_memmap(
        out_path, mode=mode, dtype=np.float32, shape=cache_shape
    )

    def _encode(seg30: np.ndarray) -> np.ndarray:
        """TMR-encode a [t,77,3] @30fps segment -> [256] (pooled clip-level)."""
        posed = torch.from_numpy(np.ascontiguousarray(seg30)).float().to(device)[None]
        with torch.no_grad():
            e = tmr.encode_motion(posed_joints=posed, original_skeleton=skel77)
        return e[0].detach().cpu().numpy().astype(np.float32)

    def _encode_perframe(seg30: np.ndarray) -> np.ndarray:
        """TMR-encode a [t,77,3] @30fps segment -> per-frame states [T_feat, 256]."""
        posed = torch.from_numpy(np.ascontiguousarray(seg30)).float().to(device)[None]
        with torch.no_grad():
            pf = tmr.encode_motion_perframe(posed_joints=posed, original_skeleton=skel77)
        return pf[0].detach().cpu().numpy().astype(np.float32)

    def _pool_perframe_to_tokens(pf: np.ndarray) -> np.ndarray:
        """Map per-frame states [T_feat, 256] (spanning one `window`@50fps clip)
        to [t_token, 256] by mean-pooling a `tmr_window`(50fps) field, hop=token_stride, centred per token."""
        t_feat = pf.shape[0]
        scale = t_feat / max(1.0, float(len30))
        half30 = (tmr_window * ratio) / 2.0
        out = np.zeros((t_token, DSEM), np.float32)
        for lt in range(t_token):
            c30 = (lt * token_stride + token_stride / 2.0) * ratio
            a = int(max(0, round((c30 - half30) * scale)))
            b = int(min(t_feat, round((c30 + half30) * scale)))
            if b <= a:
                a = int(min(t_feat - 1, max(0, round(c30 * scale))))
                b = a + 1
            v = pf[a:b].mean(axis=0)
            nrm = float(np.linalg.norm(v))
            out[lt] = v / nrm if nrm > 1e-8 else 0.0
        return out

    done = 0
    n_failed = 0
    n_misaligned = 0
    failures: list[str] = []
    for umr_source, gidx in by_clip.items():
        seg_umr = None
        try:
            if joints_source == "soma77":
                soma_path = Path(umr_source).with_name(SOMA77_NPZ_NAME)
                canonical = Soma77Canonical.load(soma_path)
                joints120 = soma77_joints_world_xyz(soma_path, device=device)
                umr30 = soma77_to_umr499(canonical, joints77_world=joints120, target_fps=TMR_FPS)
                j30 = np.asarray(umr30.joints77_pos, dtype=np.float32)
                if j30.ndim != 3 or j30.shape[1] != 77:
                    raise ValueError(f"unexpected joints77_pos shape {j30.shape}")

                def seg_umr(a50: float, b50: float, _j30=j30):
                    a30 = max(0, int(round(a50 * ratio)))
                    b30 = min(_j30.shape[0], int(round(b50 * ratio)))
                    s = _j30[a30:b30][:TMR_MAX_FRAMES]
                    return _encode(s) if s.shape[0] >= 2 else None

                def win_perframe(a50: float, b50: float, _j30=j30):
                    a30 = max(0, int(round(a50 * ratio)))
                    b30 = min(_j30.shape[0], int(round(b50 * ratio)))
                    s = _j30[a30:b30][:TMR_MAX_FRAMES]
                    return _encode_perframe(s) if s.shape[0] >= 2 else None
            else:
                with np.load(umr_source) as rec:
                    j50 = np.asarray(rec["joints77_pos"], dtype=np.float32)
                    src_fps = float(rec["fps"]) if "fps" in rec else umr_fps
                if j50.ndim != 3 or j50.shape[1] != 77:
                    raise ValueError(f"unexpected joints77_pos shape {j50.shape}")

                def seg_umr(a50: float, b50: float, _j50=j50, _fps=src_fps):
                    a = max(0, int(round(a50)))
                    b = min(_j50.shape[0], int(round(b50)))
                    s = _j50[a:b]
                    if s.shape[0] < 2:
                        return None
                    s30 = resample_linear_time(s, _fps, TMR_FPS)[:TMR_MAX_FRAMES]
                    return _encode(s30) if s30.shape[0] >= 2 else None

                def win_perframe(a50: float, b50: float, _j50=j50, _fps=src_fps):
                    a = max(0, int(round(a50)))
                    b = min(_j50.shape[0], int(round(b50)))
                    s = _j50[a:b]
                    if s.shape[0] < 2:
                        return None
                    s30 = resample_linear_time(s, _fps, TMR_FPS)[:TMR_MAX_FRAMES]
                    return _encode_perframe(s30) if s30.shape[0] >= 2 else None
        except Exception as e:
            n_failed += len(gidx)
            if len(failures) < 32:
                failures.append(f"{umr_source}: {type(e).__name__}: {e}")
            for gi in gidx:
                emb_cache[gi] = 0.0
            done += len(gidx)
            continue

        if per_token and teacher == "tmr_frame":
            for gi in gidx:
                start = int(samples[gi]["start_frame"])
                pf = win_perframe(start, start + window)
                if pf is None or pf.shape[0] < 2:
                    emb_cache[gi] = 0.0
                    n_failed += 1
                    done += 1
                    continue
                emb_cache[gi] = _pool_perframe_to_tokens(pf)
                done += 1
                if done % int(args.log_every) == 0:
                    print(f"  shard {args.shard_index}: [{done}/{hi - lo}] windows (tmr_frame)")
        elif per_token:
            tok_emb: dict[object, np.ndarray | None] = {}
            rec_events = events_by_rec.get(Path(umr_source).parent.name, []) if event_aligned else []
            for gi in gidx:
                start = int(samples[gi]["start_frame"])
                if start % token_stride != 0:
                    n_misaligned += 1
                g0 = int(round(start / token_stride))
                for lt in range(t_token):
                    g = g0 + lt
                    if event_aligned:
                        t_sec = (g * token_stride + token_stride / 2.0) / umr_fps
                        ei = -1
                        for k, (a_s, b_s) in enumerate(rec_events):
                            if a_s <= t_sec < b_s:
                                ei = k
                                break
                        if ei < 0:
                            emb_cache[gi, lt] = 0.0
                            n_failed += 1
                            continue
                        key = ("evt", ei)
                        a50 = rec_events[ei][0] * umr_fps
                        b50 = rec_events[ei][1] * umr_fps
                    elif block_mode:
                        blk = (g * token_stride) // tmr_window
                        key: object = ("blk", blk)
                        a50 = float(blk * tmr_window)
                        b50 = a50 + tmr_window
                    else:
                        key = g
                        center50 = g * token_stride + token_stride / 2.0
                        a50 = center50 - tmr_window / 2.0
                        b50 = a50 + tmr_window
                    if key not in tok_emb:
                        tok_emb[key] = seg_umr(a50, b50)
                    e = tok_emb[key]
                    if e is None:
                        emb_cache[gi, lt] = 0.0
                        n_failed += 1
                    else:
                        emb_cache[gi, lt] = e
                done += 1
                if done % int(args.log_every) == 0:
                    print(f"  shard {args.shard_index}: [{done}/{hi - lo}] windows, "
                          f"{len(tok_emb)} unique tokens this clip")
        else:
            for gi in gidx:
                start = int(samples[gi]["start_frame"])
                e = seg_umr(start, start + window)
                if e is None:
                    emb_cache[gi] = 0.0
                    n_failed += 1
                    done += 1
                    continue
                emb_cache[gi] = e
                done += 1
                if done % int(args.log_every) == 0:
                    print(f"  shard {args.shard_index}: [{done}/{hi - lo}] done")

    emb_cache.flush()
    del emb_cache
    print(f"[tmr-cache] shard {args.shard_index} DONE: wrote {done} rows, failed={n_failed}, "
          f"misaligned_windows_snapped={n_misaligned}")
    if failures:
        for f in failures[:10]:
            print(f"  FAIL {f}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
