"""Build a training cache from a UMR-499 parquet export — variable-length OR fixed.

Both modes share ONE deterministic row plan (:func:`plan_rows`) so the perwin
TMR teacher (:mod:`tools.build_perwin_teacher_parquet`) re-derives the same plan
and stays row-aligned.

``--mode varlen``: whole clips clamped to ``[min_frames, max_frames]`` -> ragged
``<out>.npy`` fp32 ``[total_frames, 499]`` + ``.index.npy`` int64 ``[N_clips, 2]``
(offset, length) + ``.rec_ids.json`` + ``.meta.json``.

``--mode fixed``: sliding windows -> dense ``<out>.npy`` fp32 ``[N_windows,
window, 499]`` (same layout as :mod:`tools.build_training_cache`).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from data.umr_schema import DIM_FEATURES, FEATURE_VARIANT

def _floor_mult(x: int, m: int) -> int:
    return (x // m) * m

def _window_starts(n: int, window: int, step: int, cap: int | None) -> list[int]:
    """Sliding-window start indices for a clip with ``n`` records."""
    if n < window:
        return []
    last = n - window
    if step <= 0:
        return [0]
    starts = list(range(0, last + 1, step))
    if starts[-1] != last:
        starts.append(last)
    if cap is not None and cap > 0 and len(starts) > cap:
        starts = starts[:cap]
    return starts

def plan_rows(
    shards: list[str],
    *,
    mode: str,
    min_frames: int = 48,
    max_frames: int = 300,
    window: int = 64,
    step: int = 32,
    max_windows_per_clip: int | None = 64,
    token_stride: int = 4,
    max_clips: int | None = None,
) -> tuple[list[tuple[str, int, int, int]], list[str]]:
    """Deterministic per-row plan shared by the feature + teacher builders.

    Returns ``(rows, rec_ids)`` where each row is ``(shard_path, row_idx,
    start_frame, length)`` in write order, and ``rec_ids[i]`` is the source
    recording id of row ``i``.
    """
    rows: list[tuple[str, int, int, int]] = []
    rec_ids: list[str] = []
    for sh in shards:
        tbl = pq.read_table(sh, columns=["rec_id", "num_records"])
        recs = tbl.column("rec_id").to_pylist()
        nrec = tbl.column("num_records").to_pylist()
        for row, (rid, n) in enumerate(zip(recs, nrec)):
            n = int(n)
            if mode == "varlen":
                if n < min_frames:
                    continue
                take = _floor_mult(min(n, max_frames), token_stride)
                if take < min_frames:
                    continue
                rows.append((sh, row, 0, take))
                rec_ids.append(str(rid))
            elif mode == "fixed":
                for s in _window_starts(n, window, step, max_windows_per_clip):
                    rows.append((sh, row, int(s), int(window)))
                    rec_ids.append(str(rid))
            else:
                raise ValueError(f"unknown mode {mode!r}")
            if max_clips is not None and len(rows) >= max_clips:
                return rows, rec_ids
    return rows, rec_ids

def _iter_shard_flat(sh: str):
    """Yield (flat_float32_array, child_offsets) for a shard's features column."""
    ca = pq.read_table(sh, columns=["features"]).column("features").combine_chunks()
    flat = np.asarray(ca.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    child_off = ca.offsets.to_numpy()
    return flat, child_off

def _build_varlen(rows, rec_ids, out: Path, args) -> None:
    idx_path = out.with_suffix(out.suffix + ".index.npy")
    rec_path = out.with_suffix(out.suffix + ".rec_ids.json")
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    lens = np.array([r[3] for r in rows], dtype=np.int64)
    offsets = np.zeros(len(rows), dtype=np.int64)
    np.cumsum(lens[:-1], out=offsets[1:])
    total = int(lens.sum())
    gib = total * DIM_FEATURES * 4 / (1024**3)
    print(f"[cache] varlen: {len(rows)} clips, total_frames={total} (~{gib:.1f} GiB) "
          f"len[min/med/max]={lens.min()}/{int(np.median(lens))}/{lens.max()}")
    data = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32,
                                     shape=(total, int(DIM_FEATURES)))
    by_shard: dict[str, list[tuple[int, int, int]]] = {}
    for (sh, row, _s, take), off in zip(rows, offsets):
        by_shard.setdefault(sh, []).append((row, take, int(off)))
    shards = sorted(by_shard)
    done = 0
    for si, sh in enumerate(shards):
        flat, child_off = _iter_shard_flat(sh)
        for row, take, off in by_shard[sh]:
            s = int(child_off[row])
            data[off:off + take] = flat[s:s + take * DIM_FEATURES].reshape(take, DIM_FEATURES)
            done += 1
        del flat
        if (si + 1) % 10 == 0 or si + 1 == len(shards):
            print(f"[cache]   shard {si+1}/{len(shards)} clips_written={done}/{len(rows)}", flush=True)
    data.flush(); del data
    np.save(idx_path, np.stack([offsets, lens], axis=1))
    rec_path.write_text(json.dumps(rec_ids))
    meta = {
        "varlen": True, "feature_variant": FEATURE_VARIANT, "dtype": "float32",
        "fps": float(args.fps), "min_frames": int(args.min_frames),
        "max_frames": int(args.max_frames), "token_stride": int(args.token_stride),
        "n_clips": int(len(rows)), "total_frames": total,
        "shape": [total, int(DIM_FEATURES)], "index_path": idx_path.name,
        "rec_ids_path": rec_path.name, "source_parquet_dir": str(args.parquet_dir),
        "mode": "varlen",
    }
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2)); os.replace(tmp, meta_path)
    print(f"[cache] done varlen: {out} shape={meta['shape']}")

def _build_fixed(rows, rec_ids, out: Path, args) -> None:
    rec_path = out.with_suffix(out.suffix + ".rec_ids.json")
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    W = int(args.window)
    n = len(rows)
    gib = n * W * DIM_FEATURES * 4 / (1024**3)
    print(f"[cache] fixed: {n} windows window={W} step={args.step} (~{gib:.1f} GiB)")
    data = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32,
                                     shape=(n, W, int(DIM_FEATURES)))
    by_shard: dict[str, list[tuple[int, int, int]]] = {}
    for gi, (sh, row, start, length) in enumerate(rows):
        by_shard.setdefault(sh, []).append((row, start, gi))
    shards = sorted(by_shard)
    done = 0
    for si, sh in enumerate(shards):
        flat, child_off = _iter_shard_flat(sh)
        for row, start, gi in by_shard[sh]:
            s = int(child_off[row]) + start * DIM_FEATURES
            data[gi] = flat[s:s + W * DIM_FEATURES].reshape(W, DIM_FEATURES)
            done += 1
        del flat
        if (si + 1) % 10 == 0 or si + 1 == len(shards):
            print(f"[cache]   shard {si+1}/{len(shards)} windows_written={done}/{n}", flush=True)
    data.flush(); del data
    rec_path.write_text(json.dumps(rec_ids))
    meta = {
        "feature_variant": FEATURE_VARIANT, "dtype": "float32", "fps": float(args.fps),
        "window": W, "step": int(args.step),
        "max_windows_per_clip": int(args.max_windows_per_clip),
        "shape": [n, W, int(DIM_FEATURES)], "rec_ids_path": rec_path.name,
        "source_parquet_dir": str(args.parquet_dir), "mode": "fixed",
    }
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2)); os.replace(tmp, meta_path)
    print(f"[cache] done fixed: {out} shape={meta['shape']}")

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", choices=("varlen", "fixed"), default="varlen")
    ap.add_argument("--min-frames", type=int, default=48, help="varlen: drop clips shorter than this")
    ap.add_argument("--max-frames", type=int, default=300, help="varlen: clamp long clips")
    ap.add_argument("--window", type=int, default=64, help="fixed: window length")
    ap.add_argument("--step", type=int, default=32, help="fixed: sliding hop")
    ap.add_argument("--max-windows-per-clip", type=int, default=64, help="fixed: cap windows/clip")
    ap.add_argument("--token-stride", type=int, default=4)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--max-clips", type=int, default=None, help="smoke cap on rows")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    shards = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {args.parquet_dir}")
    out = args.out
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    if out.is_file() and meta_path.is_file() and not args.force:
        print(f"[cache] {out} exists; use --force to rebuild")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cache] pass 1/2: planning mode={args.mode} over {len(shards)} shards")
    rows, rec_ids = plan_rows(
        shards, mode=args.mode, min_frames=args.min_frames, max_frames=args.max_frames,
        window=args.window, step=args.step, max_windows_per_clip=args.max_windows_per_clip,
        token_stride=args.token_stride, max_clips=args.max_clips,
    )
    if not rows:
        raise RuntimeError("no rows survived planning")
    print(f"[cache] pass 2/2: streaming features -> {out}")
    if args.mode == "varlen":
        _build_varlen(rows, rec_ids, out, args)
    else:
        _build_fixed(rows, rec_ids, out, args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
