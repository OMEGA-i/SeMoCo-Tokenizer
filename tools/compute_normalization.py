"""Compute UMR normalization stats over a train manifest.

Walks a manifest of ``umr499.npz`` files and produces ``norm_stats.json``
(see :class:`data.normalization.NormalizationStats`) for the trainer.

Usage::

    python -m tools.compute_normalization --manifest motions/manifests/train.txt \
      --recordings-root motions/raw/recordings --out motions/cache/norm_stats.json
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np

from data.local_uri import default_data_root, resolve_local_uri
from data.normalization import NormalizationStats, NormalizationStatsBuilder
from data.umr_schema import (
    DIM_FEATURES,
    DIM_FOOT_CONTACT,
    NUM_SPARSE_VEL_JOINTS,
    SLICE_FOOT_CONTACT,
    SLICE_JOINTS76_ROT6D,
    SLICE_ROOT_ROT6D,
    SLICE_ROOT_TRAJ,
    SLICE_SPARSE_VEL,
    UMR499,
    UMR_FPS,
    UMR_NUM_JOINTS76,
)

def _resolve(entry: str, recordings_root: Path | None, data_root: Path) -> Path:
    if entry.startswith("local://"):
        return resolve_local_uri(entry, data_root)
    if entry.endswith(".npz") or "/" in entry:
        return Path(entry)
    if recordings_root is None:
        raise ValueError(
            f"manifest entry {entry!r} requires --recordings-root"
        )
    return recordings_root / entry / "umr499.npz"

def _read_manifest(paths: Iterable[Path], recordings_root: Path | None, data_root: Path) -> list[Path]:
    out: list[Path] = []
    for mp in paths:
        for ln in mp.read_text().splitlines():
            s = ln.strip()
            if s and not s.startswith("#"):
                out.append(_resolve(s, recordings_root, data_root))
    return out

def _chunked(seq: list[Path], n: int) -> list[list[Path]]:
    if n <= 1:
        return [seq]
    return [seq[i::n] for i in range(n)]

def _partial_stats(paths_s: list[str]) -> dict:
    """Compute partial raw sums in one worker process."""
    count = 0
    num_clips = 0
    root_sum = np.zeros((9,), dtype=np.float64)
    root_sumsq = np.zeros((9,), dtype=np.float64)
    root_p95_norms: list[float] = []
    root_rot_sum = np.zeros((6,), dtype=np.float64)
    root_rot_sumsq = np.zeros((6,), dtype=np.float64)
    joints_sum = np.zeros((UMR_NUM_JOINTS76, 6), dtype=np.float64)
    joints_sumsq = np.zeros((UMR_NUM_JOINTS76, 6), dtype=np.float64)
    sparse_sum = np.zeros((NUM_SPARSE_VEL_JOINTS, 3), dtype=np.float64)
    sparse_sumsq = np.zeros((NUM_SPARSE_VEL_JOINTS, 3), dtype=np.float64)
    contact_sum = np.zeros((DIM_FOOT_CONTACT,), dtype=np.float64)
    packed_sum = np.zeros((DIM_FEATURES,), dtype=np.float64)
    packed_sumsq = np.zeros((DIM_FEATURES,), dtype=np.float64)
    p01_buf: list[np.ndarray] = []
    p99_buf: list[np.ndarray] = []
    skipped: list[str] = []

    for path_s in paths_s:
        try:
            umr = UMR499.from_npz(path_s)
        except Exception as e:
            skipped.append(f"{path_s}: {e}")
            continue
        x = np.asarray(umr.features, dtype=np.float64)
        if x.ndim != 2 or x.shape[-1] != DIM_FEATURES or x.shape[0] == 0:
            skipped.append(f"{path_s}: bad features shape {x.shape}")
            continue
        n = x.shape[0]
        count += n
        num_clips += 1
        root = x[:, SLICE_ROOT_TRAJ]
        root_sum += root.sum(axis=0)
        root_sumsq += (root * root).sum(axis=0)
        root_p95_norms.append(float(np.quantile(np.linalg.norm(root[:, 0:2], axis=-1), 0.95)))

        rr = x[:, SLICE_ROOT_ROT6D]
        root_rot_sum += rr.sum(axis=0)
        root_rot_sumsq += (rr * rr).sum(axis=0)

        joints = x[:, SLICE_JOINTS76_ROT6D].reshape(n, UMR_NUM_JOINTS76, 6)
        joints_sum += joints.sum(axis=0)
        joints_sumsq += (joints * joints).sum(axis=0)

        sparse = x[:, SLICE_SPARSE_VEL].reshape(n, NUM_SPARSE_VEL_JOINTS, 3)
        sparse_sum += sparse.sum(axis=0)
        sparse_sumsq += (sparse * sparse).sum(axis=0)

        contact_sum += x[:, SLICE_FOOT_CONTACT].sum(axis=0)
        packed_sum += x.sum(axis=0)
        packed_sumsq += (x * x).sum(axis=0)
        p01_buf.append(np.quantile(x, 0.01, axis=0))
        p99_buf.append(np.quantile(x, 0.99, axis=0))

    return {
        "count": count,
        "num_clips": num_clips,
        "root_sum": root_sum,
        "root_sumsq": root_sumsq,
        "root_p95_norms": root_p95_norms,
        "root_rot_sum": root_rot_sum,
        "root_rot_sumsq": root_rot_sumsq,
        "joints_sum": joints_sum,
        "joints_sumsq": joints_sumsq,
        "sparse_sum": sparse_sum,
        "sparse_sumsq": sparse_sumsq,
        "contact_sum": contact_sum,
        "packed_sum": packed_sum,
        "packed_sumsq": packed_sumsq,
        "p01_sum": np.stack(p01_buf).sum(axis=0) if p01_buf else np.zeros((DIM_FEATURES,), dtype=np.float64),
        "p99_sum": np.stack(p99_buf).sum(axis=0) if p99_buf else np.zeros((DIM_FEATURES,), dtype=np.float64),
        "quantile_clips": len(p01_buf),
        "skipped": skipped[:20],
    }

def _mean_std(sum_: np.ndarray, sumsq: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    mean = sum_ / max(count, 1)
    var = np.maximum(sumsq / max(count, 1) - mean * mean, 0.0)
    return mean, np.sqrt(var + 1e-8)

def _finalize_parallel(parts: list[dict], *, fps: float) -> NormalizationStats:
    count = sum(p["count"] for p in parts)
    if count == 0:
        raise ValueError("no records contributed to stats")
    num_clips = sum(p["num_clips"] for p in parts)

    root_mean, root_std = _mean_std(sum(p["root_sum"] for p in parts), sum(p["root_sumsq"] for p in parts), count)
    root_rot_mean, root_rot_std = _mean_std(sum(p["root_rot_sum"] for p in parts), sum(p["root_rot_sumsq"] for p in parts), count)
    joints_mean, joints_std = _mean_std(sum(p["joints_sum"] for p in parts), sum(p["joints_sumsq"] for p in parts), count)
    sparse_mean, sparse_std = _mean_std(sum(p["sparse_sum"] for p in parts), sum(p["sparse_sumsq"] for p in parts), count)
    packed_mean, packed_std = _mean_std(sum(p["packed_sum"] for p in parts), sum(p["packed_sumsq"] for p in parts), count)
    contact_rate = sum(p["contact_sum"] for p in parts) / count

    quantile_clips = sum(p["quantile_clips"] for p in parts)
    p01 = sum(p["p01_sum"] for p in parts) / max(quantile_clips, 1)
    p99 = sum(p["p99_sum"] for p in parts) / max(quantile_clips, 1)
    p95_norms = [v for p in parts for v in p["root_p95_norms"]]

    return NormalizationStats(
        fps=float(fps),
        num_clips=int(num_clips),
        num_records=int(count),
        root_traj_mean=root_mean.astype(float).tolist(),
        root_traj_std=root_std.astype(float).tolist(),
        root_traj_p95_norm=float(np.mean(p95_norms)) if p95_norms else 0.0,
        root_rot6d_mean=root_rot_mean.astype(float).tolist(),
        root_rot6d_std=root_rot_std.astype(float).tolist(),
        joints76_rot6d_mean=joints_mean.astype(float).tolist(),
        joints76_rot6d_std=joints_std.astype(float).tolist(),
        sparse_vel_mean=sparse_mean.astype(float).tolist(),
        sparse_vel_std=sparse_std.astype(float).tolist(),
        foot_contact_positive_rate=contact_rate.astype(float).tolist(),
        packed_mean=packed_mean.astype(float).tolist(),
        packed_std=packed_std.astype(float).tolist(),
        packed_clip_p01=p01.astype(float).tolist(),
        packed_clip_p99=p99.astype(float).tolist(),
    )

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--manifest", action="append", required=True,
                   help="One or more manifest files (path or local:// URI).")
    p.add_argument("--recordings-root", default=None,
                   help="Accepts local:// URIs.")
    p.add_argument("--data-root", default=None,
                   help="Resolve local:// URIs against this root (default: $MOTIONVERSE_DATA_ROOT or ../omega-MotionVerse).")
    p.add_argument("--out", required=True,
                   help="Output JSON path (accepts local:// URIs).")
    p.add_argument("--fps", type=float, default=UMR_FPS)
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args(argv)

    data_root = Path(args.data_root) if args.data_root else default_data_root()
    manifest_paths = [resolve_local_uri(m, data_root) for m in args.manifest]
    recordings_root = (
        resolve_local_uri(args.recordings_root, data_root) if args.recordings_root else None
    )
    out_path = resolve_local_uri(args.out, data_root)
    entries = _read_manifest(manifest_paths, recordings_root, data_root)
    if args.max_clips is not None:
        entries = entries[: args.max_clips]
    if not entries:
        print("no manifest entries", file=sys.stderr)
        return 1

    workers = max(1, int(args.workers))
    if workers == 1:
        builder = NormalizationStatsBuilder(fps=args.fps)
        n_ok = 0
        for path in entries:
            try:
                umr = UMR499.from_npz(path)
            except Exception as e:
                print(f"SKIP {path}: {e}", file=sys.stderr)
                continue
            if abs(umr.fps - args.fps) > 1e-3:
                print(f"SKIP {path}: fps={umr.fps} != {args.fps}", file=sys.stderr)
                continue
            builder.update(umr.features)
            n_ok += 1
        if n_ok == 0:
            print("no clips contributed to stats", file=sys.stderr)
            return 2
        stats = builder.finalize()
    else:
        chunks = _chunked(entries, workers)
        parts: list[dict] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_partial_stats, [str(p) for p in chunk]) for chunk in chunks if chunk]
            for i, fut in enumerate(as_completed(futures), start=1):
                part = fut.result()
                parts.append(part)
                print(f"partial {i}/{len(futures)} clips={part['num_clips']} records={part['count']}")
                for msg in part.get("skipped", []):
                    print(f"SKIP {msg}", file=sys.stderr)
        stats = _finalize_parallel(parts, fps=args.fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats.write_json(out_path)
    print(
        f"compute_normalization: wrote {out_path} num_clips={stats.num_clips} "
        f"num_records={stats.num_records}"
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
