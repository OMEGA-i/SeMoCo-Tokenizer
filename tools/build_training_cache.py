"""Build a flat sliding-window training cache from a train manifest.

Writes a dense fp32 ``<out>.npy`` of shape ``[N_windows, window, 499]`` plus a
sibling ``.meta.json`` with per-window provenance (``samples=[{clip_id,
start_frame, source_path}, ...]``) and a ``manifest_hash``. Validation does not
use this cache (it keeps the on-the-fly :class:`data.umr_dataset.UMRDataset` path).

Usage::

    python -m tools.build_training_cache --manifest local://manifests/train.txt \
      --recordings-root local://recordings --out local://cache/umr499_train.npy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from data.local_uri import LOCAL_URI_PREFIX, default_data_root, resolve_local_uri
from data.umr_schema import (
    DIM_FEATURES,
    FEATURE_VARIANT,
    ROOT_MULTISCALE_HORIZONS_SEC,
    UMR_FPS,
    WINDOW_DEFAULT,
    root_multiscale_windows_for_fps,
)

DEFAULT_UMR499_FILENAME = "umr499.npz"

def _resolve_entry(entry: str, recordings_root: Path | None, data_root: Path) -> tuple[str, Path]:
    """Return ``(clip_id, npz_path)`` for one manifest line; ``clip_id`` is recorded
    in the meta so downstream tools can identify each window."""
    if entry.startswith(LOCAL_URI_PREFIX):
        p = resolve_local_uri(entry, data_root)
        clip_id = p.parent.name if p.name.endswith(".npz") else p.name
        return clip_id, p
    if entry.endswith(".npz") or "/" in entry:
        p = Path(entry)
        clip_id = p.parent.name if p.name.endswith(".npz") else p.name
        return clip_id, p
    if recordings_root is None:
        raise ValueError(
            f"manifest entry {entry!r} looks like a bare recording_id but "
            f"--recordings-root was not given"
        )
    return entry, recordings_root / entry / DEFAULT_UMR499_FILENAME

def _read_manifest(
    paths: list[Path], recordings_root: Path | None, data_root: Path
) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for mp in paths:
        for ln in mp.read_text().splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            out.append(_resolve_entry(s, recordings_root, data_root))
    return out

def _manifest_hash(entries: list[tuple[str, Path]]) -> str:
    h = hashlib.sha256()
    for clip_id, p in entries:
        h.update(clip_id.encode("utf-8"))
        h.update(b"\0")
        h.update(str(p).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

def _window_starts(num_records: int, window: int, step: int, cap: int | None) -> list[int]:
    """Sliding-window start indices for a clip with ``num_records`` records."""
    if num_records < window:
        return []
    last_start = num_records - window
    if step <= 0:
        return [0]
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    if cap is not None and cap > 0 and len(starts) > cap:
        starts = starts[:cap]
    return starts

@dataclass
class ClipPlan:
    clip_id: str
    source_path: str
    num_frames: int
    starts: list[int]
    error: Optional[str] = None

def _plan_one(entry: tuple[str, str], window: int, step: int, cap: int | None) -> ClipPlan:
    clip_id, source_path = entry
    try:
        with np.load(source_path, allow_pickle=False) as data:
            n = int(data["features"].shape[0])
    except Exception as e:
        tb = "".join(traceback.format_exception_only(type(e), e)).strip()
        return ClipPlan(clip_id=clip_id, source_path=source_path, num_frames=0, starts=[], error=tb)
    starts = _window_starts(n, window, step, cap)
    return ClipPlan(clip_id=clip_id, source_path=source_path, num_frames=n, starts=starts)

def _open_writer(out_path: Path, n_windows: int, window: int) -> np.ndarray:
    """Allocate a writable ``.npy`` with the final shape."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_windows, window, DIM_FEATURES),
    )

def _fill_clip_chunk(
    args: tuple[str, list[int], list[int], int, str]
) -> tuple[int, Optional[str]]:
    """Worker: load one clip and write its windows into the shared ``.npy``.

    Returns ``(num_windows_written, optional_error)``.
    """
    source_path, starts, global_indices, window, out_path = args
    try:
        with np.load(source_path, allow_pickle=False) as data:
            features = np.asarray(data["features"], dtype=np.float32)
    except Exception as e:                                  # noqa: BLE001
        tb = "".join(traceback.format_exception_only(type(e), e)).strip()
        return 0, f"{source_path}: {tb}"

    # Open the cache as a memmap in this worker; all workers share the file
    # under the OS page cache, so contiguous row writes serialize cleanly.
    writer = np.lib.format.open_memmap(out_path, mode="r+")
    written = 0
    for start, gi in zip(starts, global_indices):
        end = start + window
        if end > features.shape[0]:
            # Defensive: clip got shorter than planned. Skip these windows.
            continue
        writer[gi] = features[start:end]
        written += 1
    return written, None

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--manifest", action="append", required=True,
                   help="One or more manifest files (path or local:// URI). Train manifest only.")
    p.add_argument("--recordings-root", default=None,
                   help="Resolve bare recording ids to <root>/<rec_id>/umr499.npz. Accepts local:// URIs.")
    p.add_argument("--data-root", default=None,
                   help="Resolve local:// URIs against this root (default: $MOTIONVERSE_DATA_ROOT or ../omega-MotionVerse).")
    p.add_argument("--out", required=True,
                   help="Output .npy path (accepts local:// URIs). Sibling .meta.json is written alongside.")
    p.add_argument("--window", type=int, default=WINDOW_DEFAULT,
                   help=f"Feature frames (records) per window (default {WINDOW_DEFAULT}).")
    p.add_argument("--step", type=int, default=60,
                   help="Stride in frames between consecutive window starts (default 60).")
    p.add_argument("--max-windows-per-clip", type=int, default=64,
                   help="Cap on windows per clip; 0 disables the cap (default 64).")
    p.add_argument("--target-fps", type=float, default=UMR_FPS,
                   help=f"Expected UMR fps; recorded in meta (default {UMR_FPS}).")
    p.add_argument("--workers", type=int, default=1,
                   help="Worker processes for filling clips (1 = inline).")
    p.add_argument("--max-clips", type=int, default=None,
                   help="Process at most this many manifest entries (debug).")
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if the existing cache + meta match.")
    return p.parse_args(argv)

def _meta_path_for(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".meta.json")

def _existing_cache_matches(
    *,
    out_path: Path,
    meta_path: Path,
    window: int,
    step: int,
    max_windows_per_clip: int,
    manifest_hash: str,
) -> bool:
    if not out_path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    keys_ok = (
        int(meta.get("window", -1)) == int(window)
        and int(meta.get("step", -1)) == int(step)
        and int(meta.get("max_windows_per_clip", -1)) == int(max_windows_per_clip)
        and str(meta.get("manifest_hash", "")) == manifest_hash
        and str(meta.get("feature_variant", "")) == FEATURE_VARIANT
        and str(meta.get("dtype", "")) == "float32"
    )
    if not keys_ok:
        return False
    expected_shape = tuple(int(s) for s in meta.get("shape", []))
    if len(expected_shape) != 3 or expected_shape[1] != int(window) or expected_shape[2] != DIM_FEATURES:
        return False
    # Inspect the .npy header to confirm shape/dtype (the file has a variable
    # header so direct file-size comparison is unreliable).
    try:
        with open(out_path, "rb") as fh:
            version = np.lib.format.read_magic(fh)
            if version[0] == 1:
                shape, _fortran, dtype = np.lib.format.read_array_header_1_0(fh)
            else:
                shape, _fortran, dtype = np.lib.format.read_array_header_2_0(fh)
    except Exception:
        return False
    return dtype == np.dtype(np.float32) and tuple(int(s) for s in shape) == expected_shape

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_root = Path(args.data_root) if args.data_root else default_data_root()
    manifest_paths = [resolve_local_uri(p, data_root) for p in args.manifest]
    recordings_root = (
        resolve_local_uri(args.recordings_root, data_root) if args.recordings_root else None
    )
    out_path = resolve_local_uri(args.out, data_root)
    meta_path = _meta_path_for(out_path)

    window = int(args.window)
    if window < 1:
        print(f"window must be >= 1; got {window}", file=sys.stderr)
        return 1
    step = max(1, int(args.step))
    cap = int(args.max_windows_per_clip)
    cap_for_starts = cap if cap > 0 else None

    entries = _read_manifest(manifest_paths, recordings_root, data_root)
    if args.max_clips is not None:
        entries = entries[: args.max_clips]
    if not entries:
        print("no manifest entries", file=sys.stderr)
        return 1

    manifest_hash = _manifest_hash(entries)
    if not args.force and _existing_cache_matches(
        out_path=out_path,
        meta_path=meta_path,
        window=window,
        step=step,
        max_windows_per_clip=cap,
        manifest_hash=manifest_hash,
    ):
        print(f"build_training_cache: cache already valid at {out_path}; skip")
        return 0

    print(
        f"build_training_cache: planning windows for {len(entries)} clips "
        f"(window={window}, step={step}, cap={cap or 'unlimited'})"
    )
    plan_args = [((cid, str(p)), window, step, cap_for_starts) for cid, p in entries]
    plans: list[ClipPlan] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for i, args_tup in enumerate(plan_args, start=1):
            plans.append(_plan_one(*args_tup))
            if i % 5000 == 0:
                print(f"  planned {i}/{len(plan_args)}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_plan_one, *args_tup): args_tup[0][0] for args_tup in plan_args}
            done = 0
            for fut in as_completed(futures):
                plans.append(fut.result())
                done += 1
                if done % 5000 == 0:
                    print(f"  planned {done}/{len(plan_args)}")

    # Stable order matches the manifest order (makes the meta diffable).
    order = {cid: i for i, (cid, _) in enumerate(entries)}
    plans.sort(key=lambda p: order.get(p.clip_id, 1 << 30))

    samples: list[dict] = []
    plan_failures: list[ClipPlan] = []
    short_clips: list[str] = []
    for plan in plans:
        if plan.error is not None:
            plan_failures.append(plan)
            continue
        if not plan.starts:
            short_clips.append(plan.clip_id)
            continue
        for start in plan.starts:
            samples.append(
                {
                    "clip_id": plan.clip_id,
                    "source_path": plan.source_path,
                    "start_frame": int(start),
                }
            )
    n_windows = len(samples)
    if n_windows == 0:
        print("no windows planned (all clips short or failed)", file=sys.stderr)
        return 2

    gib = n_windows * window * DIM_FEATURES * 4 / (1024**3)
    print(
        f"build_training_cache: planned {n_windows} windows from "
        f"{len(plans) - len(plan_failures) - len(short_clips)} good clips "
        f"({len(plan_failures)} failed, {len(short_clips)} too short); "
        f"target size ≈ {gib:.1f} GiB"
    )

    # Pre-allocate so worker processes can mmap-write their assigned row ranges.
    writer = _open_writer(out_path, n_windows, window)
    writer.flush()
    del writer

    clip_jobs: list[tuple[str, list[int], list[int], int, str]] = []
    gi = 0
    out_path_s = str(out_path)
    for plan in plans:
        if plan.error is not None or not plan.starts:
            continue
        global_indices = list(range(gi, gi + len(plan.starts)))
        gi += len(plan.starts)
        clip_jobs.append(
            (plan.source_path, list(plan.starts), global_indices, window, out_path_s)
        )
    assert gi == n_windows, (gi, n_windows)

    fill_failures: list[str] = []
    written_windows = 0
    if workers == 1:
        for i, job in enumerate(clip_jobs, start=1):
            written, err = _fill_clip_chunk(job)
            if err is not None:
                fill_failures.append(err)
            written_windows += written
            if i % 2000 == 0:
                print(f"  filled {i}/{len(clip_jobs)} clips ({written_windows}/{n_windows} windows)")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fill_clip_chunk, job): job for job in clip_jobs}
            done = 0
            for fut in as_completed(futures):
                written, err = fut.result()
                if err is not None:
                    fill_failures.append(err)
                written_windows += written
                done += 1
                if done % 2000 == 0:
                    print(f"  filled {done}/{len(clip_jobs)} clips ({written_windows}/{n_windows} windows)")

    meta = {
        "schema_version": 1,
        "feature_variant": FEATURE_VARIANT,
        "fps": float(args.target_fps),
        "root_multiscale_horizons_sec": [float(v) for v in ROOT_MULTISCALE_HORIZONS_SEC],
        "root_multiscale_windows": [
            int(v) for v in root_multiscale_windows_for_fps(float(args.target_fps))
        ],
        "window": int(window),
        "step": int(step),
        "max_windows_per_clip": int(cap),
        "dtype": "float32",
        "shape": [int(n_windows), int(window), int(DIM_FEATURES)],
        "manifest_hash": manifest_hash,
        "manifest_paths": [str(p) for p in manifest_paths],
        "recordings_root": str(recordings_root) if recordings_root else None,
        "num_clips_total": len(entries),
        "num_clips_used": len(entries) - len(plan_failures) - len(short_clips),
        "num_clips_failed": len(plan_failures),
        "num_clips_short": len(short_clips),
        "num_windows": int(n_windows),
        "num_windows_written": int(written_windows),
        "samples": samples,
        "plan_failures": [
            {"clip_id": p.clip_id, "source_path": p.source_path, "error": p.error}
            for p in plan_failures[:64]
        ],
        "short_clips": short_clips[:64],
        "fill_failures": fill_failures[:64],
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2))
    os.replace(tmp_meta, meta_path)

    print(
        f"build_training_cache: wrote {out_path} "
        f"(shape={meta['shape']}, windows={n_windows}, "
        f"failed_clips={len(plan_failures)}, short_clips={len(short_clips)}, "
        f"fill_failures={len(fill_failures)})"
    )
    if plan_failures:
        for f in plan_failures[:20]:
            print(f"  PLAN FAIL {f.clip_id}: {f.error}", file=sys.stderr)
    if fill_failures:
        for msg in fill_failures[:20]:
            print(f"  FILL FAIL {msg}", file=sys.stderr)
    return 0 if written_windows == n_windows else 3

if __name__ == "__main__":
    sys.exit(main())
