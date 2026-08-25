"""Offline conversion ``soma77.npz`` → ``umr499.npz``.

Materializes the UMR cache for each manifest entry: load
:class:`~data.soma77_schema.Soma77Canonical`, run live SOMA-X FK to
``[T_src, 77, 3]`` world joints, convert via
:func:`data.soma77_to_umr.soma77_to_umr499` at ``--target-fps`` (default 50),
and save ``umr499.npz`` next to the input (skipped unless ``--force``).

Example::

    python -m tools.build_umr_dataset --manifest motions/manifests/train.txt \
      --recordings-root motions/raw/recordings --target-fps 50 --workers 4
"""

from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from data.local_uri import default_data_root, resolve_local_uri
from data.soma77_schema import Soma77Canonical
from data.soma77_to_umr import soma77_to_umr499
from data.umr_schema import UMR_FPS

@dataclass
class ClipResult:
    soma77_path: str
    out_path: Optional[str]
    skipped: bool
    error: Optional[str]

def _resolve_entry(entry: str, recordings_root: Optional[Path], data_root: Path) -> Path:
    if entry.startswith("local://"):
        return resolve_local_uri(entry, data_root)
    if entry.endswith(".npz") or "/" in entry:
        return Path(entry)
    if recordings_root is None:
        raise ValueError(
            f"manifest entry {entry!r} looks like a bare recording_id but "
            f"--recordings-root was not given"
        )
    return recordings_root / entry / "soma77.npz"

def _fk_joints77(
    soma77_path: Path,
    *,
    num_frames: int,
    device: str,
) -> np.ndarray:
    """Return ``[T_src, 77, 3]`` FK joints via live SOMA-X FK."""
    from data.soma77_fk import soma77_joints_world_xyz

    joints = soma77_joints_world_xyz(soma77_path, device=device)
    if joints.shape != (num_frames, 77, 3):
        raise ValueError(
            f"live FK returned shape {joints.shape}; expected ({num_frames}, 77, 3)"
        )
    return joints

def _convert_one(
    soma77_path: str,
    *,
    target_fps: float,
    device: str,
    force: bool,
) -> ClipResult:
    in_path = Path(soma77_path)
    out_path = in_path.with_name("umr499.npz")
    try:
        if out_path.is_file() and not force:
            return ClipResult(soma77_path=str(in_path), out_path=str(out_path), skipped=True, error=None)

        canonical = Soma77Canonical.load(in_path)
        joints77 = _fk_joints77(
            in_path,
            num_frames=canonical.num_frames,
            device=device,
        )
        umr = soma77_to_umr499(
            canonical,
            joints77_world=joints77,
            target_fps=target_fps,
        )
        umr.to_npz(out_path)
        return ClipResult(soma77_path=str(in_path), out_path=str(out_path), skipped=False, error=None)
    except Exception as e:
        tb = "".join(traceback.format_exception_only(type(e), e)).strip()
        return ClipResult(soma77_path=str(in_path), out_path=None, skipped=False, error=tb)

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--manifest", action="append", required=True,
                   help="One or more manifest files (path or local:// URI).")
    p.add_argument("--recordings-root", default=None,
                   help="Resolve bare recording ids to <root>/<rec_id>/soma77.npz. "
                        "Accepts local:// URIs.")
    p.add_argument("--data-root", default=None,
                   help="Resolve local:// URIs against this root (default: $MOTIONVERSE_DATA_ROOT or ../omega-MotionVerse).")
    p.add_argument("--target-fps", type=float, default=UMR_FPS,
                   help=f"Output UMR fps (default {UMR_FPS}).")
    p.add_argument("--device", default="cuda", help="SOMA-X FK device (cuda / cpu).")
    p.add_argument("--workers", type=int, default=1, help="Worker processes (1 = inline).")
    p.add_argument("--force", action="store_true",
                   help="Re-build even if umr499.npz already exists.")
    p.add_argument("--max-clips", type=int, default=None,
                   help="Process at most this many manifest entries (debug).")
    return p.parse_args(argv)

def _read_manifest(paths: list[Path], recordings_root: Optional[Path], data_root: Path) -> list[Path]:
    out: list[Path] = []
    for mp in paths:
        for ln in mp.read_text().splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            out.append(_resolve_entry(s, recordings_root, data_root))
    return out

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_root = Path(args.data_root) if args.data_root else default_data_root()
    manifest_paths = [resolve_local_uri(p, data_root) for p in args.manifest]
    recordings_root = (
        resolve_local_uri(args.recordings_root, data_root) if args.recordings_root else None
    )
    entries = _read_manifest(manifest_paths, recordings_root, data_root)
    if args.max_clips is not None:
        entries = entries[: args.max_clips]
    if not entries:
        print("no manifest entries", file=sys.stderr)
        return 1

    failures: list[ClipResult] = []
    written = 0
    skipped = 0

    if args.workers <= 1:
        for path in entries:
            res = _convert_one(
                str(path),
                target_fps=args.target_fps,
                device=args.device,
                force=args.force,
            )
            if res.error:
                failures.append(res)
            elif res.skipped:
                skipped += 1
            else:
                written += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    _convert_one,
                    str(path),
                    target_fps=args.target_fps,
                    device=args.device,
                    force=args.force,
                )
                for path in entries
            ]
            for fut in as_completed(futures):
                res = fut.result()
                if res.error:
                    failures.append(res)
                elif res.skipped:
                    skipped += 1
                else:
                    written += 1

    print(
        f"build_umr_dataset: written={written} skipped={skipped} "
        f"failures={len(failures)} total={len(entries)}"
    )
    if failures:
        for r in failures[:20]:
            print(f"  FAIL {r.soma77_path}: {r.error}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more failures", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
