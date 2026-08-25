"""Retrieval-teacher adapter smoke test on a single recording.

Pipeline: soma77.npz -> SOMA-X FK ([T, 77, 3] world joints) -> resample to 30
fps -> clip <= 300 frames -> ``TMR.encode_motion(posed_joints,
original_skeleton=SOMASkeleton77())`` -> 256-D unit embedding. The FK 77-joint
order must match kimodo's ``SOMASkeleton77`` bone order (the teacher slices its
30-joint subset by joint name). Requires the ``kimodo`` package and a frozen
teacher checkpoint with its Hydra ``config.yaml`` (README §External assets).

Run:
    python -m tools.tmr_adapter_smoke path/to/soma77.npz [--teacher-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

DEFAULT_TEACHER_DIR = Path("third_party/TMR-SOMA-RP-v1")
TMR_FPS = 30.0
TMR_MAX_FRAMES = 300


def load_tmr(teacher_dir: str | Path = DEFAULT_TEACHER_DIR, device: str = "cpu"):
    """Instantiate the frozen TMR model from its shipped Hydra config."""
    teacher_dir = Path(teacher_dir)
    if not (teacher_dir / "config.yaml").is_file():
        raise FileNotFoundError(
            f"retrieval teacher not found at {teacher_dir} (expected config.yaml inside). "
            f"Download the SOMA-adapted TMR checkpoint — see README §External assets — "
            f"or pass --teacher-dir."
        )
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(teacher_dir / "config.yaml"))
    cfg = OmegaConf.merge(cfg, {"checkpoint_dir": str(teacher_dir)})
    container = OmegaConf.to_container(cfg, resolve=True)
    container.pop("checkpoint_dir", None)  # interpolation-only key, not a TMR arg
    tmr = instantiate(OmegaConf.create(container), device=device)
    tmr.eval()
    return tmr


def load_soma_skeleton77():
    """Return kimodo's ``SOMASkeleton77`` with an actionable error if missing."""
    try:
        from kimodo.skeleton import SOMASkeleton77
    except ImportError as exc:
        raise ImportError(
            "the 'kimodo' package is required for the retrieval teacher "
            "(https://github.com/nv-tlabs/kimodo). Install it and retry."
        ) from exc
    return SOMASkeleton77()


def resample_linear_time(x: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Linear-interpolate a [T, ...] sequence from src_fps to dst_fps (verification-grade)."""
    if abs(src_fps - dst_fps) < 1e-6:
        return x
    t = x.shape[0]
    new_t = max(1, int(round(t * dst_fps / src_fps)))
    idx = np.linspace(0.0, t - 1.0, new_t)
    lo = np.floor(idx).astype(int)
    hi = np.ceil(idx).astype(int)
    w = (idx - lo).reshape((-1,) + (1,) * (x.ndim - 1))
    return (1.0 - w) * x[lo] + w * x[hi]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("recording", type=Path, help="path to a soma77.npz recording")
    p.add_argument("--teacher-dir", default=str(DEFAULT_TEACHER_DIR),
                   help="teacher checkpoint dir containing config.yaml (default: %(default)s)")
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rec = args.recording
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[tmr-smoke] recording = {rec}")
    print(f"[tmr-smoke] device    = {device}")

    from data.soma77_fk import soma77_joints_world_xyz

    meta = np.load(rec)
    src_fps = float(meta["fps"])
    print(f"[tmr-smoke] source fps = {src_fps}, frames = {meta['poses'].shape[0]}")

    joints = soma77_joints_world_xyz(rec, device=device)
    print(f"[tmr-smoke] FK joints  = {joints.shape} (expect [T, 77, 3])")
    assert joints.ndim == 3 and joints.shape[1] == 77, joints.shape

    joints30 = resample_linear_time(joints, src_fps, TMR_FPS)
    print(f"[tmr-smoke] @30fps     = {joints30.shape}")

    joints30 = joints30[:TMR_MAX_FRAMES]
    print(f"[tmr-smoke] clipped    = {joints30.shape}")

    # single sample, no lengths -> nbatch must be 1
    tmr = load_tmr(args.teacher_dir, device=device)
    skel77 = load_soma_skeleton77()
    posed = torch.from_numpy(np.ascontiguousarray(joints30)).float().to(device)[None]  # [1, T, 77, 3]
    emb = tmr.encode_motion(posed_joints=posed, original_skeleton=skel77)

    print("=" * 60)
    print(f"[tmr-smoke] embedding   = {tuple(emb.shape)} (expect [1, 256])")
    print(f"[tmr-smoke] L2 norm     = {float(emb.norm(dim=-1)[0]):.6f} (unit_vector=true -> ~1.0)")
    print(f"[tmr-smoke] any NaN/Inf = {bool(torch.isnan(emb).any() or torch.isinf(emb).any())}")
    print(f"[tmr-smoke] emb[:8]     = {emb[0, :8].tolist()}")
    ok = emb.shape == (1, 256) and not torch.isnan(emb).any()
    print("TMR ADAPTER SMOKE: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
