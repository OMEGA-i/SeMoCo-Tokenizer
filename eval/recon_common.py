"""Reconstruction metrics on a common joint set.

All metrics score the SMPL-22 body joints under per-frame pelvis alignment so
numbers are comparable across feature representations:

* ``mpjpe``   — pelvis-aligned mean per-joint position error (mm).
* ``pampjpe`` — Procrustes-aligned (per-frame rotation+scale+translation) MPJPE (mm).
* ``accel``   — 2nd-difference acceleration error vs GT (mm/frame^2), plus
  GT / reconstructed magnitude (jitter).
* ``jerk``    — 3rd-difference jerk error and magnitude (mm/frame^3).
* ``ape``/``ave`` — average position / velocity error, split into ``root``,
  ``traj`` (pelvis XZ), ``pose`` (root-relative), ``joints``.

Inputs are ``[T, J, 3]`` in metres; outputs in millimetres. Acceleration/jerk
are frame-rate dependent: callers MUST resample to a common fps first.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# SMPL-22 body joints as a subset of the SOMA-77 FK joint output, pelvis
# (index 0) first; the last two SMPL-24 entries are hand tips duplicating wrists.
from data.soma77_fk import SMPL24_TO_SOMA77_INDEX

SMPL22_SOMA77_INDEX: tuple[int, ...] = tuple(SMPL24_TO_SOMA77_INDEX[:22])
"""SOMA-77 joint indices for the 22 SMPL body joints, pelvis first."""

PELVIS = 0

def soma77_to_smpl22(joints77: np.ndarray) -> np.ndarray:
    """``[T, 77, 3]`` SOMA joints → ``[T, 22, 3]`` SMPL-22 subset."""
    if joints77.shape[-2] < 77:
        raise ValueError(f"expected >=77 joints, got {joints77.shape}")
    return joints77[..., SMPL22_SOMA77_INDEX, :]

def _align_pelvis(joints: np.ndarray) -> np.ndarray:
    """Subtract the pelvis (joint 0) from every joint, per frame. ``[T,J,3]``."""
    return joints - joints[:, PELVIS : PELVIS + 1, :]

def _procrustes_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-frame similarity transform (sR, t) mapping ``pred [T, J, 3]`` onto ``target``."""
    # Work in [T, 3, J].
    S1 = np.transpose(pred, (0, 2, 1)).astype(np.float64)
    S2 = np.transpose(target, (0, 2, 1)).astype(np.float64)

    mu1 = S1.mean(axis=2, keepdims=True)
    mu2 = S2.mean(axis=2, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = (X1**2).sum(axis=(1, 2))  # [T]

    K = X1 @ np.transpose(X2, (0, 2, 1))  # [T,3,3]
    U, s, Vh = np.linalg.svd(K)
    V = np.transpose(Vh, (0, 2, 1))
    Ut = np.transpose(U, (0, 2, 1))

    Z = np.tile(np.eye(3)[None], (S1.shape[0], 1, 1))
    detR = np.linalg.det(V @ Ut)
    Z[:, -1, -1] = np.sign(detR)

    R = V @ (Z @ Ut)  # [T,3,3]
    scale = (np.einsum("tii->t", (R @ K))) / np.maximum(var1, 1e-12)  # trace / var1
    t = mu2 - scale[:, None, None] * (R @ mu1)

    S1_hat = scale[:, None, None] * (R @ S1) + t  # [T,3,J]
    return np.transpose(S1_hat, (0, 2, 1)).astype(np.float32)

# ---------------------------------------------------------------------------
# Core metrics — all take pred/target ``[T, J, 3]`` in metres, return mm
# ---------------------------------------------------------------------------

_MM = 1000.0

def mpjpe_mm(pred: np.ndarray, target: np.ndarray) -> float:
    """Per-frame pelvis-aligned MPJPE in mm (baseline-standard)."""
    p = _align_pelvis(pred)
    g = _align_pelvis(target)
    return float(np.linalg.norm(p - g, axis=-1).mean() * _MM)

def pampjpe_mm(pred: np.ndarray, target: np.ndarray) -> float:
    """Procrustes-aligned MPJPE in mm."""
    p = _procrustes_align(pred, target)
    return float(np.linalg.norm(p - target, axis=-1).mean() * _MM)

def accel_metrics_mm(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Acceleration (2nd diff) error + magnitudes, pelvis-aligned, mm/frame^2."""
    if pred.shape[0] < 3:
        return {"accel_err": float("nan"), "accel_pred": float("nan"), "accel_gt": float("nan")}
    p = _align_pelvis(pred)
    g = _align_pelvis(target)
    acc_p = p[:-2] - 2 * p[1:-1] + p[2:]
    acc_g = g[:-2] - 2 * g[1:-1] + g[2:]
    err = np.linalg.norm(acc_p - acc_g, axis=-1).mean() * _MM
    mag_p = np.linalg.norm(acc_p, axis=-1).mean() * _MM
    mag_g = np.linalg.norm(acc_g, axis=-1).mean() * _MM
    return {"accel_err": float(err), "accel_pred": float(mag_p), "accel_gt": float(mag_g)}

def jerk_metrics_mm(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Jerk (3rd diff) error + magnitudes, pelvis-aligned, mm/frame^3."""
    if pred.shape[0] < 4:
        return {"jerk_err": float("nan"), "jerk_pred": float("nan"), "jerk_gt": float("nan")}
    p = _align_pelvis(pred)
    g = _align_pelvis(target)
    jp = p[3:] - 3 * p[2:-1] + 3 * p[1:-2] - p[:-3]
    jg = g[3:] - 3 * g[2:-1] + 3 * g[1:-2] - g[:-3]
    err = np.linalg.norm(jp - jg, axis=-1).mean() * _MM
    mag_p = np.linalg.norm(jp, axis=-1).mean() * _MM
    mag_g = np.linalg.norm(jg, axis=-1).mean() * _MM
    return {"jerk_err": float(err), "jerk_pred": float(mag_p), "jerk_gt": float(mag_g)}

def ape_ave_mm(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """APE/AVE split into root / traj / pose / joints (mm)."""
    ape_joints = np.linalg.norm(pred - target, axis=-1).mean(axis=0)
    ape_root = float(np.linalg.norm(pred[:, PELVIS] - target[:, PELVIS], axis=-1).mean() * _MM)
    ape_traj = float(
        np.linalg.norm(
            pred[:, PELVIS][:, [0, 2]] - target[:, PELVIS][:, [0, 2]], axis=-1
        ).mean()
        * _MM
    )
    p_loc = _align_pelvis(pred)
    g_loc = _align_pelvis(target)
    ape_pose = float(np.linalg.norm(p_loc - g_loc, axis=-1).mean() * _MM)
    ape_joints_v = float(ape_joints.mean() * _MM)

    pv = pred[1:] - pred[:-1]
    gv = target[1:] - target[:-1]
    ave_root = float(np.linalg.norm(pv[:, PELVIS] - gv[:, PELVIS], axis=-1).mean() * _MM)
    ave_traj = float(
        np.linalg.norm(pv[:, PELVIS][:, [0, 2]] - gv[:, PELVIS][:, [0, 2]], axis=-1).mean() * _MM
    )
    pv_loc = p_loc[1:] - p_loc[:-1]
    gv_loc = g_loc[1:] - g_loc[:-1]
    ave_pose = float(np.linalg.norm(pv_loc - gv_loc, axis=-1).mean() * _MM)
    ave_joints = float(np.linalg.norm(pv - gv, axis=-1).mean() * _MM)
    return {
        "ape_root": ape_root,
        "ape_traj": ape_traj,
        "ape_pose": ape_pose,
        "ape_joints": ape_joints_v,
        "ave_root": ave_root,
        "ave_traj": ave_traj,
        "ave_pose": ave_pose,
        "ave_joints": ave_joints,
    }

@dataclass
class ReconMetrics:
    mpjpe: float
    pampjpe: float
    accel_err: float
    accel_pred: float
    accel_gt: float
    jerk_err: float
    jerk_pred: float
    jerk_gt: float
    ape_root: float
    ape_traj: float
    ape_pose: float
    ape_joints: float
    ave_root: float
    ave_traj: float
    ave_pose: float
    ave_joints: float
    n_frames: int

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

def recon_metrics(pred: np.ndarray, target: np.ndarray) -> ReconMetrics:
    """All reconstruction metrics for one clip (``pred``/``target`` ``[T,J,3]``, mm).

    Sequences are aligned to the shorter length.
    """
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if pred.shape[-2:] != target.shape[-2:]:
        raise ValueError(f"joint/dim mismatch: {pred.shape} vs {target.shape}")
    T = min(pred.shape[0], target.shape[0])
    pred, target = pred[:T], target[:T]

    acc = accel_metrics_mm(pred, target)
    jrk = jerk_metrics_mm(pred, target)
    apv = ape_ave_mm(pred, target)
    return ReconMetrics(
        mpjpe=mpjpe_mm(pred, target),
        pampjpe=pampjpe_mm(pred, target),
        n_frames=int(T),
        **acc,
        **jrk,
        **apv,
    )

__all__ = [
    "SMPL22_SOMA77_INDEX",
    "soma77_to_smpl22",
    "mpjpe_mm",
    "pampjpe_mm",
    "accel_metrics_mm",
    "jerk_metrics_mm",
    "ape_ave_mm",
    "ReconMetrics",
    "recon_metrics",
]
