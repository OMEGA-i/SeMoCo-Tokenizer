"""L0 sanity checks for SOMA77 canonical inputs and UMR artifacts.

Cheap, hard-fail validators run at the data boundary so downstream layers can
assume invariants: :func:`validate_soma77_canonical` on encoder inputs,
:func:`validate_umr499` on the on-disk artifact contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from data.soma77_schema import Soma77Canonical
from data.umr_schema import (
    DIM_FEATURES,
    DIM_ROOT_ROT6D,
    FEATURE_VARIANT,
    UMR_FPS,
    UMR_NUM_JOINTS,
    UMR_NUM_JOINTS76,
    UMR499,
)

@dataclass
class ValidationReport:
    """Outcome of an L0 sanity pass."""

    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

def _check_finite(arr: np.ndarray, name: str, report: ValidationReport) -> bool:
    if not np.isfinite(arr).all():
        nan = int(np.isnan(arr).sum())
        inf = int(np.isinf(arr).sum())
        report.fail(f"{name}: NaN={nan} Inf={inf} (total elements: {arr.size})")
        return False
    return True

def _check_axis_angle_norms(aa: np.ndarray, name: str, report: ValidationReport, *, max_norm: float = 6.5) -> None:
    if aa.size == 0:
        return
    norms = np.linalg.norm(aa, axis=-1)
    bad = int((norms > max_norm).sum())
    if bad > 0:
        report.warn(
            f"{name}: {bad} axis-angle samples exceed |aa|={max_norm:.2f} "
            f"(max={float(norms.max()):.3f})"
        )
    report.stats[f"{name}__aa_norm_p99"] = float(np.quantile(norms, 0.99))

def _check_foot_contact_distribution(fc: np.ndarray, report: ValidationReport) -> None:
    if fc.size == 0:
        return
    pos_rate = fc.mean(axis=0)
    for i, label in enumerate(("LFoot", "LToe", "RFoot", "RToe")):
        report.stats[f"foot_contact__{label}__positive_rate"] = float(pos_rate[i])
    if (pos_rate < 1e-6).all():
        report.fail("foot_contacts: all-zero across the clip")

def validate_soma77_canonical(canonical: Soma77Canonical) -> ValidationReport:
    """Run L0 sanity on a SOMA77 canonical payload."""
    report = ValidationReport()
    try:
        canonical.validate_shapes()
    except ValueError as e:
        report.fail(f"shape: {e}")
        return report

    _check_finite(canonical.poses, "poses", report)
    _check_finite(canonical.transl, "transl", report)
    _check_finite(canonical.foot_contacts, "foot_contacts", report)
    _check_axis_angle_norms(canonical.poses, "poses", report)
    _check_foot_contact_distribution(canonical.foot_contacts, report)

    y = canonical.transl[:, 1]
    report.stats["transl__y_mean"] = float(y.mean())
    report.stats["transl__y_min"] = float(y.min())
    report.stats["transl__y_max"] = float(y.max())

    if canonical.transl.shape[0] >= 2:
        delta = canonical.transl[1:] - canonical.transl[:-1]
        speed = np.linalg.norm(delta, axis=-1) * float(canonical.fps_src)
        report.stats["root_speed__mps_p99"] = float(np.quantile(speed, 0.99))

    if canonical.fps_src < 5.0 or canonical.fps_src > 240.0:
        report.warn(
            f"fps_src = {canonical.fps_src:.1f} outside expected [5, 240]; "
            f"likely a manifest.json mis-read"
        )

    report.stats["num_frames"] = float(canonical.num_frames)
    report.stats["fps_src"] = float(canonical.fps_src)
    return report

def validate_umr499(umr: UMR499) -> ValidationReport:
    """Run L0 sanity on a :class:`UMR499` artifact."""
    report = ValidationReport()

    if umr.feature_variant != FEATURE_VARIANT:
        report.fail(
            f"feature_variant {umr.feature_variant!r} != expected {FEATURE_VARIANT!r}"
        )
    if abs(umr.fps - UMR_FPS) > 1e-3:
        report.fail(f"fps = {umr.fps} != expected {UMR_FPS}")

    feat = umr.features
    if feat.ndim != 2 or feat.shape[-1] != DIM_FEATURES:
        report.fail(f"features shape {feat.shape} must be (T-1, {DIM_FEATURES})")
    else:
        _check_finite(feat, "features", report)

    if umr.canonical_anchor.init_root_pos.shape != (3,):
        report.fail(f"init_root_pos shape {umr.canonical_anchor.init_root_pos.shape} must be (3,)")
    if umr.canonical_anchor.init_root_rot6d.shape != (DIM_ROOT_ROT6D,):
        report.fail(
            f"init_root_rot6d shape {umr.canonical_anchor.init_root_rot6d.shape} "
            f"must be ({DIM_ROOT_ROT6D},)"
        )
    if umr.canonical_anchor.init_joints76_rot6d.shape != (UMR_NUM_JOINTS76, 6):
        report.fail(
            f"init_joints76_rot6d shape {umr.canonical_anchor.init_joints76_rot6d.shape} "
            f"must be ({UMR_NUM_JOINTS76}, 6)"
        )

    if umr.joints77_pos.shape != (umr.num_frames, UMR_NUM_JOINTS, 3):
        report.fail(
            f"joints77_pos shape {umr.joints77_pos.shape} must be "
            f"({umr.num_frames}, {UMR_NUM_JOINTS}, 3)"
        )
    else:
        _check_finite(umr.joints77_pos, "joints77_pos", report)
    if umr.identity_coeffs.ndim != 2 or umr.identity_coeffs.shape[0] != 1:
        report.fail(f"identity_coeffs shape {umr.identity_coeffs.shape} must be (1, C)")
    else:
        _check_finite(umr.identity_coeffs, "identity_coeffs", report)
    if umr.joint_orient.shape != (78, 3, 3):
        report.fail(f"joint_orient shape {umr.joint_orient.shape} must be (78, 3, 3)")
    else:
        _check_finite(umr.joint_orient, "joint_orient", report)

    report.stats["num_frames"] = float(umr.num_frames)
    report.stats["num_records"] = float(umr.num_records)
    return report

__all__ = [
    "ValidationReport",
    "validate_soma77_canonical",
    "validate_umr499",
]
