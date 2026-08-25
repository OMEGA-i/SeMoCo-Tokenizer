"""L0 validators."""

from __future__ import annotations

import numpy as np

from data.umr_schema import (
    DIM_FOOT_CONTACT,
    DIM_ROOT_ROT6D,
    FEATURE_VARIANT,
    NUM_SPARSE_VEL_JOINTS,
    UMR_FPS,
    UMR_NUM_JOINTS,
    UMR_NUM_JOINTS76,
    CanonicalAnchor,
    FeatureFields,
    UMR499,
    pack_features,
)
from data.validation import validate_soma77_canonical, validate_umr499

def _ok_umr(rng: np.random.Generator, T: int = 8) -> UMR499:
    fields = FeatureFields(
        root_traj=rng.standard_normal((T - 1, 9)).astype(np.float32),
        root_rot6d=rng.standard_normal((T - 1, DIM_ROOT_ROT6D)).astype(np.float32),
        joints76_rot6d=rng.standard_normal((T - 1, UMR_NUM_JOINTS76, 6)).astype(np.float32),
        sparse_vel=rng.standard_normal((T - 1, NUM_SPARSE_VEL_JOINTS, 3)).astype(np.float32),
        foot_contact=(rng.random((T - 1, DIM_FOOT_CONTACT)) > 0.5).astype(np.float32),
    )
    return UMR499(
        canonical_anchor=CanonicalAnchor(
            init_root_pos=np.zeros(3, dtype=np.float32),
            init_root_rot6d=np.zeros(DIM_ROOT_ROT6D, dtype=np.float32),
            init_joints76_rot6d=np.zeros((UMR_NUM_JOINTS76, 6), dtype=np.float32),
        ),
        features=pack_features(fields),
        joints77_pos=rng.standard_normal((T, UMR_NUM_JOINTS, 3)).astype(np.float32),
        identity_coeffs=np.zeros((1, 10), dtype=np.float32),
        joint_orient=np.tile(np.eye(3, dtype=np.float32)[None], (78, 1, 1)),
        fps=UMR_FPS,
        feature_variant=FEATURE_VARIANT,
    )

def test_validate_umr499_ok() -> None:
    rng = np.random.default_rng(0)
    umr = _ok_umr(rng)
    report = validate_umr499(umr)
    assert report.ok, report.failures
    assert report.stats["num_frames"] == 8.0
    assert report.stats["num_records"] == 7.0

def test_validate_umr499_flags_wrong_fps() -> None:
    rng = np.random.default_rng(1)
    umr = _ok_umr(rng)
    umr.fps = 30.0
    report = validate_umr499(umr)
    assert not report.ok
    assert any("fps" in f for f in report.failures)

def test_validate_umr499_flags_nan_features() -> None:
    rng = np.random.default_rng(2)
    umr = _ok_umr(rng)
    umr.features[0, 0] = np.nan
    report = validate_umr499(umr)
    assert not report.ok
    assert any("NaN" in f for f in report.failures)

def test_validate_soma77_all_one_contacts_are_valid(synthetic_canonical_short) -> None:
    synthetic_canonical_short.foot_contacts[:] = 1.0
    report = validate_soma77_canonical(synthetic_canonical_short)
    assert report.ok, report.failures

def test_validate_soma77_all_zero_contacts_fail(synthetic_canonical_short) -> None:
    synthetic_canonical_short.foot_contacts[:] = 0.0
    report = validate_soma77_canonical(synthetic_canonical_short)
    assert not report.ok
    assert any("all-zero" in f for f in report.failures)
