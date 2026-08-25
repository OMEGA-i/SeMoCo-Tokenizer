"""Schema-level checks for :class:`data.soma77_schema.Soma77Canonical`."""

from __future__ import annotations

import numpy as np
import pytest

from data.soma77_schema import (
    DEFAULT_FALLBACK_FPS,
    NUM_JOINTS,
    NUM_NON_ROOT_JOINTS,
    NUM_RIG_JOINTS,
    Soma77Canonical,
)

def test_constants_are_self_consistent():
    assert NUM_JOINTS == 77
    assert NUM_NON_ROOT_JOINTS == 76
    assert NUM_RIG_JOINTS == 78
    assert NUM_NON_ROOT_JOINTS + 1 == NUM_JOINTS

def test_validate_shapes_passes(synthetic_canonical_short):
    synthetic_canonical_short.validate_shapes()

def test_validate_shapes_rejects_wrong_pose_count(synthetic_canonical_short):
    bad = synthetic_canonical_short
    bad.poses = bad.poses[:, :NUM_JOINTS - 1, :]
    with pytest.raises(ValueError, match=r"poses shape"):
        bad.validate_shapes()

def test_validate_shapes_rejects_wrong_joint_orient(synthetic_canonical_short):
    bad = synthetic_canonical_short
    bad.joint_orient = bad.joint_orient[:NUM_RIG_JOINTS - 1]
    with pytest.raises(ValueError, match=r"joint_orient shape"):
        bad.validate_shapes()

def test_load_round_trip(write_synthetic_npz):
    npz = write_synthetic_npz(T=80)
    loaded = Soma77Canonical.load(npz)
    assert loaded.poses.shape == (80, NUM_JOINTS, 3)
    assert loaded.transl.shape == (80, 3)
    assert loaded.foot_contacts.shape == (80, 4)
    assert loaded.fps_src == pytest.approx(30.0)

def test_load_falls_back_on_missing_manifest(tmp_path):
    rec = tmp_path / "rec" / "soma77.npz"
    rec.parent.mkdir(parents=True)
    np.savez(
        rec,
        poses=np.zeros((10, 77, 3), dtype=np.float32),
        transl=np.zeros((10, 3), dtype=np.float32),
        identity_coeffs=np.zeros((1, 10), dtype=np.float32),
        joint_orient=np.tile(np.eye(3, dtype=np.float32)[None], (78, 1, 1)),
        foot_contacts=np.zeros((10, 4), dtype=np.float32),
    )
    loaded = Soma77Canonical.load(rec)
    assert loaded.fps_src == pytest.approx(DEFAULT_FALLBACK_FPS)

def test_collapse_constant_identity(synthetic_canonical_short):
    obj = synthetic_canonical_short
    out = obj.collapse_constant_identity()
    assert out.identity_coeffs.shape == (1, 10)

    obj.identity_coeffs = np.tile(obj.identity_coeffs, (obj.num_frames, 1))
    out = obj.collapse_constant_identity()
    assert out.identity_coeffs.shape == (1, 10)

def test_to_npz_round_trip(tmp_path, synthetic_canonical_short):
    out = tmp_path / "soma77.npz"
    synthetic_canonical_short.to_npz(out)
    loaded = Soma77Canonical.load(out, fps_hint=30.0)
    np.testing.assert_array_equal(loaded.poses, synthetic_canonical_short.poses)
    np.testing.assert_array_equal(loaded.transl, synthetic_canonical_short.transl)
    np.testing.assert_array_equal(loaded.foot_contacts, synthetic_canonical_short.foot_contacts)
