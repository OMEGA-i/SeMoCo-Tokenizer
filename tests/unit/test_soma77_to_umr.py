"""Encoder + decoder identity tests."""

from __future__ import annotations

import numpy as np
import pytest

from data.soma77_schema import NUM_RIG_JOINTS, Soma77Canonical
from data.soma77_to_umr import (
    apply_canon_to_root_axis_angle,
    axis_angle_to_rotmat,
    build_features_and_anchor,
    canonicalize_positions,
    rotmat_to_rot6d,
    slerp_axis_angle,
    soma77_to_umr499,
)
from data.umr_schema import (
    DIM_FEATURES,
    FEATURE_VARIANT,
    UMR_FPS,
    UMR_NUM_JOINTS,
    UMR_NUM_JOINTS76,
)
from data.umr_to_soma77 import (
    materialize_features,
    materialize_umr499,
    rot6d_to_rotmat,
    rotmat_to_axis_angle,
)

def _synthetic_joint_motion(T: int, *, fps: float, rng: np.random.Generator) -> np.ndarray:
    """A simple SMPL-X-like skeleton walking forward with mild noise."""
    base = np.zeros((UMR_NUM_JOINTS, 3), dtype=np.float32)
    base[0] = [0.0, 1.0, 0.0]
    base[1] = [-0.1, 0.95, 0.0]
    base[2] = [0.1, 0.95, 0.0]
    base[3] = [0.0, 1.15, 0.0]
    base[16] = [-0.18, 1.45, 0.0]
    base[17] = [0.18, 1.45, 0.0]
    base[15] = [0.0, 1.6, 0.0]
    rest = base.copy()
    for j in range(UMR_NUM_JOINTS):
        if (rest[j] == 0).all():
            rest[j] = base[0] + rng.standard_normal(3).astype(np.float32) * 0.2

    joints = np.repeat(rest[None], T, axis=0)
    t = np.arange(T) / fps
    joints[:, :, 0] += np.linspace(0.0, 1.5, T, dtype=np.float32)[:, None]
    joints[:, :, 1] += 0.03 * np.sin(2 * np.pi * 1.2 * t)[:, None].astype(np.float32)
    joints += (rng.standard_normal(joints.shape).astype(np.float32) * 0.01)
    return joints

def _synthetic_canonical(T: int, fps: float, rng: np.random.Generator) -> Soma77Canonical:
    poses = (rng.standard_normal((T, UMR_NUM_JOINTS, 3)).astype(np.float32) * 0.2)
    transl = (rng.standard_normal((T, 3)).astype(np.float32) * 0.02)
    transl[:, 1] = 0.95
    foot_contacts = np.zeros((T, 4), dtype=np.float32)
    foot_contacts[::4, 0] = 1.0
    foot_contacts[1::4, 2] = 1.0
    return Soma77Canonical(
        poses=poses,
        transl=transl,
        identity_coeffs=np.zeros((1, 10), dtype=np.float32),
        joint_orient=np.tile(np.eye(3, dtype=np.float32)[None], (NUM_RIG_JOINTS, 1, 1)),
        foot_contacts=foot_contacts,
        fps_src=fps,
    )

def test_axis_angle_round_trip() -> None:
    rng = np.random.default_rng(0)
    aa = rng.standard_normal((11, 3)).astype(np.float32) * 0.7
    R = axis_angle_to_rotmat(aa)
    aa_back = rotmat_to_axis_angle(R)
    R_back = axis_angle_to_rotmat(aa_back)
    np.testing.assert_allclose(R, R_back, atol=1e-5)

def test_rot6d_round_trip() -> None:
    rng = np.random.default_rng(1)
    aa = rng.standard_normal((11, 3)).astype(np.float32) * 0.7
    R = axis_angle_to_rotmat(aa)
    R_back = rot6d_to_rotmat(rotmat_to_rot6d(R))
    np.testing.assert_allclose(R, R_back, atol=1e-5)

def test_canonicalize_positions_invariants() -> None:
    rng = np.random.default_rng(2)
    joints = _synthetic_joint_motion(50, fps=30.0, rng=rng)
    canon, _quat = canonicalize_positions(joints)
    assert canon[..., 1].min() == pytest.approx(0.0, abs=1e-5)
    np.testing.assert_allclose(canon[0, 0, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(canon[0, 0, 2], 0.0, atol=1e-5)

def test_slerp_axis_angle_target_frames() -> None:
    rng = np.random.default_rng(3)
    aa = rng.standard_normal((10, 5, 3)).astype(np.float32) * 0.3
    out = slerp_axis_angle(aa, src_fps=20.0, target_fps=50.0, target_frames=24)
    assert out.shape == (24, 5, 3)

def test_encode_decode_identity_small() -> None:
    rng = np.random.default_rng(4)
    canonical = _synthetic_canonical(T=64, fps=50.0, rng=rng)
    joints77 = _synthetic_joint_motion(canonical.num_frames, fps=50.0, rng=rng)

    umr = soma77_to_umr499(canonical, joints77_world=joints77)
    assert umr.feature_variant == FEATURE_VARIANT
    assert umr.fps == pytest.approx(UMR_FPS)
    assert umr.features.shape == (canonical.num_frames - 1, DIM_FEATURES)
    assert umr.joints77_pos.shape == (canonical.num_frames, UMR_NUM_JOINTS, 3)
    assert umr.identity_coeffs.shape == (1, 10)
    assert umr.joint_orient.shape == (78, 3, 3)

    decoded = materialize_umr499(umr)
    assert decoded.rotvec77.shape == (canonical.num_frames, UMR_NUM_JOINTS, 3)
    assert decoded.transl.shape == (canonical.num_frames, 3)
    assert decoded.foot_contacts.shape == (canonical.num_frames, 4)

    np.testing.assert_allclose(
        decoded.transl[:, [0, 2]],
        umr.joints77_pos[:, 0, [0, 2]],
        atol=1e-3,
    )
    np.testing.assert_allclose(
        decoded.transl[:, 1],
        umr.joints77_pos[:, 0, 1],
        atol=1e-3,
    )

    expected_joints76 = canonical.poses[:, 1:UMR_NUM_JOINTS]
    np.testing.assert_allclose(
        decoded.rotvec77[:, 1:UMR_NUM_JOINTS],
        expected_joints76,
        atol=2e-3,
    )

def test_anchor_consistency_with_first_record() -> None:
    """anchor.init_root_rot6d must match the canonicalized frame-0 root."""
    rng = np.random.default_rng(5)
    canonical = _synthetic_canonical(T=32, fps=50.0, rng=rng)
    joints77 = _synthetic_joint_motion(canonical.num_frames, fps=50.0, rng=rng)
    umr = soma77_to_umr499(canonical, joints77_world=joints77)

    canon_positions, canon_quat = canonicalize_positions(joints77)
    canon_aa = canonical.poses.copy()
    canon_aa[:, 0] = apply_canon_to_root_axis_angle(canonical.poses[:, 0], canon_quat)
    expected_root_rot6d = rotmat_to_rot6d(axis_angle_to_rotmat(canon_aa[0, 0]))
    np.testing.assert_allclose(
        umr.canonical_anchor.init_root_rot6d,
        expected_root_rot6d,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        umr.canonical_anchor.init_root_pos,
        canon_positions[0, 0],
        atol=1e-5,
    )

def test_build_features_min_two_frames() -> None:
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError):
        build_features_and_anchor(
            rng.standard_normal((1, UMR_NUM_JOINTS, 3)).astype(np.float32),
            rng.standard_normal((1, UMR_NUM_JOINTS, 3)).astype(np.float32),
        )
