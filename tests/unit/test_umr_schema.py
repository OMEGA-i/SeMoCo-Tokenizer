"""Schema-level unit tests for UMR."""

from __future__ import annotations

import numpy as np
import pytest

from data.umr_schema import (
    DIM_FEATURES,
    DIM_FOOT_CONTACT,
    DIM_JOINTS76_ROT6D,
    DIM_ROOT_ROT6D,
    DIM_ROOT_TRAJ,
    DIM_SPARSE_VEL,
    DIM_TRAJ_CONTACT,
    FEATURE_VARIANT,
    NUM_SPARSE_VEL_JOINTS,
    SLICE_FOOT_CONTACT,
    SLICE_JOINTS76_ROT6D,
    SLICE_ROOT_ROT6D,
    SLICE_ROOT_TRAJ,
    SLICE_SPARSE_VEL,
    UMR_FPS,
    UMR_NUM_JOINTS,
    UMR_NUM_JOINTS76,
    CanonicalAnchor,
    FeatureFields,
    UMR499,
    join_streams,
    pack_features,
    split_streams,
    unpack_features,
)

def _rand_fields(rng: np.random.Generator, T: int) -> FeatureFields:
    return FeatureFields(
        root_traj=rng.standard_normal((T, DIM_ROOT_TRAJ)).astype(np.float32),
        root_rot6d=rng.standard_normal((T, DIM_ROOT_ROT6D)).astype(np.float32),
        joints76_rot6d=rng.standard_normal((T, UMR_NUM_JOINTS76, 6)).astype(np.float32),
        sparse_vel=rng.standard_normal((T, NUM_SPARSE_VEL_JOINTS, 3)).astype(np.float32),
        foot_contact=(rng.random((T, DIM_FOOT_CONTACT)) > 0.5).astype(np.float32),
    )

def test_dim_constants_consistent() -> None:
    assert DIM_FEATURES == 499
    assert DIM_ROOT_TRAJ == 9
    assert DIM_ROOT_ROT6D == 6
    assert DIM_JOINTS76_ROT6D == 76 * 6
    assert DIM_SPARSE_VEL == 8 * 3
    assert DIM_FOOT_CONTACT == 4
    assert DIM_TRAJ_CONTACT == 9 + 6 + 24 + 4
    assert DIM_ROOT_TRAJ + DIM_ROOT_ROT6D + DIM_JOINTS76_ROT6D + DIM_SPARSE_VEL + DIM_FOOT_CONTACT == DIM_FEATURES

def test_slices_cover_499d() -> None:
    seen = np.zeros(DIM_FEATURES, dtype=bool)
    for s in (SLICE_ROOT_TRAJ, SLICE_ROOT_ROT6D, SLICE_JOINTS76_ROT6D, SLICE_SPARSE_VEL, SLICE_FOOT_CONTACT):
        seen[s] = True
    assert seen.all(), "slices must partition the 499D vector"

def test_pack_unpack_round_trip() -> None:
    rng = np.random.default_rng(0)
    fields = _rand_fields(rng, T=11)
    packed = pack_features(fields)
    assert packed.shape == (11, DIM_FEATURES)

    back = unpack_features(packed)
    np.testing.assert_allclose(back.root_traj, fields.root_traj)
    np.testing.assert_allclose(back.root_rot6d, fields.root_rot6d)
    np.testing.assert_allclose(back.joints76_rot6d, fields.joints76_rot6d)
    np.testing.assert_allclose(back.sparse_vel, fields.sparse_vel)
    np.testing.assert_allclose(back.foot_contact, fields.foot_contact)

def test_split_join_streams_round_trip() -> None:
    rng = np.random.default_rng(1)
    fields = _rand_fields(rng, T=7)
    packed = pack_features(fields)
    traj, joints = split_streams(packed)
    assert traj.shape == (7, DIM_TRAJ_CONTACT)
    assert joints.shape == (7, DIM_JOINTS76_ROT6D)
    rejoined = join_streams(traj, joints)
    np.testing.assert_allclose(rejoined, packed)

def test_umr499_npz_round_trip(tmp_path) -> None:
    rng = np.random.default_rng(2)
    T = 21
    fields = _rand_fields(rng, T=T - 1)
    features = pack_features(fields)
    joints77 = rng.standard_normal((T, UMR_NUM_JOINTS, 3)).astype(np.float32)
    anchor = CanonicalAnchor(
        init_root_pos=rng.standard_normal(3).astype(np.float32),
        init_root_rot6d=rng.standard_normal(DIM_ROOT_ROT6D).astype(np.float32),
        init_joints76_rot6d=rng.standard_normal((UMR_NUM_JOINTS76, 6)).astype(np.float32),
    )
    umr = UMR499(
        canonical_anchor=anchor,
        features=features,
        joints77_pos=joints77,
        identity_coeffs=np.zeros((1, 10), dtype=np.float32),
        joint_orient=np.tile(np.eye(3, dtype=np.float32)[None], (78, 1, 1)),
        fps=UMR_FPS,
        feature_variant=FEATURE_VARIANT,
    )
    path = tmp_path / "rec_test" / "umr499.npz"
    umr.to_npz(path)
    loaded = UMR499.from_npz(path)
    np.testing.assert_allclose(loaded.features, features)
    np.testing.assert_allclose(loaded.joints77_pos, joints77)
    np.testing.assert_allclose(loaded.canonical_anchor.init_root_pos, anchor.init_root_pos)
    np.testing.assert_allclose(loaded.canonical_anchor.init_root_rot6d, anchor.init_root_rot6d)
    np.testing.assert_allclose(loaded.canonical_anchor.init_joints76_rot6d, anchor.init_joints76_rot6d)
    np.testing.assert_allclose(loaded.identity_coeffs, np.zeros((1, 10), dtype=np.float32))
    np.testing.assert_allclose(loaded.joint_orient, np.tile(np.eye(3, dtype=np.float32)[None], (78, 1, 1)))
    assert loaded.fps == UMR_FPS
    assert loaded.feature_variant == FEATURE_VARIANT

def test_umr499_rejects_wrong_variant(tmp_path) -> None:
    rng = np.random.default_rng(3)
    T = 5
    features = pack_features(_rand_fields(rng, T=T - 1))
    joints77 = rng.standard_normal((T, UMR_NUM_JOINTS, 3)).astype(np.float32)
    anchor = CanonicalAnchor(
        init_root_pos=np.zeros(3, dtype=np.float32),
        init_root_rot6d=np.zeros(DIM_ROOT_ROT6D, dtype=np.float32),
        init_joints76_rot6d=np.zeros((UMR_NUM_JOINTS76, 6), dtype=np.float32),
    )
    umr = UMR499(
        canonical_anchor=anchor,
        features=features,
        joints77_pos=joints77,
        identity_coeffs=np.zeros((1, 10), dtype=np.float32),
        joint_orient=np.tile(np.eye(3, dtype=np.float32)[None], (78, 1, 1)),
        feature_variant="bogus_variant",
    )
    path = tmp_path / "bad.npz"
    umr.to_npz(path)
    with pytest.raises(ValueError, match="feature_variant"):
        UMR499.from_npz(path)
