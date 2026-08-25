"""Smoke tests for L1–L5 metric modules."""

from __future__ import annotations

import numpy as np

from data.umr_schema import (
    DIM_FEATURES,
    DIM_FOOT_CONTACT,
    NUM_SPARSE_VEL_JOINTS,
    SLICE_FOOT_CONTACT,
    UMR_NUM_JOINTS,
    UMR_NUM_JOINTS76,
    CanonicalAnchor,
    FeatureFields,
    pack_features,
)
from eval.codebook import codebook_metrics_per_layer, codebook_metrics_single_layer
from eval.codec_reconstruction import codec_reconstruction_metrics
from eval.geometry import soma_canonical_metrics
from eval.motion_validity import motion_validity_metrics
from eval.streaming import streaming_equivalence_metrics

def _rand_features(rng: np.random.Generator, T: int) -> np.ndarray:
    return pack_features(
        FeatureFields(
            root_traj=rng.standard_normal((T, 9)).astype(np.float32),
            root_rot6d=rng.standard_normal((T, 6)).astype(np.float32),
            joints76_rot6d=rng.standard_normal((T, UMR_NUM_JOINTS76, 6)).astype(np.float32),
            sparse_vel=rng.standard_normal((T, NUM_SPARSE_VEL_JOINTS, 3)).astype(np.float32),
            foot_contact=(rng.random((T, DIM_FOOT_CONTACT)) > 0.5).astype(np.float32),
        )
    )

def test_codec_reconstruction_identity_is_zero_l2_components() -> None:
    rng = np.random.default_rng(0)
    feat = _rand_features(rng, T=8)
    gt = feat.copy()
    feat = feat.copy()
    feat[:, SLICE_FOOT_CONTACT] = (gt[:, SLICE_FOOT_CONTACT] - 0.5) * 1e6
    metrics = codec_reconstruction_metrics(feat, gt)
    assert metrics.feat_mse == 0.0
    assert metrics.feat_mae == 0.0
    assert metrics.foot_contact_f1 == 1.0
    keys = metrics.to_dict()
    assert set(keys) == {"feat_mse", "feat_mae", "foot_contact_f1"}

def test_soma_canonical_metrics_zero_on_identity() -> None:
    rng = np.random.default_rng(1)
    T = 10
    transl = rng.standard_normal((T, 3)).astype(np.float32)
    rotvec = rng.standard_normal((T, UMR_NUM_JOINTS, 3)).astype(np.float32) * 0.5
    joints = rng.standard_normal((T, UMR_NUM_JOINTS, 3)).astype(np.float32)
    metrics = soma_canonical_metrics(
        transl,
        transl.copy(),
        rotvec,
        rotvec.copy(),
        joints77_pos_rec=joints,
        joints77_pos_gt=joints.copy(),
    )
    assert metrics.soma_root_trans_rmse_mm == 0.0
    assert metrics.soma_root_rot_rmse_rad < 1e-5
    assert metrics.soma_joints76_rot_rmse_rad < 1e-5
    assert metrics.soma_mpjpe_mm == 0.0

def test_codebook_metrics_full_codebook() -> None:
    indices = np.arange(64).repeat(10)[None, :]
    m = codebook_metrics_single_layer(indices, codebook_size=64)
    assert m.usage == 1.0
    assert m.dead_code_ratio == 0.0
    assert 60 <= m.perplexity <= 64

def test_codebook_metrics_per_layer_shape() -> None:
    indices = np.zeros((4, 8, 3), dtype=np.int64)
    metrics = codebook_metrics_per_layer(indices, codebook_size=64)
    assert len(metrics) == 3
    assert metrics[0].usage == 1 / 64

def test_motion_validity_metrics_finite() -> None:
    rng = np.random.default_rng(2)
    joints = rng.standard_normal((20, UMR_NUM_JOINTS, 3)).astype(np.float32) * 0.1
    m = motion_validity_metrics(joints)
    assert m.validity_nan_fraction == 0.0
    assert m.validity_root_drift_m >= 0.0

def test_streaming_equivalence_zero_diff() -> None:
    rng = np.random.default_rng(3)
    T = 8
    features = _rand_features(rng, T)
    anchor = CanonicalAnchor(
        init_root_pos=np.zeros(3, dtype=np.float32),
        init_root_rot6d=np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
        init_joints76_rot6d=np.tile(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32), (UMR_NUM_JOINTS76, 1)),
    )
    m = streaming_equivalence_metrics(features, anchor)
    assert m.streaming_feat_rmse < 1e-4
    assert m.streaming_root_trans_rmse_mm < 1e-1
    assert m.streaming_root_rot_rmse_rad < 1e-4
