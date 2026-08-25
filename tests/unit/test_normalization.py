"""Normalization stats unit tests."""

from __future__ import annotations

import numpy as np

from data.normalization import (
    BinaryRateAccumulator,
    NormalizationStats,
    NormalizationStatsBuilder,
    WelfordAccumulator,
)
from data.umr_schema import (
    DIM_FEATURES,
    DIM_FOOT_CONTACT,
    DIM_ROOT_ROT6D,
    DIM_ROOT_TRAJ,
    NUM_SPARSE_VEL_JOINTS,
    SLICE_FOOT_CONTACT,
    UMR_NUM_JOINTS76,
    UMR_FPS,
)

def test_welford_matches_numpy_stats() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4096, 5)).astype(np.float64)
    acc = WelfordAccumulator((5,))
    for chunk in np.array_split(x, 17):
        acc.update(chunk)
    np.testing.assert_allclose(acc.mean, x.mean(axis=0), atol=1e-10)
    np.testing.assert_allclose(acc.std(eps=0.0), x.std(axis=0), atol=1e-7)

def test_binary_rate_accumulator() -> None:
    rng = np.random.default_rng(1)
    x = (rng.random((1000, 4)) > 0.3).astype(np.float64)
    acc = BinaryRateAccumulator((4,))
    for chunk in np.array_split(x, 13):
        acc.update(chunk)
    np.testing.assert_allclose(acc.positive_rate(), x.mean(axis=0), atol=1e-10)

def test_builder_finalize_shapes() -> None:
    rng = np.random.default_rng(2)
    builder = NormalizationStatsBuilder()
    for _ in range(4):
        features = rng.standard_normal((180, DIM_FEATURES)).astype(np.float32)
        features[:, SLICE_FOOT_CONTACT] = (rng.random((180, DIM_FOOT_CONTACT)) > 0.5).astype(np.float32)
        builder.update(features)
    stats = builder.finalize()
    assert stats.fps == UMR_FPS
    assert stats.num_clips == 4
    assert stats.num_records == 4 * 180
    assert len(stats.root_traj_mean) == DIM_ROOT_TRAJ
    assert len(stats.root_rot6d_mean) == DIM_ROOT_ROT6D
    assert len(stats.joints76_rot6d_mean) == UMR_NUM_JOINTS76
    assert all(len(row) == 6 for row in stats.joints76_rot6d_mean)
    assert len(stats.sparse_vel_mean) == NUM_SPARSE_VEL_JOINTS
    assert all(len(row) == 3 for row in stats.sparse_vel_mean)
    assert len(stats.foot_contact_positive_rate) == DIM_FOOT_CONTACT
    assert len(stats.packed_mean) == DIM_FEATURES

def test_stats_json_round_trip(tmp_path) -> None:
    rng = np.random.default_rng(3)
    builder = NormalizationStatsBuilder()
    features = rng.standard_normal((180, DIM_FEATURES)).astype(np.float32)
    features[:, SLICE_FOOT_CONTACT] = (rng.random((180, DIM_FOOT_CONTACT)) > 0.5).astype(np.float32)
    builder.update(features)
    stats = builder.finalize()
    path = tmp_path / "stats.json"
    stats.write_json(path)
    loaded = NormalizationStats.from_json(path)
    np.testing.assert_allclose(stats.packed_mean, loaded.packed_mean, atol=1e-10)
    np.testing.assert_allclose(stats.foot_contact_positive_rate, loaded.foot_contact_positive_rate, atol=1e-10)
    assert loaded.num_records == stats.num_records
