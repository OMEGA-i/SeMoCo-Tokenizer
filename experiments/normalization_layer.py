"""Torch normalization layer keyed off :class:`NormalizationStats`.

Z-scores the 499D ``features`` before the tokenizer and denormalizes the decode
before geometry / L2 metrics. Foot contact channels stay in ``{0, 1}``
(mean=0 / std=1). Stats are buffers so they ride along with checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from data.normalization import NormalizationStats
from data.umr_schema import (
    DIM_FEATURES,
    DIM_JOINTS76_ROT6D,
    DIM_SPARSE_VEL,
    NUM_SPARSE_VEL_JOINTS,
    SLICE_FOOT_CONTACT,
    SLICE_JOINTS76_ROT6D,
    SLICE_ROOT_ROT6D,
    SLICE_ROOT_TRAJ,
    SLICE_SPARSE_VEL,
    UMR_NUM_JOINTS76,
)


def _stats_to_packed_mean_std(
    stats: NormalizationStats, *, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Build mean / std vectors of shape ``[DIM_FEATURES]`` aligned with the UMR layout.

    Foot contact channels get ``mean=0 / std=1`` so the BCE target stays in ``{0, 1}``.
    """
    mean = np.zeros(DIM_FEATURES, dtype=np.float32)
    std = np.ones(DIM_FEATURES, dtype=np.float32)

    mean[SLICE_ROOT_TRAJ] = np.asarray(stats.root_traj_mean, dtype=np.float32)
    std[SLICE_ROOT_TRAJ] = np.maximum(np.asarray(stats.root_traj_std, dtype=np.float32), eps)

    mean[SLICE_ROOT_ROT6D] = np.asarray(stats.root_rot6d_mean, dtype=np.float32)
    std[SLICE_ROOT_ROT6D] = np.maximum(np.asarray(stats.root_rot6d_std, dtype=np.float32), eps)

    joints_mean = np.asarray(stats.joints76_rot6d_mean, dtype=np.float32)
    joints_std = np.asarray(stats.joints76_rot6d_std, dtype=np.float32)
    if joints_mean.shape != (UMR_NUM_JOINTS76, 6) or joints_std.shape != (UMR_NUM_JOINTS76, 6):
        raise ValueError(
            f"joints76_rot6d stats must be [76, 6]; got mean={joints_mean.shape} "
            f"std={joints_std.shape}"
        )
    mean[SLICE_JOINTS76_ROT6D] = joints_mean.reshape(DIM_JOINTS76_ROT6D)
    std[SLICE_JOINTS76_ROT6D] = np.maximum(joints_std.reshape(DIM_JOINTS76_ROT6D), eps)

    sparse_mean = np.asarray(stats.sparse_vel_mean, dtype=np.float32)
    sparse_std = np.asarray(stats.sparse_vel_std, dtype=np.float32)
    if sparse_mean.shape != (NUM_SPARSE_VEL_JOINTS, 3) or sparse_std.shape != (NUM_SPARSE_VEL_JOINTS, 3):
        raise ValueError(
            f"sparse_vel stats must be [8, 3]; got mean={sparse_mean.shape} "
            f"std={sparse_std.shape}"
        )
    mean[SLICE_SPARSE_VEL] = sparse_mean.reshape(DIM_SPARSE_VEL)
    std[SLICE_SPARSE_VEL] = np.maximum(sparse_std.reshape(DIM_SPARSE_VEL), eps)

    # foot_contact: keep mean=0 / std=1.
    mean[SLICE_FOOT_CONTACT] = 0.0
    std[SLICE_FOOT_CONTACT] = 1.0
    return mean, std


class FeatureNormalizationLayer(nn.Module):
    """Z-score the 499D ``features`` channels-first stream."""

    mean: Tensor
    std: Tensor

    def __init__(self, stats: NormalizationStats, *, eps: float = 1e-6) -> None:
        super().__init__()
        mean_np, std_np = _stats_to_packed_mean_std(stats, eps=eps)
        # Shape [1, C, 1] for broadcasting against channels-first features.
        self.register_buffer("mean", torch.from_numpy(mean_np).view(1, DIM_FEATURES, 1))
        self.register_buffer("std", torch.from_numpy(std_np).view(1, DIM_FEATURES, 1))

    def forward(self, features: Tensor) -> Tensor:
        dtype = features.dtype
        return (features - self.mean.to(dtype)) / self.std.to(dtype)

    def inverse(self, features_norm: Tensor) -> Tensor:
        dtype = features_norm.dtype
        return features_norm * self.std.to(dtype) + self.mean.to(dtype)

    @classmethod
    def from_json(cls, path: str | Path) -> "FeatureNormalizationLayer":
        stats = NormalizationStats.from_json(Path(path))
        return cls(stats)


__all__ = [
    "FeatureNormalizationLayer",
]
