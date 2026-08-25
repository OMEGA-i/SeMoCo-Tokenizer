"""Train-only normalization stats for UMR ``features [N-1, 499]``.

Per-field streaming Welford accumulators plus a per-channel positive-rate for
binary ``foot_contact``; the packed 499D accumulator gives global mean/std and
p01/p99 clipping bounds::

    root_traj         [9]      Welford
    root_rot6d        [6]      Welford
    joints76_rot6d    [76, 6]  Welford
    sparse_vel        [8, 3]   Welford
    foot_contact      [4]      BinaryRate (positive_rate)
    packed            [499]    Welford + clip_p01 / clip_p99
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data.umr_schema import (
    DIM_FEATURES,
    DIM_FOOT_CONTACT,
    DIM_JOINTS76_ROT6D,
    DIM_ROOT_ROT6D,
    DIM_ROOT_TRAJ,
    DIM_SPARSE_VEL,
    NUM_SPARSE_VEL_JOINTS,
    SLICE_FOOT_CONTACT,
    SLICE_JOINTS76_ROT6D,
    SLICE_ROOT_ROT6D,
    SLICE_ROOT_TRAJ,
    SLICE_SPARSE_VEL,
    UMR_FPS,
    UMR_NUM_JOINTS76,
    UMR_VERSION,
)

# ---------------------------------------------------------------------------
# Welford streaming accumulator
# ---------------------------------------------------------------------------

class WelfordAccumulator:
    """Online mean / variance over a fixed-shape feature (Chan parallel update)."""

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype = np.float64) -> None:
        self.shape = tuple(shape)
        self.count: int = 0
        self.mean = np.zeros(shape, dtype=dtype)
        self.M2 = np.zeros(shape, dtype=dtype)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=self.mean.dtype)
        if x.shape[1:] != self.shape:
            raise ValueError(f"update expected trailing shape {self.shape}; got {x.shape[1:]}")
        n = int(x.shape[0])
        if n == 0:
            return
        new_count = self.count + n
        chunk_mean = x.mean(axis=0)
        delta = chunk_mean - self.mean
        if self.count == 0:
            self.mean[:] = chunk_mean
            self.M2[:] = ((x - chunk_mean) ** 2).sum(axis=0)
        else:
            self.mean += delta * (n / new_count)
            chunk_M2 = ((x - chunk_mean) ** 2).sum(axis=0)
            self.M2 += chunk_M2 + (delta ** 2) * (self.count * n / new_count)
        self.count = new_count

    def variance(self, *, ddof: int = 0) -> np.ndarray:
        if self.count <= ddof:
            return np.zeros_like(self.M2)
        return self.M2 / (self.count - ddof)

    def std(self, *, eps: float = 1e-8, ddof: int = 0) -> np.ndarray:
        return np.sqrt(self.variance(ddof=ddof) + eps)

class BinaryRateAccumulator:
    """Per-channel ``positive_rate`` for binary fields."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = tuple(shape)
        self.count: int = 0
        self.sum = np.zeros(shape, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=self.sum.dtype)
        if x.shape[1:] != self.shape:
            raise ValueError(f"update expected trailing shape {self.shape}; got {x.shape[1:]}")
        self.sum += x.sum(axis=0)
        self.count += int(x.shape[0])

    def positive_rate(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros_like(self.sum)
        return self.sum / self.count

@dataclass
class NormalizationStats:
    """Train-only stats for ``features``."""

    computed_from: str = "train_manifest_only"
    umr_version: str = UMR_VERSION
    fps: float = UMR_FPS
    num_clips: int = 0
    num_records: int = 0

    root_traj_mean: list[float] = field(default_factory=lambda: [0.0] * DIM_ROOT_TRAJ)
    root_traj_std: list[float] = field(default_factory=lambda: [1.0] * DIM_ROOT_TRAJ)
    root_traj_p95_norm: float = 0.0

    root_rot6d_mean: list[float] = field(default_factory=lambda: [0.0] * DIM_ROOT_ROT6D)
    root_rot6d_std: list[float] = field(default_factory=lambda: [1.0] * DIM_ROOT_ROT6D)

    joints76_rot6d_mean: list[list[float]] = field(
        default_factory=lambda: [[0.0] * 6 for _ in range(UMR_NUM_JOINTS76)]
    )
    joints76_rot6d_std: list[list[float]] = field(
        default_factory=lambda: [[1.0] * 6 for _ in range(UMR_NUM_JOINTS76)]
    )

    sparse_vel_mean: list[list[float]] = field(
        default_factory=lambda: [[0.0] * 3 for _ in range(NUM_SPARSE_VEL_JOINTS)]
    )
    sparse_vel_std: list[list[float]] = field(
        default_factory=lambda: [[1.0] * 3 for _ in range(NUM_SPARSE_VEL_JOINTS)]
    )

    foot_contact_positive_rate: list[float] = field(default_factory=lambda: [0.0] * DIM_FOOT_CONTACT)

    packed_mean: list[float] = field(default_factory=lambda: [0.0] * DIM_FEATURES)
    packed_std: list[float] = field(default_factory=lambda: [1.0] * DIM_FEATURES)
    packed_clip_p01: list[float] = field(default_factory=lambda: [0.0] * DIM_FEATURES)
    packed_clip_p99: list[float] = field(default_factory=lambda: [0.0] * DIM_FEATURES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "computed_from": self.computed_from,
            "umr_version": self.umr_version,
            "fps": self.fps,
            "num_clips": self.num_clips,
            "num_records": self.num_records,
            "fields": {
                "root_traj": {
                    "mean": self.root_traj_mean,
                    "std": self.root_traj_std,
                    "p95_norm": self.root_traj_p95_norm,
                },
                "root_rot6d": {
                    "mean": self.root_rot6d_mean,
                    "std": self.root_rot6d_std,
                },
                "joints76_rot6d": {
                    "mean": self.joints76_rot6d_mean,
                    "std": self.joints76_rot6d_std,
                },
                "sparse_vel": {
                    "mean": self.sparse_vel_mean,
                    "std": self.sparse_vel_std,
                },
                "foot_contact": {
                    "positive_rate": self.foot_contact_positive_rate,
                },
            },
            "packed": {
                "mean": self.packed_mean,
                "std": self.packed_std,
                "clip_p01": self.packed_clip_p01,
                "clip_p99": self.packed_clip_p99,
            },
        }

    def write_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return out

    @classmethod
    def from_json(cls, path: str | Path) -> "NormalizationStats":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NormalizationStats":
        fields_blk = payload["fields"]
        packed = payload["packed"]
        return cls(
            computed_from=payload.get("computed_from", "train_manifest_only"),
            umr_version=payload.get("umr_version", UMR_VERSION),
            fps=float(payload.get("fps", UMR_FPS)),
            num_clips=int(payload.get("num_clips", 0)),
            num_records=int(payload.get("num_records", 0)),
            root_traj_mean=list(fields_blk["root_traj"]["mean"]),
            root_traj_std=list(fields_blk["root_traj"]["std"]),
            root_traj_p95_norm=float(fields_blk["root_traj"]["p95_norm"]),
            root_rot6d_mean=list(fields_blk["root_rot6d"]["mean"]),
            root_rot6d_std=list(fields_blk["root_rot6d"]["std"]),
            joints76_rot6d_mean=list(fields_blk["joints76_rot6d"]["mean"]),
            joints76_rot6d_std=list(fields_blk["joints76_rot6d"]["std"]),
            sparse_vel_mean=list(fields_blk["sparse_vel"]["mean"]),
            sparse_vel_std=list(fields_blk["sparse_vel"]["std"]),
            foot_contact_positive_rate=list(fields_blk["foot_contact"]["positive_rate"]),
            packed_mean=list(packed["mean"]),
            packed_std=list(packed["std"]),
            packed_clip_p01=list(packed["clip_p01"]),
            packed_clip_p99=list(packed["clip_p99"]),
        )

class NormalizationStatsBuilder:
    """Streaming builder. Call :meth:`update` per clip, then :meth:`finalize`."""

    def __init__(self, *, fps: float = UMR_FPS) -> None:
        self.fps = float(fps)
        self.num_clips = 0
        self.root_traj = WelfordAccumulator((DIM_ROOT_TRAJ,))
        self.root_traj_xz_norms: list[float] = []
        self.root_rot6d = WelfordAccumulator((DIM_ROOT_ROT6D,))
        self.joints76_rot6d = WelfordAccumulator((UMR_NUM_JOINTS76, 6))
        self.sparse_vel = WelfordAccumulator((NUM_SPARSE_VEL_JOINTS, 3))
        self.foot_contact = BinaryRateAccumulator((DIM_FOOT_CONTACT,))
        self.packed = WelfordAccumulator((DIM_FEATURES,))
        self._p01_buf: list[np.ndarray] = []
        self._p99_buf: list[np.ndarray] = []

    def update(self, features: np.ndarray) -> None:
        """Consume one clip's ``features [T-1, 499]``."""
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[-1] != DIM_FEATURES:
            raise ValueError(f"features must be [T-1, {DIM_FEATURES}]; got {x.shape}")
        if x.shape[0] == 0:
            return

        root_traj = x[:, SLICE_ROOT_TRAJ]
        root_rot6d = x[:, SLICE_ROOT_ROT6D]
        joints76 = x[:, SLICE_JOINTS76_ROT6D].reshape(x.shape[0], UMR_NUM_JOINTS76, 6)
        sparse_vel = x[:, SLICE_SPARSE_VEL].reshape(x.shape[0], NUM_SPARSE_VEL_JOINTS, 3)
        contact = x[:, SLICE_FOOT_CONTACT]

        self.root_traj.update(root_traj)
        local_dxz_norm = np.linalg.norm(root_traj[:, 0:2], axis=-1)
        self.root_traj_xz_norms.append(float(np.quantile(local_dxz_norm, 0.95)))

        self.root_rot6d.update(root_rot6d)
        self.joints76_rot6d.update(joints76)
        self.sparse_vel.update(sparse_vel)
        self.foot_contact.update(contact)
        self.packed.update(x)

        self._p01_buf.append(np.quantile(x, 0.01, axis=0))
        self._p99_buf.append(np.quantile(x, 0.99, axis=0))
        self.num_clips += 1

    def finalize(self) -> NormalizationStats:
        if self.packed.count == 0:
            raise ValueError("NormalizationStatsBuilder.finalize: no data accumulated")

        if self._p01_buf:
            p01 = np.mean(np.stack(self._p01_buf, axis=0), axis=0)
            p99 = np.mean(np.stack(self._p99_buf, axis=0), axis=0)
        else:
            p01 = np.zeros(DIM_FEATURES, dtype=np.float64)
            p99 = np.zeros(DIM_FEATURES, dtype=np.float64)

        return NormalizationStats(
            fps=self.fps,
            num_clips=self.num_clips,
            num_records=self.packed.count,
            root_traj_mean=self.root_traj.mean.astype(float).tolist(),
            root_traj_std=self.root_traj.std().astype(float).tolist(),
            root_traj_p95_norm=float(np.mean(self.root_traj_xz_norms))
            if self.root_traj_xz_norms
            else 0.0,
            root_rot6d_mean=self.root_rot6d.mean.astype(float).tolist(),
            root_rot6d_std=self.root_rot6d.std().astype(float).tolist(),
            joints76_rot6d_mean=self.joints76_rot6d.mean.astype(float).tolist(),
            joints76_rot6d_std=self.joints76_rot6d.std().astype(float).tolist(),
            sparse_vel_mean=self.sparse_vel.mean.astype(float).tolist(),
            sparse_vel_std=self.sparse_vel.std().astype(float).tolist(),
            foot_contact_positive_rate=self.foot_contact.positive_rate().astype(float).tolist(),
            packed_mean=self.packed.mean.astype(float).tolist(),
            packed_std=self.packed.std().astype(float).tolist(),
            packed_clip_p01=p01.astype(float).tolist(),
            packed_clip_p99=p99.astype(float).tolist(),
        )

__all__ = [
    "BinaryRateAccumulator",
    "NormalizationStats",
    "NormalizationStatsBuilder",
    "WelfordAccumulator",
]
