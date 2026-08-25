"""In-RAM / mmap shared-tensor training cache (train split only).

Wraps a packed sliding-window cache from :mod:`tools.build_training_cache`;
stores only ``features`` (validation uses :class:`data.umr_dataset.UMRDataset`).

Cache layout::

    local://cache/umr499_train.npy           # fp32 [N_windows, window, 499]
    local://cache/umr499_train.npy.meta.json # window, step, samples, ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.local_uri import default_data_root, resolve_local_uri
from data.umr_schema import DIM_FEATURES, FEATURE_VARIANT

META_SUFFIX = ".meta.json"

def cache_meta_path(cache_path: str | Path) -> Path:
    """Sibling meta.json for a cache ``.npy``."""
    p = Path(cache_path)
    return p.with_suffix(p.suffix + META_SUFFIX)

def load_cache_meta(cache_path: str | Path) -> dict[str, Any]:
    return json.loads(cache_meta_path(cache_path).read_text())

def cache_is_ready(cache_path: str | Path) -> bool:
    """Quick existence/shape sanity check: meta knobs vs the on-disk ``.npy`` header.

    ``.npy`` headers are variable-length, so the shape is inspected via the
    numpy header readers instead of comparing byte size.
    """
    p = Path(cache_path)
    meta_p = cache_meta_path(p)
    if not p.is_file() or not meta_p.is_file():
        return False
    try:
        meta = json.loads(meta_p.read_text())
    except Exception:
        return False
    if str(meta.get("dtype", "")) != "float32":
        return False
    if str(meta.get("feature_variant", "")) != FEATURE_VARIANT:
        return False
    expected_shape = tuple(int(s) for s in meta.get("shape", []))
    if len(expected_shape) != 3 or expected_shape[-1] != DIM_FEATURES:
        return False
    try:
        with open(p, "rb") as fh:
            version = np.lib.format.read_magic(fh)
            if version[0] == 1:
                shape, _fortran, dtype = np.lib.format.read_array_header_1_0(fh)
            else:
                shape, _fortran, dtype = np.lib.format.read_array_header_2_0(fh)
    except Exception:
        return False
    if dtype != np.dtype(np.float32):
        return False
    return tuple(int(s) for s in shape) == expected_shape

class CachedUMRFeatureDataset(Dataset):
    """Train-only feature cache backed by a shared mmap.

    The cache file is mmap'd read-only, so under DDP all ranks share the OS
    page cache and DataLoader workers inherit the mmap on fork. With
    ``in_ram=True`` the cache is materialized into a ``share_memory_()``
    tensor instead (single-rank only).
    """

    def __init__(
        self,
        cache_path: str | Path,
        *,
        data_root: Optional[str | Path] = None,
        in_ram: bool = False,
        tmr_emb_path: Optional[str | Path] = None,
    ) -> None:
        self.data_root = Path(data_root) if data_root is not None else default_data_root()
        self.cache_path = resolve_local_uri(cache_path, self.data_root)
        if not self.cache_path.is_file():
            raise FileNotFoundError(f"training cache not found: {self.cache_path}")

        meta = load_cache_meta(self.cache_path)
        if str(meta.get("dtype", "")) != "float32":
            raise ValueError(
                f"{self.cache_path}: unsupported cache dtype {meta.get('dtype')!r}; expected float32"
            )
        if str(meta.get("feature_variant", "")) != FEATURE_VARIANT:
            raise ValueError(
                f"{self.cache_path}: feature_variant={meta.get('feature_variant')!r} != {FEATURE_VARIANT!r}"
            )
        shape = tuple(meta.get("shape", []))
        if len(shape) != 3 or shape[-1] != DIM_FEATURES:
            raise ValueError(f"{self.cache_path}: unexpected shape {shape}")
        self.meta = meta
        self.window = int(meta["window"])
        self.fps = float(meta.get("fps", 0.0))
        self.in_ram = bool(in_ram)

        # Open the cache as a memmap. Under DDP every rank mmap's the same
        # file, so the OS page cache holds at most one copy.
        self._mmap: np.ndarray = np.load(self.cache_path, mmap_mode="r")
        if self._mmap.dtype != np.dtype(np.float32):
            raise ValueError(f"{self.cache_path}: dtype on disk = {self._mmap.dtype}")

        if self.in_ram:
            arr = np.array(self._mmap, dtype=np.float32, copy=True)
            self.features: Tensor = torch.from_numpy(arr)
            self.features.share_memory_()
        else:
            self.features = None  # type: ignore[assignment]

        # Optional row-aligned TMR teacher-embedding cache; row i is fetched
        # by the same index as ``features`` (alignment guaranteed by the builder).
        self._tmr_mmap: Optional[np.ndarray] = None
        if tmr_emb_path is not None:
            tmr_path = resolve_local_uri(tmr_emb_path, self.data_root)
            if not tmr_path.is_file():
                raise FileNotFoundError(f"tmr embedding cache not found: {tmr_path}")
            self._tmr_mmap = np.load(tmr_path, mmap_mode="r")
            if self._tmr_mmap.shape[0] != self._mmap.shape[0]:
                raise ValueError(
                    f"tmr cache rows {self._tmr_mmap.shape[0]} != feature cache rows "
                    f"{self._mmap.shape[0]} ({tmr_path})"
                )

    def __len__(self) -> int:
        return int(self._mmap.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.in_ram and self.features is not None:
            sample: dict[str, Any] = {"features": self.features[idx]}
        else:
            # mmap read: copy the row into an owned writable fp32 buffer. The copy
            # is small (one window = 180 * 499 * 4 ≈ 360 KiB) and keeps the OS
            # page cache shared across DDP ranks.
            row = np.array(self._mmap[idx], dtype=np.float32, copy=True)
            sample = {"features": torch.from_numpy(row)}
        if self._tmr_mmap is not None:
            emb = np.array(self._tmr_mmap[idx], dtype=np.float32, copy=True)
            sample["tmr_emb"] = torch.from_numpy(emb)
        return sample

    def __getitems__(self, indices: list[int]) -> dict[str, Tensor]:
        """Vectorized batch fetch used by modern PyTorch DataLoader."""

        if self.in_ram and self.features is not None:
            out: dict[str, Tensor] = {"features": self.features[indices]}
        else:
            idx = np.asarray(indices, dtype=np.int64)
            batch = np.asarray(self._mmap[idx], dtype=np.float32)
            if not batch.flags.writeable or not batch.flags.c_contiguous:
                batch = np.array(batch, dtype=np.float32, copy=True)
            out = {"features": torch.from_numpy(batch)}
        if self._tmr_mmap is not None:
            tidx = np.asarray(indices, dtype=np.int64)
            tbatch = np.array(self._tmr_mmap[tidx], dtype=np.float32, copy=True)
            out["tmr_emb"] = torch.from_numpy(tbatch)
        return out

def collate_cached_umr(batch: list[dict[str, Any]] | dict[str, Tensor]) -> dict[str, Any]:
    """Stack ``features`` (and the optional row-aligned ``tmr_emb``) views."""
    if isinstance(batch, dict):
        return batch
    out: dict[str, Any] = {"features": torch.stack([s["features"] for s in batch])}
    if batch and "tmr_emb" in batch[0]:
        out["tmr_emb"] = torch.stack([s["tmr_emb"] for s in batch])
    return out

__all__ = [
    "CachedUMRFeatureDataset",
    "cache_is_ready",
    "cache_meta_path",
    "collate_cached_umr",
    "load_cache_meta",
]
