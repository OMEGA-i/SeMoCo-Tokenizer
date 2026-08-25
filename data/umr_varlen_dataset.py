"""Variable-length (ragged) training dataset + length-bucket batching.

Reads a ragged cache (one clip per item, native length clamped to
``[min_frames, max_frames]``). To avoid padding/masking,
:class:`DistributedLengthBucketSampler` groups similar-length clips per batch
and :func:`collate_varlen_crop` crops each batch to its shortest length
(floored to ``token_stride``), yielding fixed-shape ``[B, L_batch, 499]``
tensors; lengths still vary across batches. The optional per-clip TMR teacher
(``<cache>.tmr.npy`` = ``[N_clips, dsem]``) is crop-invariant and returned
as-is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from data.local_uri import default_data_root, resolve_local_uri
from data.umr_schema import DIM_FEATURES, FEATURE_VARIANT

def _meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")

def varlen_cache_is_ready(cache_path: str | Path, data_root: str | Path | None = None) -> bool:
    root = Path(data_root) if data_root is not None else default_data_root()
    p = resolve_local_uri(cache_path, root)
    mp = _meta_path(p)
    ip = p.with_suffix(p.suffix + ".index.npy")
    if not (p.is_file() and mp.is_file() and ip.is_file()):
        return False
    try:
        meta = json.loads(mp.read_text())
    except Exception:
        return False
    return bool(meta.get("varlen")) and str(meta.get("feature_variant", "")) == FEATURE_VARIANT

class VarlenUMRDataset(Dataset):
    """Ragged event-clip feature dataset (one variable-length clip per item)."""

    def __init__(
        self,
        cache_path: str | Path,
        *,
        data_root: Optional[str | Path] = None,
        tmr_emb_path: Optional[str | Path] = None,
    ) -> None:
        self.data_root = Path(data_root) if data_root is not None else default_data_root()
        self.cache_path = resolve_local_uri(cache_path, self.data_root)
        mp = _meta_path(self.cache_path)
        if not self.cache_path.is_file() or not mp.is_file():
            raise FileNotFoundError(f"varlen cache/meta not found: {self.cache_path}")
        self.meta = json.loads(mp.read_text())
        if not self.meta.get("varlen"):
            raise ValueError(f"{self.cache_path}: meta.varlen is not true")
        if str(self.meta.get("feature_variant", "")) != FEATURE_VARIANT:
            raise ValueError(f"{self.cache_path}: feature_variant mismatch")
        self.token_stride = int(self.meta.get("token_stride", 4))
        self.fps = float(self.meta.get("fps", 50.0))

        self._data: np.ndarray = np.load(self.cache_path, mmap_mode="r")
        if self._data.dtype != np.dtype(np.float32) or self._data.shape[1] != DIM_FEATURES:
            raise ValueError(f"{self.cache_path}: bad feature array {self._data.shape} {self._data.dtype}")
        index = np.load(self.cache_path.with_suffix(self.cache_path.suffix + ".index.npy"))
        self._offsets = index[:, 0].astype(np.int64)
        self._lengths = index[:, 1].astype(np.int64)

        self._tmr: Optional[np.ndarray] = None
        self._tmr_per_token: bool = False
        self._tmr_index: Optional[np.ndarray] = None
        if tmr_emb_path is not None:
            tp = resolve_local_uri(tmr_emb_path, self.data_root)
            if not tp.is_file():
                raise FileNotFoundError(f"per-clip TMR teacher not found: {tp}")
            tmeta_p = tp.with_suffix(tp.suffix + ".meta.json")
            tmeta = json.loads(tmeta_p.read_text()) if tmeta_p.is_file() else {}
            self._tmr = np.load(tp, mmap_mode="r")
            self._tmr_per_token = bool(tmeta.get("varlen_per_token", False))
            n_clips = int(self._lengths.shape[0])
            if self._tmr_per_token:
                idx_p = tp.with_suffix(tp.suffix + ".index.npy")
                if not idx_p.is_file():
                    raise FileNotFoundError(f"per-token teacher index not found: {idx_p}")
                self._tmr_index = np.load(idx_p)
                if self._tmr_index.shape[0] != n_clips:
                    raise ValueError(
                        f"per-token teacher index rows {self._tmr_index.shape[0]} != n_clips {n_clips}"
                    )
                exp_ntok = self._lengths // self.token_stride
                if not np.array_equal(self._tmr_index[:, 1].astype(np.int64), exp_ntok):
                    raise ValueError(
                        f"per-token teacher token counts do not match lengths//stride ({tp})"
                    )
            elif self._tmr.shape[0] != n_clips:
                raise ValueError(
                    f"tmr rows {self._tmr.shape[0]} != n_clips {n_clips} ({tp})"
                )

    @property
    def lengths(self) -> np.ndarray:
        return self._lengths

    def __len__(self) -> int:
        return int(self._lengths.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        off = int(self._offsets[idx]); ln = int(self._lengths[idx])
        row = np.array(self._data[off:off + ln], dtype=np.float32, copy=True)
        sample: dict[str, Any] = {"features": torch.from_numpy(row), "length": ln}
        if self._tmr is not None:
            if self._tmr_per_token:
                o = int(self._tmr_index[idx, 0]); k = int(self._tmr_index[idx, 1])
                emb = np.array(self._tmr[o:o + k], dtype=np.float32, copy=True)
            else:
                emb = np.array(self._tmr[idx], dtype=np.float32, copy=True)
            sample["tmr_emb"] = torch.from_numpy(emb)
        return sample

class DistributedLengthBucketSampler(Sampler[list[int]]):
    """Yields fixed-size batches of similar-length clips, DDP-sharded.

    Every rank produces the SAME number of batches (``n // world // bs``) so DDP
    stays in step. Within a rank, indices are sorted by length inside megabatches
    of ``bs * megabatch_mult`` to keep batches length-homogeneous while retaining
    epoch-to-epoch randomness; batch order is then shuffled.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        megabatch_mult: int = 50,
    ) -> None:
        self.lengths = np.asarray(lengths)
        self.n = int(self.lengths.shape[0])
        self.bs = int(batch_size)
        self.world = max(1, int(num_replicas))
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.megabatch_mult = int(megabatch_mult)
        self.epoch = 0
        self.per_rank = self.n // self.world
        self.num_batches = self.per_rank // self.bs
        if self.num_batches == 0:
            raise ValueError(
                f"batch_size={self.bs} too large for {self.n} clips over {self.world} ranks"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        g = np.random.default_rng(self.seed + self.epoch)
        idx = np.arange(self.n)
        if self.shuffle:
            g.shuffle(idx)
        # strided partition -> this rank's balanced subset
        rank_idx = idx[self.rank:self.per_rank * self.world:self.world]
        # length-bucket inside megabatches
        mb = max(self.bs, self.bs * self.megabatch_mult)
        chunks = []
        for s in range(0, len(rank_idx), mb):
            c = rank_idx[s:s + mb]
            c = c[np.argsort(self.lengths[c], kind="stable")]
            chunks.append(c)
        order = np.concatenate(chunks) if chunks else rank_idx
        batches = [order[i * self.bs:(i + 1) * self.bs].tolist() for i in range(self.num_batches)]
        if self.shuffle:
            for j in g.permutation(len(batches)):
                yield batches[j]
        else:
            for b in batches:
                yield b

def collate_varlen_crop(
    batch: list[dict[str, Any]], token_stride: int = 4
) -> dict[str, Tensor]:
    """Crop every clip to the batch's shortest length (floored to token_stride).

    A random start is used as light temporal augmentation. Per-clip TMR
    teacher (``[dsem]``) is crop-invariant -> passed as-is; per-token teacher
    (``[n_tok, dsem]``) must stay frame-aligned, so the crop start is floored
    to ``token_stride`` and the teacher sliced to the same token span
    ``[start//stride, start//stride + L//stride)`` -> ``[B, L//stride, dsem]``.
    """
    lens = [int(s["features"].shape[0]) for s in batch]
    L = (min(lens) // token_stride) * token_stride
    L = max(L, token_stride)
    n_tok = L // token_stride
    per_token = "tmr_emb" in batch[0] and batch[0]["tmr_emb"].dim() == 2

    feats = []
    tmrs = []
    for s in batch:
        T = int(s["features"].shape[0])
        if T <= L:
            start = 0
        elif per_token:
            # stride-aligned start keeps feature frames <-> teacher tokens in lock-step
            start = int(torch.randint(0, (T - L) // token_stride + 1, (1,)).item()) * token_stride
        else:
            start = int(torch.randint(0, T - L + 1, (1,)).item())
        feats.append(s["features"][start:start + L])
        if "tmr_emb" in s:
            if per_token:
                t0 = start // token_stride
                tmrs.append(s["tmr_emb"][t0:t0 + n_tok])
            else:
                tmrs.append(s["tmr_emb"])
    out: dict[str, Tensor] = {"features": torch.stack(feats)}
    if tmrs:
        out["tmr_emb"] = torch.stack(tmrs)  # perwin: [B, dsem]; per-token: [B, n_tok, dsem]
    return out

__all__ = [
    "VarlenUMRDataset",
    "DistributedLengthBucketSampler",
    "collate_varlen_crop",
    "varlen_cache_is_ready",
]
