"""Unit tests for the sliding-window training cache + cached dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.umr_cached_dataset import (
    CachedUMRFeatureDataset,
    cache_is_ready,
    collate_cached_umr,
)
from data.umr_schema import (
    DIM_FEATURES,
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
from tools.build_training_cache import main as build_training_cache_main

def _make_umr499(T: int, *, rng: np.random.Generator, path: Path) -> Path:
    fields = FeatureFields(
        root_traj=rng.standard_normal((T - 1, 9)).astype(np.float32),
        root_rot6d=rng.standard_normal((T - 1, DIM_ROOT_ROT6D)).astype(np.float32),
        joints76_rot6d=rng.standard_normal((T - 1, UMR_NUM_JOINTS76, 6)).astype(np.float32),
        sparse_vel=rng.standard_normal((T - 1, NUM_SPARSE_VEL_JOINTS, 3)).astype(np.float32),
        foot_contact=(rng.random((T - 1, DIM_FOOT_CONTACT)) > 0.5).astype(np.float32),
    )
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
    umr.to_npz(path)
    return path

def _build_cache(
    tmp_path: Path,
    *,
    clip_frames: list[int],
    window: int = 7,
    step: int = 3,
    cap: int = 16,
) -> tuple[Path, Path, list[Path]]:
    """Create synthetic clips + manifest and build the cache.

    Returns ``(cache_path, manifest_path, clip_paths)``.
    """
    rng = np.random.default_rng(42)
    rec_paths: list[Path] = []
    for i, T in enumerate(clip_frames):
        p = tmp_path / "recordings" / f"rec_{i}" / "umr499.npz"
        _make_umr499(T, rng=rng, path=p)
        rec_paths.append(p)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(str(p) for p in rec_paths) + "\n")

    cache_path = tmp_path / "cache" / "umr499_train.npy"
    rc = build_training_cache_main(
        [
            "--manifest",
            str(manifest),
            "--out",
            str(cache_path),
            "--window",
            str(window),
            "--step",
            str(step),
            "--max-windows-per-clip",
            str(cap),
            "--workers",
            "1",
        ]
    )
    assert rc == 0
    return cache_path, manifest, rec_paths

def test_build_training_cache_basic(tmp_path) -> None:
    cache_path, _, _ = _build_cache(
        tmp_path,
        clip_frames=[20, 12],  # window=7 records, step=3 → both contribute windows
        window=7,
        step=3,
        cap=16,
    )
    meta = json.loads((cache_path.with_suffix(cache_path.suffix + ".meta.json")).read_text())
    assert meta["dtype"] == "float32"
    assert meta["window"] == 7
    assert meta["step"] == 3
    assert meta["shape"][2] == DIM_FEATURES

    arr = np.load(cache_path, mmap_mode="r")
    assert arr.dtype == np.float32
    assert arr.shape == tuple(meta["shape"])
    # clip 0: N=20 → last_start=12 → starts 0,3,6,9,12 → 5 windows
    # clip 1: N=12 → last_start=4 → starts 0,3,4 (dedup not required; 3 already < 4 so last appended) → 3 windows
    assert arr.shape[0] == 5 + 3
    assert all(s["start_frame"] >= 0 for s in meta["samples"])

def test_build_training_cache_skips_short_clip(tmp_path) -> None:
    cache_path, _, _ = _build_cache(
        tmp_path,
        clip_frames=[20, 5],  # second clip has 4 records < window=7
        window=7,
        step=3,
        cap=16,
    )
    meta = json.loads((cache_path.with_suffix(cache_path.suffix + ".meta.json")).read_text())
    assert meta["num_clips_short"] == 1
    assert meta["shape"][0] == 5  # only clip 0 contributed

def test_cached_dataset_returns_features(tmp_path) -> None:
    cache_path, _, _ = _build_cache(
        tmp_path,
        clip_frames=[20, 12],
        window=7,
        step=3,
        cap=16,
    )
    ds = CachedUMRFeatureDataset(cache_path, in_ram=False)
    assert cache_is_ready(cache_path)
    assert len(ds) == 8

    sample = ds[0]
    assert set(sample.keys()) == {"features"}
    assert isinstance(sample["features"], torch.Tensor)
    assert sample["features"].dtype == torch.float32
    assert sample["features"].shape == (7, DIM_FEATURES)

    batch = collate_cached_umr([ds[0], ds[1], ds[7]])
    assert batch["features"].shape == (3, 7, DIM_FEATURES)

def test_cached_dataset_matches_source_features(tmp_path) -> None:
    cache_path, _, rec_paths = _build_cache(
        tmp_path,
        clip_frames=[20],
        window=7,
        step=3,
        cap=16,
    )
    ds = CachedUMRFeatureDataset(cache_path, in_ram=False)

    # Expect first window to be features[0:7] of the only clip.
    umr = UMR499.from_npz(rec_paths[0])
    np.testing.assert_allclose(
        ds[0]["features"].numpy(),
        umr.features[0:7].astype(np.float32),
        rtol=0.0,
        atol=0.0,
    )
    # Expect third window (starts=0,3,6,9,12) to be features[6:13].
    np.testing.assert_allclose(
        ds[2]["features"].numpy(),
        umr.features[6:13].astype(np.float32),
        rtol=0.0,
        atol=0.0,
    )

def test_cache_rebuild_is_idempotent(tmp_path, capsys) -> None:
    cache_path, manifest, _ = _build_cache(
        tmp_path,
        clip_frames=[20, 12],
        window=7,
        step=3,
        cap=16,
    )
    first_mtime = cache_path.stat().st_mtime_ns
    capsys.readouterr()  # drop first-build chatter

    rc = build_training_cache_main(
        [
            "--manifest",
            str(manifest),
            "--out",
            str(cache_path),
            "--window",
            "7",
            "--step",
            "3",
            "--max-windows-per-clip",
            "16",
            "--workers",
            "1",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "cache already valid" in captured.out
    assert cache_path.stat().st_mtime_ns == first_mtime

def test_cached_dataset_in_ram_mode(tmp_path) -> None:
    cache_path, _, rec_paths = _build_cache(
        tmp_path,
        clip_frames=[20],
        window=7,
        step=3,
        cap=16,
    )
    ds = CachedUMRFeatureDataset(cache_path, in_ram=True)
    assert ds.features is not None and ds.features.is_shared()
    umr = UMR499.from_npz(rec_paths[0])
    np.testing.assert_allclose(
        ds[0]["features"].numpy(), umr.features[0:7].astype(np.float32)
    )

def test_cap_limits_windows_per_clip(tmp_path) -> None:
    cache_path, _, _ = _build_cache(
        tmp_path,
        clip_frames=[200],  # would produce many windows
        window=7,
        step=3,
        cap=4,
    )
    meta = json.loads((cache_path.with_suffix(cache_path.suffix + ".meta.json")).read_text())
    assert meta["shape"][0] == 4
