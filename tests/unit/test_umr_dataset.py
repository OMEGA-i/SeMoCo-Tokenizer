"""``UMRDataset`` unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.umr_dataset import UMRDataset, collate_umr
from data.umr_schema import (
    DIM_FEATURES,
    DIM_FOOT_CONTACT,
    DIM_ROOT_ROT6D,
    DIM_TRAJ_CONTACT,
    FEATURE_VARIANT,
    UMR_FPS,
    UMR_NUM_JOINTS,
    UMR_NUM_JOINTS76,
    CanonicalAnchor,
    FeatureFields,
    UMR499,
    pack_features,
)
from data.umr_schema import NUM_SPARSE_VEL_JOINTS

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

def _write_manifest(paths: list[Path], manifest_path: Path) -> Path:
    manifest_path.write_text("\n".join(str(p) for p in paths) + "\n")
    return manifest_path

def test_dataset_loads_full_window(tmp_path) -> None:
    rng = np.random.default_rng(0)
    rec1 = tmp_path / "recA" / "umr499.npz"
    _make_umr499(200, rng=rng, path=rec1)
    rec2 = tmp_path / "recB" / "umr499.npz"
    _make_umr499(181, rng=rng, path=rec2)

    manifest = _write_manifest([rec1, rec2], tmp_path / "all.txt")
    ds = UMRDataset(manifest)
    assert len(ds) == 2

    s = ds[0]
    assert s["features"].shape == (180, DIM_FEATURES)
    assert s["traj_contact"].shape == (180, DIM_TRAJ_CONTACT)
    assert s["joints76_rot6d"].shape == (180, UMR_NUM_JOINTS76 * 6)
    assert s["foot_contact"].shape == (180, DIM_FOOT_CONTACT)
    assert s["init_root_pos"].shape == (3,)
    assert s["init_root_rot6d"].shape == (DIM_ROOT_ROT6D,)
    assert s["init_joints76_rot6d"].shape == (UMR_NUM_JOINTS76, 6)
    assert s["joints77_pos"].shape == (181, UMR_NUM_JOINTS, 3)
    assert s["identity_coeffs"].shape == (1, 10)
    assert s["joint_orient"].shape == (78, 3, 3)

def test_dataset_collate(tmp_path) -> None:
    rng = np.random.default_rng(1)
    rec1 = tmp_path / "recA" / "umr499.npz"
    _make_umr499(190, rng=rng, path=rec1)
    rec2 = tmp_path / "recB" / "umr499.npz"
    _make_umr499(181, rng=rng, path=rec2)
    manifest = _write_manifest([rec1, rec2], tmp_path / "all.txt")
    ds = UMRDataset(manifest)
    batch = collate_umr([ds[0], ds[1]])
    assert batch["features"].shape == (2, 180, DIM_FEATURES)
    assert batch["joints77_pos"].shape == (2, 181, UMR_NUM_JOINTS, 3)
    assert batch["identity_coeffs"].shape == (2, 1, 10)

def test_dataset_rejects_short_clip(tmp_path) -> None:
    rng = np.random.default_rng(2)
    rec = tmp_path / "short" / "umr499.npz"
    _make_umr499(50, rng=rng, path=rec)
    manifest = _write_manifest([rec], tmp_path / "all.txt")
    ds = UMRDataset(manifest)
    with pytest.raises(RuntimeError):
        ds[0]

def test_dataset_recordings_root_resolution(tmp_path) -> None:
    rng = np.random.default_rng(3)
    rec_id = "rec_xyz"
    out = tmp_path / "recordings" / rec_id / "umr499.npz"
    _make_umr499(190, rng=rng, path=out)
    manifest = tmp_path / "all.txt"
    manifest.write_text(f"{rec_id}\n# comment\n\n")
    ds = UMRDataset(manifest, recordings_root=tmp_path / "recordings")
    assert ds[0]["features"].shape == (180, DIM_FEATURES)
