"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.soma77_schema import NUM_JOINTS, NUM_RIG_JOINTS, Soma77Canonical

def _make_synthetic_canonical(
    T: int = 100,
    *,
    fps_src: float = 30.0,
    seed: int = 0,
    contact_pattern: str = "alternating",
) -> Soma77Canonical:
    """Build a small synthetic SOMA77 payload for unit tests.

    ``contact_pattern``:
      * ``"alternating"`` — left foot 1 / right foot 0 then swap every 5 frames.
      * ``"none"``         — all-zero (still passes shape checks).
      * ``"all"``          — all-one.
    """
    rng = np.random.RandomState(seed)
    poses = (rng.randn(T, NUM_JOINTS, 3) * 0.3).astype(np.float32)
    transl = (rng.randn(T, 3) * 0.05).astype(np.float32)
    transl[:, 1] = 0.95 + 0.02 * rng.randn(T).astype(np.float32)
    transl[:, 0] = np.cumsum(rng.randn(T).astype(np.float32) * 0.01)
    transl[:, 2] = np.cumsum(rng.randn(T).astype(np.float32) * 0.01)
    identity_coeffs = np.zeros((1, 10), dtype=np.float32)
    joint_orient = np.tile(np.eye(3, dtype=np.float32)[None], (NUM_RIG_JOINTS, 1, 1))

    if contact_pattern == "alternating":
        contacts = np.zeros((T, 4), dtype=np.float32)
        for k in range(0, T, 5):
            phase = (k // 5) % 2
            contacts[k : k + 5, 0 if phase == 0 else 2] = 1.0
            contacts[k : k + 5, 1 if phase == 0 else 3] = 1.0
    elif contact_pattern == "none":
        contacts = np.zeros((T, 4), dtype=np.float32)
        contacts[0, 0] = 1.0  # avoid all-zero validator failure
    elif contact_pattern == "all":
        contacts = np.ones((T, 4), dtype=np.float32)
        contacts[0, :] = 0.0  # avoid all-one
    else:
        raise ValueError(f"unknown contact_pattern={contact_pattern!r}")

    return Soma77Canonical(
        poses=poses,
        transl=transl,
        identity_coeffs=identity_coeffs,
        joint_orient=joint_orient,
        foot_contacts=contacts,
        fps_src=fps_src,
    )

@pytest.fixture
def synthetic_canonical_short() -> Soma77Canonical:
    """T=100 frames, alternating contact, fps=30."""
    return _make_synthetic_canonical(T=100)

@pytest.fixture
def synthetic_canonical_long() -> Soma77Canonical:
    """T=300 frames, fps=30 (resamples to ~500 frames at fps=50, supports many crops)."""
    return _make_synthetic_canonical(T=300)

@pytest.fixture
def write_synthetic_npz(tmp_path):
    """Factory: writes a canonical payload to ``tmp_path`` and returns its npz path."""

    def _write(
        T: int = 100,
        *,
        rec_id: str = "rec_v0-test",
        fps_src: float = 30.0,
        contact_pattern: str = "alternating",
    ) -> Path:
        canonical = _make_synthetic_canonical(
            T=T, fps_src=fps_src, contact_pattern=contact_pattern
        )
        rec_dir = tmp_path / "recordings" / rec_id
        rec_dir.mkdir(parents=True, exist_ok=True)
        out = rec_dir / "soma77.npz"
        canonical.to_npz(out)
        manifest = rec_dir / "manifest.json"
        manifest.write_text(f'{{"canonical": {{"fps": {fps_src}}}}}\n')
        return out

    return _write
