"""Typed schema for the canonical SOMA77 ``.npz`` payload.

Fields::

    poses           [T, 77, 3]   float32   T-pose-relative axis-angle / rotvec
    transl          [T, 3]       float32   root / Hips world translation, meters
    identity_coeffs [1|T, C]     float32   body identity / shape (SMPL-X betas-like)
    joint_orient    [78, 3, 3]   float32   SOMA-X rig rest joint orientations
    foot_contacts   [T, 4]       float32   [LeftFoot, LeftToeBase, RightFoot, RightToeBase]

``identity_coeffs`` and ``joint_orient`` are static across the clip and never
enter the UMR token stream. ``foot_contacts`` is required and not derived
from FK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Per-clip skeleton invariants. Anchored to the SOMA-X rig.
NUM_JOINTS = 77
"""Number of body joints in the SOMA77 skeleton (root included)."""

NUM_NON_ROOT_JOINTS = 76
"""SOMA77 minus root. Disambiguates from the part-aware "body 20" split."""

NUM_RIG_JOINTS = 78
"""SOMA-X rig has a virtual ``Root`` parent of ``Hips``; ``joint_orient`` keeps it."""

DEFAULT_IDENTITY_DIM = 10
"""SMPL-X betas dimension. Some pipelines emit additional shape coefficients."""

FOOT_CONTACT_NAMES: tuple[str, str, str, str] = (
    "LeftFoot",
    "LeftToeBase",
    "RightFoot",
    "RightToeBase",
)
FOOT_CONTACT_SOMA77_INDICES: tuple[int, int, int, int] = (69, 70, 74, 75)

@dataclass
class Soma77Canonical:
    """In-memory view of one ``<rec_id>/soma77.npz`` payload.

    ``poses[t, 0]`` is the root global rotation, ``poses[t, 1:]`` parent-local.
    ``identity_coeffs`` is ``[1, C]`` or ``[T, C]`` (per-clip-constant ``[T, C]``
    compresses to ``[1, C]`` at load). ``joint_orient`` is rig metadata (78 =
    77 joints + virtual ``Root``), fixed across the clip. ``fps_src`` is read
    from ``manifest.json::canonical.fps`` when available, else 30.
    """

    poses: np.ndarray
    transl: np.ndarray
    identity_coeffs: np.ndarray
    joint_orient: np.ndarray
    foot_contacts: np.ndarray
    fps_src: float = 30.0
    source_path: Path | None = field(default=None, compare=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, npz_path: str | Path, *, fps_hint: float | None = None) -> "Soma77Canonical":
        """Read a ``<rec_id>/soma77.npz`` file.

        fps priority: explicit ``fps_hint`` > ``soma77.npz['fps']`` field >
        sibling ``manifest.json::canonical.fps`` > 30 FPS fallback.
        """
        import json

        p = Path(npz_path)
        data = np.load(p, allow_pickle=False)
        missing = {"poses", "transl", "identity_coeffs", "joint_orient", "foot_contacts"} - set(data.files)
        if missing:
            raise ValueError(
                f"{p}: SOMA77 canonical npz missing required fields: {sorted(missing)}; "
                f"got {sorted(data.files)}"
            )

        poses = np.asarray(data["poses"], dtype=np.float32)
        transl = np.asarray(data["transl"], dtype=np.float32)
        identity_coeffs = np.asarray(data["identity_coeffs"], dtype=np.float32)
        joint_orient = np.asarray(data["joint_orient"], dtype=np.float32)
        foot_contacts = np.asarray(data["foot_contacts"], dtype=np.float32)

        if fps_hint is not None:
            fps_src = float(fps_hint)
        elif "fps" in data.files:
            fps_src = float(np.asarray(data["fps"]).item())
        else:
            fps_src = _resolve_fps_from_manifest(p)
        if not (fps_src > 0.0):
            fps_src = DEFAULT_FALLBACK_FPS

        obj = cls(
            poses=poses,
            transl=transl,
            identity_coeffs=identity_coeffs,
            joint_orient=joint_orient,
            foot_contacts=foot_contacts,
            fps_src=float(fps_src),
            source_path=p,
        )
        obj.validate_shapes()
        return obj

    # ------------------------------------------------------------------
    # Sanity checks (cheap; fuller L0 lives in data/validation.py)
    # ------------------------------------------------------------------

    def validate_shapes(self) -> None:
        """Cheap shape / dtype checks. Heavier numeric L0 lives elsewhere."""
        T = int(self.poses.shape[0])
        if self.poses.ndim != 3 or self.poses.shape[1] != NUM_JOINTS or self.poses.shape[2] != 3:
            raise ValueError(
                f"{self._tag}: poses shape {self.poses.shape} must be (T={T}, 77, 3)"
            )
        if self.transl.shape != (T, 3):
            raise ValueError(
                f"{self._tag}: transl shape {self.transl.shape} must be ({T}, 3)"
            )
        if self.identity_coeffs.ndim != 2 or self.identity_coeffs.shape[0] not in (1, T):
            raise ValueError(
                f"{self._tag}: identity_coeffs shape {self.identity_coeffs.shape} "
                f"must be (1, C) or ({T}, C)"
            )
        if self.joint_orient.shape != (NUM_RIG_JOINTS, 3, 3):
            raise ValueError(
                f"{self._tag}: joint_orient shape {self.joint_orient.shape} "
                f"must be ({NUM_RIG_JOINTS}, 3, 3)"
            )
        if self.foot_contacts.shape != (T, 4):
            raise ValueError(
                f"{self._tag}: foot_contacts shape {self.foot_contacts.shape} "
                f"must be ({T}, 4)"
            )

    @property
    def num_frames(self) -> int:
        return int(self.poses.shape[0])

    @property
    def identity_dim(self) -> int:
        return int(self.identity_coeffs.shape[-1])

    @property
    def _tag(self) -> str:
        return str(self.source_path) if self.source_path else f"<Soma77Canonical T={self.num_frames}>"

    def to_npz(self, out_path: str | Path, *, compressed: bool = True) -> Path:
        """Write the payload back to disk in the canonical schema."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        save = np.savez_compressed if compressed else np.savez
        save(
            out,
            poses=self.poses.astype(np.float32, copy=False),
            transl=self.transl.astype(np.float32, copy=False),
            identity_coeffs=self.identity_coeffs.astype(np.float32, copy=False),
            joint_orient=self.joint_orient.astype(np.float32, copy=False),
            foot_contacts=self.foot_contacts.astype(np.float32, copy=False),
        )
        return out

    def collapse_constant_identity(self, *, atol: float = 1e-5) -> "Soma77Canonical":
        """Compress per-frame ``identity_coeffs`` to ``[1, C]`` if approximately constant.

        Returns ``self`` unchanged if already ``[1, C]`` or if the per-frame
        variance exceeds ``atol``.
        """
        if self.identity_coeffs.shape[0] == 1:
            return self
        per_dim_std = self.identity_coeffs.std(axis=0)
        if float(per_dim_std.max(initial=0.0)) <= atol:
            mean = self.identity_coeffs.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
            return Soma77Canonical(
                poses=self.poses,
                transl=self.transl,
                identity_coeffs=mean,
                joint_orient=self.joint_orient,
                foot_contacts=self.foot_contacts,
                fps_src=self.fps_src,
                source_path=self.source_path,
            )
        return self

# ---------------------------------------------------------------------------
# fps resolution helpers
# ---------------------------------------------------------------------------

DEFAULT_FALLBACK_FPS = 30.0
"""Used only when neither ``manifest.json`` nor a caller-supplied hint is available."""

def _resolve_fps_from_manifest(soma77_npz: Path) -> float:
    """Read ``canonical.fps`` from the sibling ``manifest.json``.

    Flat layout: ``<rec_id>/soma77.npz`` next to ``<rec_id>/manifest.json``.
    Falls back to :data:`DEFAULT_FALLBACK_FPS` on any failure (missing file,
    malformed JSON, missing or non-numeric key).
    """
    import json

    rec_dir = soma77_npz.parent
    manifest_path = rec_dir / "manifest.json"
    if not manifest_path.is_file():
        return DEFAULT_FALLBACK_FPS
    try:
        with manifest_path.open("r") as f:
            manifest: Any = json.load(f)
    except Exception:
        return DEFAULT_FALLBACK_FPS
    canonical = manifest.get("canonical") if isinstance(manifest, dict) else None
    if not isinstance(canonical, dict):
        return DEFAULT_FALLBACK_FPS
    fps = canonical.get("fps")
    if fps is None:
        return DEFAULT_FALLBACK_FPS
    try:
        return float(fps)
    except (TypeError, ValueError):
        return DEFAULT_FALLBACK_FPS
