"""Differentiable SOMA77 forward kinematics for the ``fk_geom`` joint loss.

Parent-local rot6d rotations are composed along the SOMA77 kinematic tree on a
fixed template skeleton (parent ids + rest bone offsets) to produce root-relative
joint XYZ. Rec and GT pass through the same template, so bone lengths cancel and
the term measures rotation -> position error (an MPJPE analogue). The template is
materialized once offline (:func:`build_soma77_template`) as a small ``.npz``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

SOMA77_NUM_JOINTS = 77
TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "data" / "assets" / "soma77_template.npz"

def rot6d_to_rotmat(d6: Tensor) -> Tensor:
    """6D -> rotation matrix, UMR column convention.

    Matches :func:`data.umr_to_soma77.rot6d_to_rotmat` (columns ``b1,b2,b3``)
    so the FK frame is consistent with how ``joints77_pos`` GT is decoded.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns

class SomaFK(nn.Module):
    """Differentiable SOMA77 FK on a fixed template skeleton.

    forward(rot6d_77 ``[..., 77, 6]``) -> joints ``[..., 77, 3]`` (root at origin).
    """

    def __init__(self, template_path: str | Path = TEMPLATE_PATH) -> None:
        super().__init__()
        path = Path(template_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"SOMA77 FK template missing: {path}. Generate it once with "
                f"`python -m tools.build_soma77_template` (requires the [soma] extras)."
            )
        data = np.load(path)
        parents = np.asarray(data["parents"], dtype=np.int64)
        offsets = np.asarray(data["offsets"], dtype=np.float32)
        if parents.shape != (SOMA77_NUM_JOINTS,) or offsets.shape != (SOMA77_NUM_JOINTS, 3):
            raise ValueError(
                f"bad template shapes parents={parents.shape} offsets={offsets.shape}"
            )
        self.register_buffer("offsets", torch.from_numpy(offsets))  # [77, 3] rel. parent
        self._parents: list[int] = [int(p) for p in parents.tolist()]
        self._order: list[int] = _topological_order(self._parents)

    def forward(self, rot6d_77: Tensor) -> Tensor:
        if rot6d_77.shape[-2:] != (SOMA77_NUM_JOINTS, 6):
            raise ValueError(f"expected [..., 77, 6]; got {tuple(rot6d_77.shape)}")
        rmat = rot6d_to_rotmat(rot6d_77)  # [..., 77, 3, 3]
        lead = rot6d_77.shape[:-2]
        offsets = self.offsets.to(rot6d_77.dtype)
        r_world: list[Tensor] = [rot6d_77.new_zeros(())] * SOMA77_NUM_JOINTS
        pos: list[Tensor] = [rot6d_77.new_zeros(())] * SOMA77_NUM_JOINTS
        for j in self._order:
            p = self._parents[j]
            r_j = rmat[..., j, :, :]
            if p < 0:
                r_world[j] = r_j
                pos[j] = rot6d_77.new_zeros((*lead, 3))
            else:
                r_world[j] = r_world[p] @ r_j
                off_j = offsets[j].view(3, 1)
                disp = (r_world[p] @ off_j).squeeze(-1)  # [..., 3]
                pos[j] = pos[p] + disp
        return torch.stack(pos, dim=-2)  # [..., 77, 3], root at origin

def _topological_order(parents: list[int]) -> list[int]:
    """Parent-before-child visit order for the FK loop (any input ordering)."""
    order: list[int] = []
    done = [False] * len(parents)
    while len(order) < len(parents):
        progressed = False
        for j, p in enumerate(parents):
            if done[j]:
                continue
            if p < 0 or done[p]:
                order.append(j)
                done[j] = True
                progressed = True
        if not progressed:
            raise ValueError("SOMA77 template parents do not form a tree")
    return order

def build_soma77_template(
    out_path: str | Path,
    *,
    device: str = "cpu",
) -> Path:
    """One-time: materialize ``{parents[77], offsets[77,3]}`` from SOMA-X.

    Poses the rig at rest (identity rotations, SOMA mean body) and reads joint
    world positions; bone offsets are relative to the parent joint. The SOMA
    identity model ships with the SOMA-X assets, so no licensed SMPL-X download
    is needed; absolute bone lengths cancel in the FK loss anyway.
    """
    from data.soma77_fk import soma77_joints_world_xyz_from_matrices, soma77_parent_indices

    parents = np.asarray(soma77_parent_indices(), dtype=np.int64)
    ident = np.zeros((1, 128), dtype=np.float32)  # SOMA mean body (zero PCA coeffs)
    eye = np.broadcast_to(np.eye(3, dtype=np.float32), (1, SOMA77_NUM_JOINTS, 3, 3)).copy()
    transl = np.zeros((1, 3), dtype=np.float32)
    rest = soma77_joints_world_xyz_from_matrices(
        eye, transl, ident, device=device, identity_model_type="soma"
    )[0]  # [77,3]
    offsets = np.zeros_like(rest)
    for j, p in enumerate(parents.tolist()):
        offsets[j] = rest[j] - rest[p] if p >= 0 else 0.0
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, parents=parents, offsets=offsets.astype(np.float32))
    return out
