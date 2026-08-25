"""Field-aware UMR reconstruction loss.

:class:`UMRLoss` consumes the 499D feature reconstruction (channels-first
``[B, 499, T]``) and matching ground truth, returning a typed
:class:`UMRLossOutput` with ``L_total = sum_i w_i * L_i + w_vq * L_vq +
w_ortho * L_rot6d_ortho`` over field components: L2 on root_traj (9D),
root_rot6d (6D), joints76_rot6d (456D), sparse_vel (24D); BCE-with-logits on
foot_contact (4D); optional rot6d_ortho ‖b1·b2‖². ``foot_contact`` channels in
``features_rec`` are raw logits; the VQ loss is a tokenizer-output passthrough.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from data.umr_schema import (
    DIM_FEATURES,
    DIM_FOOT_CONTACT,
    FOOT_CONTACT_SOMA77_INDICES,
    SLICE_FOOT_CONTACT,
    SLICE_JOINTS76_ROT6D,
    SLICE_ROOT_ROT6D,
    SLICE_ROOT_TRAJ,
    SLICE_SPARSE_VEL,
    UMR_NUM_JOINTS76,
)
from losses.contact_losses import contact_bce_loss


@dataclass
class UMRLossWeights:
    """Per-field loss weights."""

    root_traj: float = 1.0
    root_rot6d: float = 1.0
    joints76_rot6d: float = 1.0
    sparse_vel: float = 0.5
    foot_contact: float = 0.5
    vq: float = 1.0
    rot6d_ortho: float = 0.0


@dataclass
class UMRLossOutput:
    """Typed loss components."""

    total: Tensor
    root_traj: Tensor
    root_rot6d: Tensor
    joints76_rot6d: Tensor
    sparse_vel: Tensor
    foot_contact: Tensor
    vq: Tensor
    rot6d_ortho: Tensor
    recon_smooth: Tensor | None = None
    root_header: Tensor | None = None
    fk_joint: Tensor | None = None
    fk_vel: Tensor | None = None
    fk_acc: Tensor | None = None
    fk_footskate: Tensor | None = None

    def to_dict(self) -> dict[str, Tensor]:
        out = {
            "total": self.total,
            "root_traj": self.root_traj,
            "root_rot6d": self.root_rot6d,
            "joints76_rot6d": self.joints76_rot6d,
            "sparse_vel": self.sparse_vel,
            "foot_contact": self.foot_contact,
            "vq": self.vq,
            "rot6d_ortho": self.rot6d_ortho,
        }
        if self.recon_smooth is not None:
            out["recon_smooth"] = self.recon_smooth
        if self.root_header is not None:
            out["root_header"] = self.root_header
        for name in ("fk_joint", "fk_vel", "fk_acc", "fk_footskate"):
            val = getattr(self, name)
            if val is not None:
                out[name] = val
        return out


class UMRLoss(torch.nn.Module):
    """Field-aware reconstruction loss for the 499D UMR record."""

    def __init__(
        self,
        *,
        weights: UMRLossWeights | None = None,
        contact_pos_weight: float | None = None,
        loss_mode: Literal["field", "smooth_recon", "fk_geom"] = "field",
        root_header_alpha: float = 0.0,
        loss_vel: float = 0.0,
        fk_weight: float = 1.0,
        fk_vel_weight: float = 0.0,
        fk_acc_weight: float = 0.0,
        fk_footskate_weight: float = 0.0,
        fk_template_path: str | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or UMRLossWeights()
        self.contact_pos_weight = contact_pos_weight
        self.loss_mode = loss_mode
        self.root_header_alpha = float(root_header_alpha)
        self.loss_vel = float(loss_vel)
        self.fk_weight = float(fk_weight)
        self.fk_vel_weight = float(fk_vel_weight)
        self.fk_acc_weight = float(fk_acc_weight)
        self.fk_footskate_weight = float(fk_footskate_weight)
        self._fk_template_path = fk_template_path
        self._fk: torch.nn.Module | None = None
        if self.loss_mode not in {"field", "smooth_recon", "fk_geom"}:
            raise ValueError(f"unknown loss_mode {self.loss_mode!r}")

    def _ensure_fk(self, device: torch.device) -> torch.nn.Module:
        if self._fk is None:
            from losses.fk_geom import SomaFK, TEMPLATE_PATH

            self._fk = SomaFK(self._fk_template_path or TEMPLATE_PATH).to(device)
        return self._fk

    def _fk_joints(self, features: Tensor) -> Tensor:
        """``[B, 499, T]`` features -> root-relative joints ``[B, T, 77, 3]``.

        ``features`` must already be de-normalized (real-scale rot6d).
        """
        root6 = features[:, SLICE_ROOT_ROT6D, :].transpose(1, 2)            # [B, T, 6]
        joints6 = features[:, SLICE_JOINTS76_ROT6D, :].transpose(1, 2)      # [B, T, 456]
        b, t = root6.shape[0], root6.shape[1]
        joints6 = joints6.reshape(b, t, UMR_NUM_JOINTS76, 6)
        rot6d_77 = torch.cat([root6.unsqueeze(2), joints6], dim=2)          # [B, T, 77, 6]
        fk = self._ensure_fk(features.device)
        return fk(rot6d_77.float())

    def forward(
        self,
        features_rec: Tensor,                # [B, 499, T]   foot_contact = LOGITS
        features_gt: Tensor,                 # [B, 499, T]   foot_contact = {0, 1}
        *,
        vq_loss: Tensor | float = 0.0,
        norm: object | None = None,          # FeatureNormalizationLayer (for fk_geom denorm)
    ) -> UMRLossOutput:
        if features_rec.shape != features_gt.shape:
            raise ValueError(
                f"features_rec / features_gt shapes differ: "
                f"{tuple(features_rec.shape)} vs {tuple(features_gt.shape)}"
            )
        if features_rec.shape[1] != DIM_FEATURES:
            raise ValueError(
                f"features channel dim {features_rec.shape[1]} != {DIM_FEATURES}"
            )

        w = self.weights
        l_vq = (
            vq_loss.to(features_rec.dtype)
            if isinstance(vq_loss, Tensor)
            else features_rec.new_tensor(float(vq_loss))
        )

        if self.loss_mode == "fk_geom":
            l_recon = F.smooth_l1_loss(features_rec, features_gt)
            rec_root = torch.cat(
                [features_rec[:, SLICE_ROOT_TRAJ], features_rec[:, SLICE_ROOT_ROT6D]], dim=1
            )
            gt_root = torch.cat(
                [features_gt[:, SLICE_ROOT_TRAJ], features_gt[:, SLICE_ROOT_ROT6D]], dim=1
            )
            l_root_header = F.smooth_l1_loss(rec_root, gt_root)

            # FK joint-position term: rot6d is z-scored in feature space, so
            # de-normalize first; rec/gt share one template (bone lengths cancel).
            if norm is not None:
                rec_d = norm.inverse(features_rec)
                gt_d = norm.inverse(features_gt)
            else:
                rec_d, gt_d = features_rec, features_gt
            rec_j = self._fk_joints(rec_d)                  # [B, T, 77, 3]
            gt_j = self._fk_joints(gt_d)
            l_fk = F.l1_loss(rec_j, gt_j)

            zero = features_rec.new_zeros(())
            l_fk_vel = zero
            l_fk_acc = zero
            l_fk_skate = zero
            if self.fk_vel_weight > 0 or self.fk_acc_weight > 0:
                rec_v = rec_j[:, 1:] - rec_j[:, :-1]
                gt_v = gt_j[:, 1:] - gt_j[:, :-1]
                if self.fk_vel_weight > 0:
                    l_fk_vel = F.l1_loss(rec_v, gt_v)
                if self.fk_acc_weight > 0 and rec_j.shape[1] >= 3:
                    l_fk_acc = F.l1_loss(rec_v[:, 1:] - rec_v[:, :-1], gt_v[:, 1:] - gt_v[:, :-1])
            if self.fk_footskate_weight > 0:
                # Penalize foot-joint motion on GT-contact frames (anti-sliding).
                foot_idx = list(FOOT_CONTACT_SOMA77_INDICES)
                foot_v = rec_j[:, 1:, foot_idx, :] - rec_j[:, :-1, foot_idx, :]  # [B,T-1,4,3]
                contact = features_gt[:, SLICE_FOOT_CONTACT, :].transpose(1, 2)  # [B,T,4]
                contact = contact[:, 1:].unsqueeze(-1)                          # [B,T-1,4,1]
                l_fk_skate = (foot_v.abs() * contact).sum() / (contact.sum() * 3.0 + 1e-6)

            total = (
                l_recon
                + self.root_header_alpha * l_root_header
                + self.weights.vq * l_vq
                + self.fk_weight * l_fk
                + self.fk_vel_weight * l_fk_vel
                + self.fk_acc_weight * l_fk_acc
                + self.fk_footskate_weight * l_fk_skate
            )
            return UMRLossOutput(
                total=total,
                root_traj=l_root_header,
                root_rot6d=zero,
                joints76_rot6d=zero,
                sparse_vel=zero,
                foot_contact=zero,
                vq=l_vq,
                rot6d_ortho=zero,
                recon_smooth=l_recon,
                root_header=l_root_header,
                fk_joint=l_fk,
                fk_vel=l_fk_vel,
                fk_acc=l_fk_acc,
                fk_footskate=l_fk_skate,
            )

        if self.loss_mode == "smooth_recon":
            l_recon = F.smooth_l1_loss(features_rec, features_gt)
            rec_root_traj = features_rec[:, SLICE_ROOT_TRAJ]
            gt_root_traj = features_gt[:, SLICE_ROOT_TRAJ]
            rec_root_rot = features_rec[:, SLICE_ROOT_ROT6D]
            gt_root_rot = features_gt[:, SLICE_ROOT_ROT6D]
            l_root_traj_header = F.smooth_l1_loss(rec_root_traj, gt_root_traj)
            l_root_rot_header = F.smooth_l1_loss(rec_root_rot, gt_root_rot)
            # Channel-count-weighted mean of the two root-header terms; equals the
            # concatenated root-header loss when both weights are 1.0.
            l_root_header = (
                w.root_traj * 9.0 * l_root_traj_header
                + w.root_rot6d * 6.0 * l_root_rot_header
            ) / 15.0
            # Extra weight on the joints76_rot6d pose block; loss_vel=0 disables it.
            if self.loss_vel > 0:
                l_joints = F.smooth_l1_loss(
                    features_rec[:, SLICE_JOINTS76_ROT6D],
                    features_gt[:, SLICE_JOINTS76_ROT6D],
                )
            else:
                l_joints = features_rec.new_zeros(())
            zero = features_rec.new_zeros(())
            total = (
                l_recon
                + self.loss_vel * l_joints
                + self.root_header_alpha * l_root_header
                + w.vq * l_vq
            )
            return UMRLossOutput(
                total=total,
                root_traj=l_root_traj_header,
                root_rot6d=l_root_rot_header,
                joints76_rot6d=l_joints,
                sparse_vel=zero,
                foot_contact=zero,
                vq=l_vq,
                rot6d_ortho=zero,
                recon_smooth=l_recon,
                root_header=l_root_header,
            )

        l_root_traj = F.mse_loss(
            features_rec[:, SLICE_ROOT_TRAJ], features_gt[:, SLICE_ROOT_TRAJ]
        )
        l_root_rot6d = F.mse_loss(
            features_rec[:, SLICE_ROOT_ROT6D], features_gt[:, SLICE_ROOT_ROT6D]
        )
        l_joints76_rot6d = F.mse_loss(
            features_rec[:, SLICE_JOINTS76_ROT6D],
            features_gt[:, SLICE_JOINTS76_ROT6D],
        )
        l_sparse_vel = F.mse_loss(
            features_rec[:, SLICE_SPARSE_VEL], features_gt[:, SLICE_SPARSE_VEL]
        )

        # Contact: channels-first [B, 4, T] → channels-last [B, T, 4] for BCE helper.
        contact_logits = features_rec[:, SLICE_FOOT_CONTACT].transpose(1, 2)
        contact_gt = features_gt[:, SLICE_FOOT_CONTACT].transpose(1, 2)
        if contact_logits.shape[-1] != DIM_FOOT_CONTACT:
            raise RuntimeError(
                f"foot_contact slice mismatched: {contact_logits.shape[-1]}"
            )
        l_foot_contact = contact_bce_loss(
            contact_logits, contact_gt, pos_weight=self.contact_pos_weight
        )

        if w.rot6d_ortho > 0.0:
            l_rot6d_ortho = _rot6d_orthogonality_penalty(features_rec)
        else:
            l_rot6d_ortho = features_rec.new_zeros(())

        total = (
            w.root_traj * l_root_traj
            + w.root_rot6d * l_root_rot6d
            + w.joints76_rot6d * l_joints76_rot6d
            + w.sparse_vel * l_sparse_vel
            + w.foot_contact * l_foot_contact
            + w.vq * l_vq
            + w.rot6d_ortho * l_rot6d_ortho
        )
        return UMRLossOutput(
            total=total,
            root_traj=l_root_traj,
            root_rot6d=l_root_rot6d,
            joints76_rot6d=l_joints76_rot6d,
            sparse_vel=l_sparse_vel,
            foot_contact=l_foot_contact,
            vq=l_vq,
            rot6d_ortho=l_rot6d_ortho,
        )


def _rot6d_orthogonality_penalty(features_rec: Tensor) -> Tensor:
    """Penalty ``‖b1·b2‖²`` averaged over root + joints76 rot6d channels.

    Gram-Schmidt orthogonalizes the 6D pair post hoc; driving the penalty to 0
    keeps the regressor in a well-conditioned region.
    """
    root = features_rec[:, SLICE_ROOT_ROT6D].transpose(1, 2)                     # [B, T, 6]
    b1, b2 = root[..., :3], root[..., 3:]
    pen = (b1 * b2).sum(dim=-1).pow(2).mean()

    joints = features_rec[:, SLICE_JOINTS76_ROT6D]                                # [B, 456, T]
    B, _, T = joints.shape
    joints = joints.transpose(1, 2).reshape(B, T, UMR_NUM_JOINTS76, 6)
    b1, b2 = joints[..., :3], joints[..., 3:]
    pen = pen + (b1 * b2).sum(dim=-1).pow(2).mean()
    return pen


def tmr_distill_loss(h_sem: Tensor, e_teacher: Tensor) -> Tensor:
    """Cosine semantic-distillation loss ``1 - cos(h_sem, stopgrad(e_teacher))``.

    ``h_sem`` is the student embedding from
    :class:`~models.umr.structured_vq.SemanticHead`; ``e_teacher`` is the frozen
    TMR motion embedding. Accepts ``[B, dsem]`` (window-pool) or
    ``[B, T_token, dsem]`` (per-token); in the per-token case, tokens with an
    all-zero teacher row (failed FK / too-short segment) are masked out of the
    mean, while the head still projects every token so DDP sees the params as used.
    """
    h = F.normalize(h_sem, dim=-1)
    e_raw = e_teacher.detach()
    e = F.normalize(e_raw, dim=-1)
    cos = (h * e).sum(dim=-1)              # [B] or [B, T]
    if cos.dim() >= 2:
        # per-token: mask all-zero teacher rows (failed FK / too-short segment).
        valid = (e_raw.abs().sum(dim=-1) > 0).to(cos.dtype)   # [B, T]
        denom = valid.sum().clamp(min=1.0)
        return ((1.0 - cos) * valid).sum() / denom
    return (1.0 - cos).mean()


__all__ = [
    "UMRLoss",
    "UMRLossOutput",
    "UMRLossWeights",
    "tmr_distill_loss",
]
