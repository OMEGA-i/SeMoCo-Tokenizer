"""UMRLoss component tests."""

from __future__ import annotations

import torch

from data.umr_schema import (
    DIM_FEATURES,
    DIM_FOOT_CONTACT,
    SLICE_FOOT_CONTACT,
    SLICE_JOINTS76_ROT6D,
    SLICE_ROOT_ROT6D,
    SLICE_ROOT_TRAJ,
    SLICE_SPARSE_VEL,
)
from losses.umr_loss import UMRLoss, UMRLossWeights

def _random_features(B: int = 2, T: int = 32, *, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    feat = torch.randn(B, DIM_FEATURES, T) * 0.05
    feat[:, SLICE_FOOT_CONTACT] = torch.randn(B, DIM_FOOT_CONTACT, T)
    return feat

def test_loss_zero_on_identity() -> None:
    feat = _random_features()
    gt = feat.clone()
    gt[:, SLICE_FOOT_CONTACT] = (torch.rand(*feat[:, SLICE_FOOT_CONTACT].shape) > 0.5).float()
    feat[:, SLICE_FOOT_CONTACT] = (gt[:, SLICE_FOOT_CONTACT] - 0.5) * 1e6

    loss_fn = UMRLoss()
    out = loss_fn(feat, gt)
    assert out.root_traj.item() == 0.0
    assert out.root_rot6d.item() == 0.0
    assert out.joints76_rot6d.item() == 0.0
    assert out.sparse_vel.item() == 0.0
    assert out.foot_contact.item() < 1e-3

def test_loss_components_match_field_l2() -> None:
    pred = _random_features(seed=1)
    gt = _random_features(seed=2)
    gt[:, SLICE_FOOT_CONTACT] = (torch.rand(*gt[:, SLICE_FOOT_CONTACT].shape) > 0.5).float()

    loss_fn = UMRLoss()
    out = loss_fn(pred, gt)

    expected_root_traj = torch.nn.functional.mse_loss(
        pred[:, SLICE_ROOT_TRAJ], gt[:, SLICE_ROOT_TRAJ]
    )
    torch.testing.assert_close(out.root_traj, expected_root_traj)

def test_total_includes_vq_loss() -> None:
    pred = _random_features(seed=3)
    gt = _random_features(seed=4)
    gt[:, SLICE_FOOT_CONTACT] = (torch.rand(*gt[:, SLICE_FOOT_CONTACT].shape) > 0.5).float()

    loss_fn = UMRLoss(weights=UMRLossWeights(vq=2.0))
    out_no_vq = loss_fn(pred, gt, vq_loss=0.0)
    out_with_vq = loss_fn(pred, gt, vq_loss=torch.tensor(0.5))
    assert (out_with_vq.total - out_no_vq.total).item() > 0.999

def test_rot6d_ortho_penalty() -> None:
    pred = _random_features(seed=5)
    gt = _random_features(seed=6)
    gt[:, SLICE_FOOT_CONTACT] = (torch.rand(*gt[:, SLICE_FOOT_CONTACT].shape) > 0.5).float()

    loss_fn = UMRLoss(weights=UMRLossWeights(rot6d_ortho=1.0))
    out = loss_fn(pred, gt)
    assert out.rot6d_ortho.item() > 0.0
