"""Foot-contact classification losses."""

from __future__ import annotations

import math

import pytest
import torch

from losses.contact_losses import contact_bce_loss, contact_focal_loss

def test_contact_bce_zero_for_perfect_logits():
    """BCE → 0 as logits → ±∞ matching target."""
    pred = torch.tensor([[[20.0, -20.0, 20.0, -20.0]]])
    gt = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    loss = contact_bce_loss(pred, gt)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

def test_contact_bce_returns_log_two_for_zero_logits():
    """BCE on logits=0 with target=0/1 → ln(2) per element (uniform)."""
    pred = torch.zeros(1, 1, 4)
    gt = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    loss = contact_bce_loss(pred, gt)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)

def test_contact_bce_pos_weight_scalar():
    """Scalar pos_weight should not crash and should match per-channel tensor."""
    pred = torch.zeros(1, 1, 4)
    gt = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    a = contact_bce_loss(pred, gt, pos_weight=2.0)
    b = contact_bce_loss(pred, gt, pos_weight=torch.tensor([2.0, 2.0, 2.0, 2.0]))
    assert a.item() == pytest.approx(b.item(), rel=1e-6)

def test_contact_bce_gradient_flows_through_logits():
    pred = torch.zeros(1, 1, 4, requires_grad=True)
    gt = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    loss = contact_bce_loss(pred, gt)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0

def test_contact_focal_reduces_weight_on_easy_examples():
    """Focal at γ>0 should be smaller than plain BCE on confident-correct logits."""
    pred = torch.full((1, 1, 4), 3.0)
    gt = torch.ones(1, 1, 4)
    bce = contact_bce_loss(pred, gt).item()
    focal = contact_focal_loss(pred, gt, alpha=0.5, gamma=2.0).item()
    assert focal < bce

def test_contact_focal_reduces_to_bce_when_gamma_zero():
    pred = torch.zeros(1, 1, 4)
    gt = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])
    focal = contact_focal_loss(pred, gt, alpha=0.5, gamma=0.0).item()
    bce = contact_bce_loss(pred, gt).item()
    assert focal == pytest.approx(0.5 * bce, rel=1e-6)
