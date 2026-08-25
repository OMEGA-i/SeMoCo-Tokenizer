"""Regression tests for :class:`models.quantizers.ema_rvq.QuantizeEMAReset`."""

from __future__ import annotations

import torch

from models.quantizers.ema_rvq import EMAResidualVQ, QuantizeEMAReset


def _train_step(vq: QuantizeEMAReset, x: torch.Tensor) -> None:
    vq.train()
    vq(x)


def test_first_training_batch_initializes_codebook() -> None:
    torch.manual_seed(0)
    vq = QuantizeEMAReset(nb_code=16, code_dim=8)
    assert not vq.init
    x = torch.randn(4, 8, 32)  # [N, C, T]
    _train_step(vq, x)
    assert vq.init
    assert vq.code_sum is not None and vq.code_count is not None


def test_load_state_dict_preserves_codebook_on_resume() -> None:
    """A resumed run must not re-initialise the trained codebook from the
    first incoming batch, and EMA updates must keep working afterwards."""
    torch.manual_seed(0)
    vq = QuantizeEMAReset(nb_code=16, code_dim=8)
    for _ in range(3):
        _train_step(vq, torch.randn(4, 8, 32))
    trained_codebook = vq.codebook.clone()

    fresh = QuantizeEMAReset(nb_code=16, code_dim=8)
    fresh.load_state_dict(vq.state_dict())
    assert fresh.init  # EMA state rebuilt from the loaded codebook
    assert fresh.code_sum is not None and fresh.code_count is not None
    assert torch.allclose(fresh.codebook, trained_codebook)

    # Next training forward must neither reset the codebook nor crash on
    # missing code_sum / code_count.
    _train_step(fresh, torch.randn(4, 8, 32))
    assert torch.allclose(fresh.code_sum.sum(), fresh.code_sum.sum())  # no NaN
    assert not torch.allclose(fresh.codebook, torch.zeros_like(fresh.codebook))


def test_ema_residual_vq_roundtrip() -> None:
    torch.manual_seed(0)
    rvq = EMAResidualVQ(dim=8, num_quantizers=2, codebook_size=16)
    rvq.train()
    x = torch.randn(4, 32, 8)  # [B, T, C]
    rvq(x)

    fresh = EMAResidualVQ(dim=8, num_quantizers=2, codebook_size=16)
    fresh.load_state_dict(rvq.state_dict())
    rvq.eval()
    fresh.eval()
    with torch.no_grad():
        z_q_a, *_ = rvq(x)
        z_q_b, *_ = fresh(x)
    assert torch.allclose(z_q_a, z_q_b)
