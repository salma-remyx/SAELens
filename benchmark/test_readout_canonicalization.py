import copy
import math

import pytest
import torch

from sae_lens.saes.readout_canonicalization import (
    fold_activation_transform,
    gain_free_reconstruction_error,
    readout_canonicalizing_transform,
)
from sae_lens.saes.topk_sae import TopKSAE
from tests.helpers import assert_close, build_topk_sae_cfg, random_params


def _readout_with_condition_number(
    d_readout: int, d_in: int, condition_number: float
) -> torch.Tensor:
    """Random readout whose singular values logspace from 1 to condition_number."""
    gains = torch.logspace(0, math.log10(condition_number), d_in, dtype=torch.float64)
    left, _ = torch.linalg.qr(torch.randn(d_readout, d_readout, dtype=torch.float64))
    right, _ = torch.linalg.qr(torch.randn(d_in, d_in, dtype=torch.float64))
    return left[:, :d_in] @ torch.diag(gains) @ right.T


def test_rcds_transform_removes_gains_at_condition_number_100():
    readout = _readout_with_condition_number(128, 32, 100.0)
    assert torch.linalg.cond(readout) == pytest.approx(100.0, rel=1e-9)
    transform = readout_canonicalizing_transform(readout)
    canonical_readout = readout @ transform
    assert torch.linalg.cond(canonical_readout) == pytest.approx(1.0, rel=1e-8)
    # the canonicalized readout retains exactly the decision information of
    # the original: same rank, invertible transform.
    assert torch.linalg.matrix_rank(canonical_readout) == torch.linalg.matrix_rank(
        readout
    )
    assert torch.linalg.matrix_rank(transform) == 32


def test_rcds_wrapper_on_topk_sae_reproduces_to_numerical_precision():
    cfg = build_topk_sae_cfg(d_in=32, d_sae=96, k=16, dtype="float64")
    sae = TopKSAE(cfg)
    random_params(sae)
    readout = _readout_with_condition_number(128, 32, 100.0)
    transform = readout_canonicalizing_transform(readout)
    wrapped_sae = copy.deepcopy(sae)
    fold_activation_transform(wrapped_sae, transform)
    x = torch.randn(128, 32, dtype=torch.float64)
    inverse = torch.linalg.inv(transform)
    assert_close(wrapped_sae(x), sae(x @ transform) @ inverse)


def test_gain_free_metric_separates_gain_exploiting_reconstruction_errors():
    readout = _readout_with_condition_number(16, 8, 100.0)
    _, _, vh = torch.linalg.svd(readout)
    magnitude = 0.5
    sae_in = torch.zeros(2, 8, dtype=torch.float64)
    high_gain_out = magnitude * vh[0].unsqueeze(0).expand(2, 8)
    low_gain_out = magnitude * vh[-1].unsqueeze(0).expand(2, 8)
    # equal-magnitude errors along the highest- and lowest-gain decision
    # directions are equal in gain-free coordinates...
    high_gain_free = gain_free_reconstruction_error(sae_in, high_gain_out, readout)
    low_gain_free = gain_free_reconstruction_error(sae_in, low_gain_out, readout)
    assert high_gain_free == pytest.approx(magnitude**2)
    assert low_gain_free == pytest.approx(magnitude**2)
    # ...but the readout's own gains separate them by condition_number**2.
    readout_space_high = ((high_gain_out @ readout.T) ** 2).sum(-1).mean()
    readout_space_low = ((low_gain_out @ readout.T) ** 2).sum(-1).mean()
    assert readout_space_high / readout_space_low == pytest.approx(100.0**2)
