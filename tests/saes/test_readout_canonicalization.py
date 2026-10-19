import copy

import pytest
import torch

from sae_lens.saes.readout_canonicalization import (
    fold_activation_transform,
    gain_free_reconstruction_error,
    readout_canonicalizing_transform,
)
from sae_lens.saes.standard_sae import StandardSAE
from sae_lens.saes.topk_sae import TopKSAE
from tests.helpers import assert_close, build_sae_cfg, build_topk_sae_cfg, random_params


def test_readout_canonicalizing_transform_removes_all_singular_value_gains():
    readout = torch.randn(128, 32, dtype=torch.float64)
    transform = readout_canonicalizing_transform(readout)
    assert_close(
        torch.linalg.svdvals(readout @ transform),
        torch.ones(32, dtype=torch.float64),
    )
    assert torch.linalg.matrix_rank(transform) == 32


def test_readout_canonicalizing_transform_leaves_invisible_directions_unchanged():
    d_in = 8
    d_readout = 5
    left, _ = torch.linalg.qr(torch.randn(d_readout, d_readout, dtype=torch.float64))
    right, _ = torch.linalg.qr(torch.randn(d_in, d_in, dtype=torch.float64))
    gains = torch.tensor([10.0, 5.0, 2.0, 0.0, 0.0], dtype=torch.float64)
    readout = left @ torch.diag(gains) @ right[:, :d_readout].T
    transform = readout_canonicalizing_transform(readout)
    assert_close(
        torch.linalg.svdvals(readout @ transform),
        torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float64),
    )
    # every direction the readout cannot see passes through the transform
    # unchanged, so no information is discarded.
    for invisible_direction in right[:, 3:].T:
        assert_close(invisible_direction @ transform, invisible_direction)
    assert torch.linalg.matrix_rank(transform) == d_in


def test_fold_activation_transform_reproduces_wrapped_topk_sae_exactly():
    cfg = build_topk_sae_cfg(d_in=32, d_sae=96, k=16, dtype="float64")
    sae = TopKSAE(cfg)
    random_params(sae)
    readout = torch.randn(64, 32, dtype=torch.float64)
    transform = readout_canonicalizing_transform(readout)
    folded_sae = copy.deepcopy(sae)
    fold_activation_transform(folded_sae, transform)
    x = torch.randn(256, 32, dtype=torch.float64)
    inverse = torch.linalg.inv(transform)
    assert_close(folded_sae(x), sae(x @ transform) @ inverse)


def test_fold_activation_transform_matches_fold_activation_norm_scaling_factor():
    cfg = build_sae_cfg(d_in=16, d_sae=32, dtype="float64")
    scalar_folded = StandardSAE(cfg)
    random_params(scalar_folded)
    matrix_folded = StandardSAE(cfg)
    matrix_folded.load_state_dict(scalar_folded.state_dict())
    scaling_factor = 2.5
    scalar_folded.fold_activation_norm_scaling_factor(scaling_factor)
    identity = scaling_factor * torch.eye(cfg.d_in, dtype=torch.float64)
    fold_activation_transform(matrix_folded, identity)
    assert_close(scalar_folded.W_enc, matrix_folded.W_enc)
    assert_close(scalar_folded.W_dec, matrix_folded.W_dec)
    assert_close(scalar_folded.b_dec, matrix_folded.b_dec)


def test_gain_free_reconstruction_error_ignores_readout_gains():
    d_in = 8
    d_readout = 5
    gains = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5], dtype=torch.float64)
    left, _ = torch.linalg.qr(torch.randn(d_readout, d_readout, dtype=torch.float64))
    right, _ = torch.linalg.qr(torch.randn(d_in, d_in, dtype=torch.float64))
    readout = left @ torch.diag(gains) @ right[:, :d_readout].T
    magnitude = 0.7
    sae_in = torch.zeros(2, d_in, dtype=torch.float64)
    high_gain_out = magnitude * right[:, 0].unsqueeze(0).expand(2, d_in)
    low_gain_out = magnitude * right[:, 4].unsqueeze(0).expand(2, d_in)
    high_gain_free = gain_free_reconstruction_error(sae_in, high_gain_out, readout)
    low_gain_free = gain_free_reconstruction_error(sae_in, low_gain_out, readout)
    assert high_gain_free == pytest.approx(magnitude**2)
    assert low_gain_free == pytest.approx(magnitude**2)
    # the readout itself weights the same errors by the square of its gains.
    readout_space_high = ((high_gain_out @ readout.T) ** 2).sum(-1).mean()
    readout_space_low = ((low_gain_out @ readout.T) ** 2).sum(-1).mean()
    assert readout_space_high == pytest.approx((gains[0] * magnitude) ** 2)
    assert readout_space_low == pytest.approx((gains[4] * magnitude) ** 2)


def test_gain_free_reconstruction_error_ignores_invisible_directions():
    d_in = 8
    d_readout = 5
    left, _ = torch.linalg.qr(torch.randn(d_readout, d_readout, dtype=torch.float64))
    right, _ = torch.linalg.qr(torch.randn(d_in, d_in, dtype=torch.float64))
    gains = torch.tensor([10.0, 5.0, 2.0, 0.0, 0.0], dtype=torch.float64)
    readout = left @ torch.diag(gains) @ right[:, :d_readout].T
    magnitude = 0.9
    sae_in = torch.zeros(4, d_in, dtype=torch.float64)
    sae_out = magnitude * right[:, 5].unsqueeze(0).expand(4, d_in)
    gain_free = gain_free_reconstruction_error(sae_in, sae_out, readout)
    assert gain_free == pytest.approx(0.0, abs=1e-12)
    plain_mse = ((sae_out - sae_in) ** 2).sum(-1).mean()
    assert plain_mse == pytest.approx(magnitude**2)


def test_gain_free_reconstruction_error_equals_plain_mse_for_orthonormal_readout():
    d_in = 6
    orthonormal_readout, _ = torch.linalg.qr(torch.randn(d_in, d_in))
    sae_in = torch.randn(128, d_in)
    sae_out = sae_in + 0.1 * torch.randn(128, d_in)
    expected = ((sae_out - sae_in) ** 2).sum(-1).mean().item()
    gain_free = gain_free_reconstruction_error(sae_in, sae_out, orthonormal_readout)
    assert gain_free == pytest.approx(expected)
