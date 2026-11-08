import math

import pytest
import torch

from sae_lens.saes.jumprelu_sae import JumpReLUSAE
from sae_lens.saes.topk_selection import (
    compare_threshold_and_topk,
    hidden_pre_from_sae,
    l0_per_sample,
    matched_budget_k,
    select_topk_acts,
    selection_budget,
)
from tests.helpers import build_jumprelu_sae_cfg, random_params


def test_select_topk_acts_keeps_exactly_k_highest_per_sample():
    hidden_pre = torch.tensor(
        [
            [0.1, 2.0, -1.0, 3.0, 0.5],
            [-4.0, -0.5, -2.0, -1.0, -3.0],
        ]
    )

    acts = select_topk_acts(hidden_pre, k=2)

    expected = torch.tensor(
        [
            [0.0, 2.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(acts, expected)
    assert l0_per_sample(acts).tolist() == [2, 0]


def test_select_topk_acts_keeps_each_samples_own_highest_pre_activations():
    hidden_pre = torch.randn(64, 128) * 3

    acts = select_topk_acts(hidden_pre, k=7)

    top7 = hidden_pre.relu().topk(7, dim=-1)
    # the kept values are the pre-activations themselves, not rescaled
    torch.testing.assert_close(acts.gather(-1, top7.indices), top7.values)
    torch.testing.assert_close(acts.sum(dim=-1), top7.values.sum(dim=-1))
    assert (acts >= 0).all()


def test_selection_budget_reports_per_sample_counts():
    acts = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0, 5.0],
        ]
    )

    budget = selection_budget(l0_per_sample(acts), d_sae=5, nominal_l0=2.0)

    assert budget.mean_l0 == pytest.approx(1.75)
    assert budget.std_l0 == pytest.approx(1.5)
    assert budget.min_l0 == 0
    assert budget.max_l0 == 3
    assert budget.read_fraction == pytest.approx(0.35)
    assert budget.budget_ratio == pytest.approx(0.875)

    assert selection_budget(l0_per_sample(acts), d_sae=5).budget_ratio is None


def test_matched_budget_k_rounds_ties_up_and_clamps_to_a_valid_k():
    assert matched_budget_k(torch.tensor([7, 7, 7, 7]), d_sae=100) == 7
    # mean 7.5 rounds up, not to even
    assert matched_budget_k(torch.tensor([8, 8, 7, 7]), d_sae=100) == 8
    assert matched_budget_k(torch.tensor([0, 0, 0, 0]), d_sae=100) == 1
    assert matched_budget_k(torch.tensor([300, 300]), d_sae=256) == 256


def test_topk_selection_keeps_every_feature_the_threshold_kept():
    sae = JumpReLUSAE(build_jumprelu_sae_cfg(apply_b_dec_to_input=False))
    random_params(sae)
    sae.threshold.data = torch.ones_like(sae.threshold)
    x = torch.randn(500, sae.cfg.d_in)

    threshold_acts = sae.encode(x)
    threshold_counts = l0_per_sample(threshold_acts)

    # a threshold leaves the per-sample count free to vary
    assert threshold_counts.min() < threshold_counts.max()

    # given k large enough to cover the largest count, top-k keeps every feature
    # the threshold kept, so it dominates it featurewise
    topk_acts = select_topk_acts(
        hidden_pre_from_sae(sae, x), k=int(threshold_counts.max().item())
    )
    assert (topk_acts >= threshold_acts).all()


def test_compare_threshold_and_topk_agrees_with_the_threshold_at_a_constant_budget():
    cfg = build_jumprelu_sae_cfg(d_in=4, d_sae=8, apply_b_dec_to_input=False)
    sae = JumpReLUSAE(cfg)
    random_params(sae)
    # route the input straight through to the first four latents and pin the
    # other four below threshold, so hidden_pre is the input padded with -1
    sae.W_enc.data = torch.zeros_like(sae.W_enc)
    sae.W_enc.data[: cfg.d_in, : cfg.d_in] = torch.eye(cfg.d_in)
    sae.b_enc.data[: cfg.d_in] = 0.0
    sae.b_enc.data[cfg.d_in :] = -1.0
    sae.threshold.data = torch.zeros_like(sae.threshold)
    x = torch.tensor(
        [
            [1.0, -1.0, 2.0, -3.0],
            [0.5, -2.0, -1.0, 4.0],
        ]
    )

    comparison = compare_threshold_and_topk(sae, x)

    # every sample has exactly two positive entries, so both policies pick the
    # same two features and must reconstruct identically
    assert comparison.k == 2
    assert comparison.threshold_budget.mean_l0 == pytest.approx(2.0)
    assert comparison.threshold_budget.std_l0 == pytest.approx(0.0, abs=1e-12)
    assert comparison.topk_budget.mean_l0 == pytest.approx(2.0)
    assert comparison.topk_mse.item() == pytest.approx(
        comparison.threshold_mse.item(), rel=1e-6
    )
    assert comparison.topk_explained_variance.item() == pytest.approx(
        comparison.threshold_explained_variance.item(), rel=1e-6
    )


def test_compare_threshold_and_topk_keeps_exactly_k_when_k_features_are_available():
    sae = JumpReLUSAE(build_jumprelu_sae_cfg(apply_b_dec_to_input=False))
    random_params(sae)
    # signed encoder weights keep the per-sample pre-activation mean near zero,
    # so every sample has far more than k positive features to choose from
    sae.W_enc.data = torch.randn_like(sae.W_enc)
    sae.threshold.data = torch.zeros_like(sae.threshold)
    x = torch.randn(500, sae.cfg.d_in)

    comparison = compare_threshold_and_topk(sae, x, k=8)

    assert comparison.k == 8
    assert comparison.topk_budget.min_l0 == 8
    assert comparison.topk_budget.max_l0 == 8
    assert comparison.topk_budget.mean_l0 == pytest.approx(8.0)
    assert comparison.topk_budget.std_l0 == pytest.approx(0.0, abs=1e-12)
    assert comparison.topk_budget.read_fraction == pytest.approx(8 / sae.cfg.d_sae)


def test_compare_threshold_and_topk_caps_the_budget_the_threshold_overspends():
    sae = JumpReLUSAE(build_jumprelu_sae_cfg(apply_b_dec_to_input=False))
    random_params(sae)
    sae.threshold.data = torch.full_like(sae.threshold, 1.0)
    x = torch.randn(500, sae.cfg.d_in)

    comparison = compare_threshold_and_topk(sae, x, nominal_l0=10.0)
    threshold_l0 = l0_per_sample(sae.encode(x))

    assert comparison.k == matched_budget_k(threshold_l0, sae.cfg.d_sae)
    # top-k never spends more than k features on a sample ...
    assert comparison.topk_budget.max_l0 <= comparison.k
    assert comparison.topk_budget.mean_l0 <= comparison.k
    # ... while the threshold policy has no such cap and varies per sample
    assert comparison.threshold_budget.max_l0 > comparison.k
    assert comparison.threshold_budget.std_l0 > 0.0
    assert comparison.threshold_budget.min_l0 < comparison.threshold_budget.max_l0
    assert comparison.topk_budget.budget_ratio == pytest.approx(
        comparison.topk_budget.mean_l0 / 10.0
    )

    # the reported mse is the reconstruction error of the top-k activations
    topk_acts = select_topk_acts(hidden_pre_from_sae(sae, x), comparison.k)
    expected_mse = (sae.decode(topk_acts).float() - x.float()).pow(2).mean()
    assert comparison.topk_mse.item() == pytest.approx(expected_mse.item(), rel=1e-6)
    assert math.isfinite(comparison.threshold_mse.item())
    assert math.isfinite(comparison.topk_explained_variance.item())
