import torch
from pytest import approx

from benchmark.selection_budget import (
    calibrate_threshold,
    compare_selection_policies,
    format_reports,
    sae_hidden_pre,
    select_threshold,
    select_topk,
)
from sae_lens.saes.jumprelu_sae import JumpReLUSAE
from sae_lens.saes.topk_sae import TopKSAE
from tests.helpers import (
    assert_close,
    build_jumprelu_sae_cfg,
    build_topk_sae_cfg,
    random_params,
)

D_IN = 64
D_SAE = 256
K = 16
# inputs are non-negative with a per-token scale, so tokens differ in how many
# features a selector scores highly: the regime where a global threshold cannot
# spend a fixed budget per token
SCALE_RANGE = (0.25, 4.0)


def _make_topk_sae() -> TopKSAE:
    sae = TopKSAE(
        build_topk_sae_cfg(d_in=D_IN, d_sae=D_SAE, k=K, apply_b_dec_to_input=False)
    )
    random_params(sae)
    return sae


def _activations(num_tokens: int, min_scale: float, max_scale: float) -> torch.Tensor:
    scales = torch.empty(num_tokens, 1).uniform_(min_scale, max_scale)
    return torch.rand(num_tokens, D_IN) * scales


def test_calibrate_threshold_keeps_target_budget_on_its_calibration_scores():
    sae = _make_topk_sae()
    scores = sae_hidden_pre(sae, _activations(1024, *SCALE_RANGE))

    threshold = calibrate_threshold(scores, k=K)

    num_tokens = scores.shape[0]
    # a float32 score that bit-matches the boundary is dropped by the strict
    # comparison, so allow a couple of tied scores out of K * num_tokens kept
    assert (scores > threshold).sum().item() == approx(K * num_tokens, abs=4)
    assert threshold > 0


def test_topk_spends_budget_exactly_while_threshold_spends_it_unevenly():
    sae = _make_topk_sae()
    x = _activations(1024, *SCALE_RANGE)
    reports = compare_selection_policies(sae, x, k=K)
    print("\n" + format_reports(reports))

    topk = reports["topk"]
    assert topk.realized_active == approx(K)
    assert topk.active_p10 == approx(K)
    assert topk.active_p90 == approx(K)
    assert topk.intended_budget == approx(K / D_SAE)
    assert topk.budget_ratio == approx(1.0)

    threshold = reports["threshold"]
    assert threshold.threshold is not None
    assert threshold.realized_active == approx(K, abs=0.01)
    assert threshold.active_p10 < K / 2
    assert threshold.active_p90 > 2 * K

    assert topk.reconstruction_mse < threshold.reconstruction_mse


def test_threshold_calibrated_off_batch_undershoots_its_budget():
    sae = _make_topk_sae()
    calibration_scores = sae_hidden_pre(
        sae, _activations(1024, min_scale=1.0, max_scale=1.1)
    )
    x = _activations(1024, min_scale=0.8, max_scale=1.0)

    reports = compare_selection_policies(
        sae, x, k=K, calibration_scores=calibration_scores
    )

    threshold = reports["threshold"]
    assert threshold.intended_active == approx(K)
    assert threshold.realized_active < K / 2
    assert threshold.budget_ratio < 0.5
    # top-k reads the same scores and needs no calibration batch to hit k
    assert reports["topk"].realized_active == approx(K)
    assert reports["topk"].budget_ratio == approx(1.0)


def test_selection_policies_reproduce_the_sae_encode_paths():
    sae = _make_topk_sae()
    threshold_sae = JumpReLUSAE(
        build_jumprelu_sae_cfg(d_in=D_IN, d_sae=D_SAE, apply_b_dec_to_input=False)
    )
    with torch.no_grad():
        threshold_sae.W_enc.copy_(sae.W_enc)
        threshold_sae.b_enc.copy_(sae.b_enc)
        threshold_sae.W_dec.copy_(sae.W_dec)
        threshold_sae.b_dec.copy_(sae.b_dec)

    x = _activations(256, min_scale=0.5, max_scale=2.0)
    scores = sae_hidden_pre(sae, x)
    threshold = calibrate_threshold(scores, k=K)
    with torch.no_grad():
        threshold_sae.threshold.fill_(threshold)

    assert_close(select_topk(scores, K), sae.encode(x))
    assert_close(select_threshold(scores, threshold), threshold_sae.encode(x))


def test_sparse_topk_decode_matches_dense_decode():
    sae = _make_topk_sae()
    x = _activations(512, *SCALE_RANGE)

    reports = compare_selection_policies(sae, x, k=K)

    dense, sparse = reports["topk"], reports["topk_sparse"]
    assert sparse.realized_active == approx(dense.realized_active)
    assert sparse.active_p10 == approx(dense.active_p10)
    assert sparse.active_p90 == approx(dense.active_p90)
    assert sparse.reconstruction_mse == approx(dense.reconstruction_mse, rel=1e-5)
