import pytest
import torch
from transformer_lens import HookedTransformerConfig

from benchmark.feature_probe_evals import (
    auroc,
    causal_steering_gain,
    contrastive_feature_scores,
    feature_activations_from_cache,
    feature_ranks,
    logit_diff,
    make_steering_hook,
    random_control_directions,
)
from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from sae_lens.saes.standard_sae import StandardSAE
from tests.helpers import assert_close, build_sae_cfg, random_params


def build_identity_encoder_sae() -> StandardSAE:
    sae = StandardSAE(build_sae_cfg())
    random_params(sae)
    with torch.no_grad():
        # feature j < d_in encodes relu(x_j), everything else stays silent
        sae.W_enc.zero_()
        sae.W_enc[:, : sae.cfg.d_in] = torch.eye(sae.cfg.d_in)
        sae.b_enc.zero_()
    return sae


def test_auroc_counts_ties_as_half_wins():
    positive = torch.tensor([3.0, 4.0, 5.0])
    control = torch.tensor([0.0, 1.0, 2.0])
    assert auroc(positive, control) == pytest.approx(1.0)
    assert auroc(control, positive) == pytest.approx(0.0)
    assert auroc(positive, positive) == pytest.approx(0.5)
    # pairs: (1,1) tie, (1,3) loss, (2,1) win, (2,3) loss -> 1.5 / 4
    mixed_auroc = auroc(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))
    assert mixed_auroc == pytest.approx(0.375)


def test_discovery_ranks_planted_feature_first_through_sae():
    sae = build_identity_encoder_sae()
    planted, runner_up = 3, 7
    positive_acts = torch.randn(5, 6, sae.cfg.d_in)
    control_acts = torch.randn(5, 6, sae.cfg.d_in)
    positive_acts[:, :, planted] = 8.0
    positive_acts[:, :, runner_up] = 4.0
    tokens = torch.zeros(5, 6, dtype=torch.long)

    positive_feats = feature_activations_from_cache(sae, positive_acts, tokens)
    control_feats = feature_activations_from_cache(sae, control_acts, tokens)

    scores = contrastive_feature_scores(positive_feats, control_feats)
    ranks = feature_ranks(scores)

    assert ranks[planted].item() == 1
    assert ranks[runner_up].item() == 2
    planted_auroc = auroc(positive_feats[:, planted], control_feats[:, planted])
    assert planted_auroc == pytest.approx(1.0)


def test_feature_ranks_orders_all_features():
    scores = torch.tensor([0.1, 5.0, -1.0, 2.0])
    assert feature_ranks(scores).tolist() == [3, 1, 4, 2]


def test_feature_activations_exclude_special_token_positions():
    sae = build_identity_encoder_sae()
    cache_acts = torch.zeros(1, 4, sae.cfg.d_in)
    cache_acts[0, 0, 2] = 100.0
    cache_acts[0, 2, 2] = 3.0
    cache_acts[0, 2, 5] = 1.5
    tokens = torch.tensor([[0, 5, 5, 5]])

    feats = feature_activations_from_cache(sae, cache_acts, tokens, [0])

    assert feats[0, 2].item() == pytest.approx(3.0)
    assert feats[0, 5].item() == pytest.approx(1.5)


def test_all_special_token_probe_falls_back_to_all_positions():
    sae = build_identity_encoder_sae()
    cache_acts = torch.zeros(1, 3, sae.cfg.d_in)
    cache_acts[0, 1, 1] = 3.0
    tokens = torch.tensor([[9, 9, 9]])

    feats = feature_activations_from_cache(sae, cache_acts, tokens, [9])

    assert feats[0, 1].item() == pytest.approx(3.0)


def test_steering_hook_adds_scaled_direction_to_all_positions():
    activations = torch.zeros(2, 3, 4)
    direction = torch.tensor([1.0, -1.0, 2.0, 0.5], dtype=torch.float64)

    steered = make_steering_hook(direction, 5.0)(activations, None)

    assert steered.dtype == torch.float32
    assert_close(steered, 5.0 * direction.float().expand(2, 3, 4))
    assert torch.equal(
        make_steering_hook(direction, 0.0)(activations, None), activations
    )


def test_logit_diff_uses_final_position_and_averages_batch():
    logits = torch.tensor([[[1.0, 9.0, 3.0], [4.0, 2.0, 8.0]]])
    assert logit_diff(logits, 2, 0) == pytest.approx(4.0)

    batched = torch.tensor([[[1.0, 2.0]], [[5.0, 3.0]]])
    assert logit_diff(batched, 0, 1) == pytest.approx(0.5)


def test_random_control_directions_match_norm_and_miss_reference():
    reference = torch.randn(64)
    reference = 3.0 * reference / reference.norm()

    controls = random_control_directions(reference, 64)

    assert controls.shape == (64, 64)
    assert_close(controls.norm(dim=-1), reference.norm().expand(64))
    cosines = controls @ reference / (controls.norm(dim=-1) * reference.norm())
    assert cosines.abs().max().item() < 0.6


def test_causal_steering_gain_runs_the_feature_direction_through_the_model():
    model = HookedSAETransformer(
        HookedTransformerConfig(
            d_model=64, d_head=32, n_heads=2, n_layers=1, n_ctx=16, d_vocab=64
        )
    )
    sae = build_identity_encoder_sae()
    sae.cfg.metadata.hook_name = "blocks.0.hook_resid_post"

    metrics = causal_steering_gain(
        sae,
        model,
        tokens=torch.zeros(1, 4, dtype=torch.long),
        feature_index=0,
        target_token_id=1,
        baseline_token_id=2,
        steering_strength=5.0,
        n_random_controls=4,
    )

    assert set(metrics) == {
        "baseline_logit_diff",
        "steered_logit_diff",
        "steering_gain",
        "random_control_gain",
        "steering_gain_over_control",
    }
    assert metrics["steered_logit_diff"] != pytest.approx(
        metrics["baseline_logit_diff"]
    )
    assert metrics["steering_gain"] == pytest.approx(
        metrics["steered_logit_diff"] - metrics["baseline_logit_diff"]
    )
    assert metrics["steering_gain_over_control"] == pytest.approx(
        metrics["steering_gain"] - metrics["random_control_gain"]
    )
