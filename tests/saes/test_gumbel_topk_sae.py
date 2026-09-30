import pytest
import torch

from sae_lens.saes.gumbel_topk_sae import (
    GumbelTopKSelection,
    GumbelTopKTrainingSAE,
    GumbelTopKTrainingSAEConfig,
    gumbel_topk_selection,
    topk_indicator,
)
from sae_lens.saes.sae import TrainStepInput
from tests.helpers import assert_close, random_params


def build_gumbel_topk_sae_training_cfg(**kwargs):
    defaults = {
        "d_in": 8,
        "d_sae": 16,
        "k": 4,
        "dtype": "float32",
        "device": "cpu",
        "rescale_acts_by_decoder_norm": False,
        "apply_b_dec_to_input": False,
        "selection_hidden_dim": 6,
        "selection_tau": 1.0,
    }
    return GumbelTopKTrainingSAEConfig(**{**defaults, **kwargs})


def test_topk_indicator_selects_exactly_the_k_largest_scores():
    scores = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0, 5.0],
            [-5.0, 6.0, -7.0, 8.0, -9.0],
        ]
    )
    # k=2 keeps the two largest scores per row, regardless of sign.
    expected = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
        ]
    )

    mask = topk_indicator(scores, k=2)

    assert_close(mask, expected)
    assert (mask.sum(dim=-1) == 2).all()


def test_GumbelTopKSelection_eval_mode_is_deterministic_topk():
    selection = GumbelTopKSelection(k=3).eval()
    scores = torch.tensor(
        [
            [9.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 9.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0, 9.0],
        ]
    )

    first_mask = selection(scores)
    second_mask = selection(scores)

    assert torch.equal(first_mask, second_mask)
    assert torch.equal(first_mask, topk_indicator(scores, k=3))
    # Rows with different score rankings must select different feature sets:
    # the mask is conditioned on the sample, not shared across the batch.
    assert not torch.equal(first_mask[0], first_mask[1])
    assert not torch.equal(first_mask[0], first_mask[2])
    assert (first_mask.sum(dim=-1) == 3).all()


def test_GumbelTopKSelection_train_mode_keeps_exact_budget_and_is_stochastic():
    selection = GumbelTopKSelection(k=2)
    scores = torch.randn(100, 10)

    first_mask = selection(scores)
    second_mask = selection(scores)

    # The straight-through forward value is an exact-budget hard sample.
    assert ((first_mask == 0.0) | (first_mask == 1.0)).all()
    assert (first_mask.sum(dim=-1) == 2).all()
    # Gumbel noise means repeated draws explore different subsets.
    assert not torch.equal(first_mask, second_mask)


def test_GumbelTopKSelection_train_mode_gradients_flow_to_scores():
    selection = GumbelTopKSelection(k=3)
    scores = torch.randn(6, 12, requires_grad=True)

    mask = selection(scores)
    # The hard mask alone has zero gradient everywhere, so any nonzero
    # gradient must come from the soft Gumbel-softmax path.
    mask.sum().backward()

    assert scores.grad is not None
    assert (scores.grad != 0).any()


def test_gumbel_topk_selection_with_flat_scores_samples_uniform_subsets():
    num_features = 50
    k = 5
    num_samples = 4000
    scores = torch.zeros(num_samples, num_features)

    hard_mask, soft_mask = gumbel_topk_selection(scores, k=k, tau=1.0)

    assert (hard_mask.sum(dim=-1) == k).all()
    # With identical scores the Gumbel Top-k sample must be a uniformly random
    # k-subset: every feature is selected with probability k / num_features.
    per_feature_frequency = hard_mask.mean(dim=0)
    assert per_feature_frequency.max() == pytest.approx(k / num_features, abs=0.03)
    assert per_feature_frequency.min() == pytest.approx(k / num_features, abs=0.03)
    assert hard_mask.mean() == pytest.approx(k / num_features)
    # Each sequential draw contributes 1 / num_features in expectation, so the
    # soft mask's expected entry is exactly 1 - (1 - 1 / num_features) ** k.
    expected_soft_entry = 1 - (1 - 1 / num_features) ** k
    assert soft_mask.mean() == pytest.approx(expected_soft_entry, abs=0.001)
    assert (soft_mask >= 0).all()
    assert (soft_mask <= 1).all()


def test_GumbelTopKTrainingSAE_encode_matches_manual_computation():
    cfg = build_gumbel_topk_sae_training_cfg()
    sae = GumbelTopKTrainingSAE(cfg)
    random_params(sae)
    sae.eval()
    sae_in = torch.randn(20, cfg.d_in)

    feature_acts = sae.encode(sae_in)

    hidden_pre = sae_in @ sae.W_enc + sae.b_enc
    selection_scores = sae.selection_head(hidden_pre)
    mask = topk_indicator(selection_scores, k=cfg.k)
    assert_close(feature_acts, hidden_pre.relu() * mask)
    # The mask is predicted per sample: different inputs select different sets.
    assert not torch.equal(mask[0], mask[1])


def test_GumbelTopKTrainingSAE_training_step_respects_budget_and_trains_head():
    cfg = build_gumbel_topk_sae_training_cfg()
    sae = GumbelTopKTrainingSAE(cfg)
    random_params(sae)
    sae.train()
    step_input = TrainStepInput(
        sae_in=torch.randn(32, cfg.d_in),
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=False,
    )

    output = sae.training_forward_pass(step_input)

    assert torch.isfinite(output.loss)
    # ReLU can zero selected features, but the selection budget is a hard cap.
    num_active = (output.feature_acts != 0).sum(dim=-1)
    assert (num_active <= cfg.k).all()
    assert output.metrics["mean_active_features"] == num_active.float().mean()

    output.loss.backward()
    head_grads = [p.grad for p in sae.selection_head.parameters() if p.grad is not None]
    assert len(head_grads) == len(list(sae.selection_head.parameters()))
    assert any((grad != 0).any() for grad in head_grads)
    assert (sae.W_enc.grad != 0).any()


def test_GumbelTopKTrainingSAE_eval_mode_encode_is_deterministic():
    cfg = build_gumbel_topk_sae_training_cfg()
    sae = GumbelTopKTrainingSAE(cfg)
    random_params(sae)
    sae.eval()
    sae_in = torch.randn(16, cfg.d_in)

    first_acts = sae.encode(sae_in)
    second_acts = sae.encode(sae_in)

    assert torch.equal(first_acts, second_acts)
    assert ((first_acts != 0).sum(dim=-1) <= cfg.k).all()


def test_GumbelTopKTrainingSAE_save_inference_model_is_not_supported(tmp_path):
    cfg = build_gumbel_topk_sae_training_cfg()
    sae = GumbelTopKTrainingSAE(cfg)

    with pytest.raises(NotImplementedError):
        sae.save_inference_model(tmp_path)
