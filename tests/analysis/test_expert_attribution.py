import pytest
import torch

from sae_lens.analysis.expert_attribution import (
    aggregate_expert_contributions,
    analyze_expert_dominance,
    dominance_sets,
    dominant_set_overlap,
    effective_contributors,
    expert_latent_contributions,
    expert_latent_contributions_from_sae,
)
from sae_lens.saes.topk_sae import TopKTrainingSAE
from tests.helpers import build_topk_sae_training_cfg, random_params


def test_expert_latent_contributions_matches_hand_computed_values():
    W_enc = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    expert_outputs = torch.tensor([[[2.0, 0.0, 1.0], [0.0, 4.0, 0.0]]])
    router_weights = torch.tensor([[0.25, 0.5]])

    contributions = expert_latent_contributions(expert_outputs, router_weights, W_enc)

    expected = torch.tensor([[[0.75, 0.25], [0.0, 2.0]]])
    assert torch.allclose(contributions, expected)


def test_expert_latent_contributions_rejects_mismatched_expert_dimensions():
    with pytest.raises(ValueError, match="incompatible"):
        expert_latent_contributions(
            torch.randn(4, 3, 8), torch.randn(4, 2), torch.randn(8, 16)
        )


def test_expert_latent_contributions_decomposes_hidden_pre_exactly():
    d_in, d_sae, num_experts, num_tokens = 6, 8, 3, 20
    W_enc = torch.randn(d_in, d_sae)
    b_enc = torch.randn(d_sae)
    expert_outputs = torch.randn(num_tokens, num_experts, d_in)
    router_weights = torch.softmax(torch.randn(num_tokens, num_experts), dim=-1)
    moe_output = (router_weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)

    contributions = expert_latent_contributions(expert_outputs, router_weights, W_enc)

    hidden_pre = moe_output @ W_enc + b_enc
    assert torch.allclose(
        contributions.sum(dim=-2) + b_enc, hidden_pre, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("rescale_acts_by_decoder_norm", [False, True])
def test_contributions_from_sae_reconstruct_encode_with_hidden_pre(
    rescale_acts_by_decoder_norm: bool,
):
    cfg = build_topk_sae_training_cfg(
        d_in=8,
        d_sae=32,
        k=5,
        rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
        device="cpu",
    )
    sae = TopKTrainingSAE(cfg)
    random_params(sae)
    expert_outputs = torch.randn(16, 4, 8)
    router_weights = torch.softmax(torch.randn(16, 4), dim=-1)
    moe_output = (router_weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)

    contributions = expert_latent_contributions_from_sae(
        sae, expert_outputs, router_weights
    )
    _, hidden_pre = sae.encode_with_hidden_pre(moe_output)

    scale = (
        sae.W_dec.norm(dim=-1)
        if rescale_acts_by_decoder_norm
        else torch.ones(sae.cfg.d_sae)
    )
    assert torch.allclose(
        contributions.sum(dim=-2) + sae.b_enc * scale, hidden_pre, rtol=1e-5, atol=1e-6
    )


def test_aggregate_expert_contributions_uses_positive_mass_and_latent_mask():
    contributions = torch.tensor([[[2.0, -1.0, 1.0], [1.0, 4.0, -2.0]]])

    aggregate = aggregate_expert_contributions(contributions)

    assert torch.equal(aggregate, torch.tensor([[2.0, 0.0, 1.0], [1.0, 4.0, 0.0]]))
    masked = aggregate_expert_contributions(
        contributions, latent_mask=torch.tensor([[True, True, False]])
    )
    assert torch.equal(masked, torch.tensor([[2.0, 0.0, 0.0], [1.0, 4.0, 0.0]]))


def test_dominance_sets_and_overlap_match_hand_computed_values():
    aggregate = torch.tensor([[0.9, 0.85, 0.15, 0.6], [0.1, 0.15, 0.85, 0.4]])

    dominant = dominance_sets(aggregate, tau=0.2)

    expected = torch.tensor([[True, True, False, True], [False, False, True, True]])
    assert torch.equal(dominant, expected)
    overlap = dominant_set_overlap(dominant)
    # Dominant sets are {0, 1, 3} and {2, 3}: intersection 1, union 4.
    assert overlap[0, 1].item() == pytest.approx(0.25)
    assert overlap[1, 0].item() == pytest.approx(0.25)
    assert overlap[0, 0].item() == pytest.approx(1.0)
    assert overlap[1, 1].item() == pytest.approx(1.0)


def test_dominance_sets_excludes_latents_with_no_contribution_mass():
    aggregate = torch.tensor([[0.0, 0.5], [0.0, 0.5]])

    dominant = dominance_sets(aggregate, tau=0.2)

    assert not dominant[:, 0].any()
    assert dominant[:, 1].all()


def test_effective_contributors_for_uniform_and_single_expert_mass():
    uniform_over_four = torch.ones(4, 8)
    assert torch.allclose(
        effective_contributors(uniform_over_four, dim=0), torch.full((8,), 4.0)
    )
    assert torch.allclose(effective_contributors(torch.eye(4), dim=-1), torch.ones(4))


def test_analyze_expert_dominance_recovers_planted_expert_latent_blocks():
    num_experts, latents_per_expert, num_tokens = 4, 6, 50
    d_sae = num_experts * latents_per_expert
    contributions = torch.zeros(num_tokens, num_experts, d_sae)
    router_weights = torch.softmax(torch.randn(num_tokens, num_experts), dim=-1)
    for expert in range(num_experts):
        block = slice(expert * latents_per_expert, (expert + 1) * latents_per_expert)
        contributions[:, expert, block] = (
            torch.rand(num_tokens, latents_per_expert) + 0.1
        )
        # Small cross-expert leakage so shares are close to but not exactly 1.
        contributions[:, :, block] += 0.002

    analysis = analyze_expert_dominance(contributions, router_weights, tau=0.2)

    assert torch.equal(
        analysis.dominators_per_latent, torch.ones(d_sae, dtype=torch.long)
    )
    for expert in range(num_experts):
        assert analysis.dominant[expert].sum() == latents_per_expert
    assert torch.allclose(analysis.overlap, torch.eye(num_experts), atol=1e-6)
    assert analysis.latent_concentration.min() >= 1.0
    assert analysis.latent_concentration.max() < 1.05
    # The paper's headline stat: contributions are far more concentrated than
    # gating (~1 contributing expert per latent vs ~num_experts routed to).
    assert analysis.gating_concentration > analysis.latent_concentration.max()
