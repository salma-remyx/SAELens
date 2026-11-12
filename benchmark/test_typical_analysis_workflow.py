import pytest
import torch

from sae_lens.analysis.expert_attribution import (
    analyze_expert_dominance,
    expert_latent_contributions_from_sae,
)
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig

# Run with: poetry run pytest benchmark/test_typical_analysis_workflow.py -v -s


def test_typical_moe_expert_dominance_workflow():
    num_experts, latents_per_expert = 8, 16
    d_model = num_experts * latents_per_expert
    num_tokens, experts_per_token = 2048, 4

    # Synthetic MoE layer with planted structure: expert e only ever writes to
    # the block of basis directions [e * latents_per_expert, ...).
    logits = torch.randn(num_tokens, num_experts)
    top = logits.topk(experts_per_token, dim=-1)
    router_weights = torch.zeros(num_tokens, num_experts).scatter(
        -1, top.indices, torch.softmax(top.values, dim=-1)
    )
    expert_outputs = torch.zeros(num_tokens, num_experts, d_model)
    magnitudes = torch.rand(num_tokens, num_experts, latents_per_expert) + 0.1
    for expert in range(num_experts):
        block = slice(expert * latents_per_expert, (expert + 1) * latents_per_expert)
        expert_outputs[:, expert, block] = magnitudes[:, expert]

    # TopK SAE whose encoder is the identity, so latents align with the
    # planted expert directions (decoder-norm rescaling is then a no-op).
    sae = TopKTrainingSAE(
        TopKTrainingSAEConfig(
            d_in=d_model, d_sae=d_model, k=32, apply_b_dec_to_input=False, device="cpu"
        )
    )
    identity = torch.eye(d_model)
    sae.W_enc.data = identity.clone()
    sae.W_dec.data = identity.clone()
    sae.b_enc.data = torch.zeros(d_model)
    sae.b_dec.data = torch.zeros(d_model)

    moe_output = (router_weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)
    feature_acts, hidden_pre = sae.encode_with_hidden_pre(moe_output)
    contributions = expert_latent_contributions_from_sae(
        sae, expert_outputs, router_weights
    )

    # The paper's exact affine decomposition: expert contributions plus the
    # encoder bias reproduce the SAE's own pre-activations.
    assert torch.allclose(contributions.sum(dim=-2) + sae.b_enc, hidden_pre, atol=1e-5)

    analysis = analyze_expert_dominance(
        contributions, router_weights, latent_mask=feature_acts > 0
    )
    active = analysis.dominators_per_latent > 0
    dominator_counts = analysis.dominators_per_latent[active].unique().tolist()
    print(f"Active latents: {int(active.sum())} / {d_model}")
    print(f"Dominators per active latent: {dominator_counts}")
    print(f"Gating concentration: {analysis.gating_concentration:.3f}")
    print(f"Dominant set sizes: {analysis.dominant.sum(dim=-1).tolist()}")

    # Planted structure recovered: a single dominator per active latent,
    # disjoint dominant sets across experts, and contributions far more
    # concentrated than gating.
    assert (analysis.dominators_per_latent[active] == 1).all()
    assert analysis.overlap[~torch.eye(num_experts, dtype=torch.bool)].sum() == 0
    assert analysis.latent_concentration[active].max().item() == pytest.approx(1.0)
    assert analysis.gating_concentration > 5.0
