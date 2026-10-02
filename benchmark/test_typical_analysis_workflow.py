"""Paraphrase-paired training workflow for BatchTopK SAEs.

Exercises the training + boundary-reliability workflow recommended by "Active
Budget Can Kill Sensitivity: Diagnosing and Repairing TopK Sparse Autoencoder
Reliability" (arxiv:2609.37857): train a BatchTopK SAE with and without
pairwise rank stabilization on activations whose rows are grouped into
paraphrase pairs (same underlying features, independent surface noise), then
measure feature sensitivity (active-set overlap across pair members),
selection-boundary geometry (active margin), and how reliably features stay
selected across pair members as a function of their active margin.

The margin-conditioned flip-rate check reproduces the paper's diagnosis at
this scale: ordering failures concentrate at the selection boundary, among
features whose active margin is small. Whether the rank stabilization loss
moves sensitivity on top of the baseline is a full-scale training question
(the paper trains wide SAEs on real paraphrase data for far longer than this
CPU miniature), so the baseline-vs-stabilized numbers are reported rather
than asserted.

Run with: poetry run pytest benchmark/test_typical_analysis_workflow.py -v -s
"""

import torch

from sae_lens.registry import (
    SAE_TRAINING_CLASS_REGISTRY,
    get_sae_training_class,
    register_sae_training_class,
)
from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.rank_stabilized_batchtopk_sae import (
    RankStabilizedBatchTopKTrainingSAE,
    RankStabilizedBatchTopKTrainingSAEConfig,
    active_margin,
    batchtopk_cutoff,
    feature_sensitivity,
    mean_active_margin,
)
from sae_lens.saes.sae import TrainStepInput

D_IN = 16
N_CONCEPTS = 24
D_SAE = 128
K = 6
NOISE_SCALE = 0.5
N_TRAIN_PAIRS = 512
N_EVAL_PAIRS = 2048
N_STEPS = 600
BATCH_PAIRS = 32
LR = 1e-3
RANK_LOSS_COEFFICIENT = 1.0


def generate_paired_activations(
    directions: torch.Tensor,
    concept_probs: torch.Tensor,
    n_pairs: int,
    active_concepts: int,
    noise_scale: float,
) -> torch.Tensor:
    """
    Build activations of shape (2 * n_pairs, d_in) where rows (2i, 2i+1) are a
    paraphrase pair: both members combine the same randomly drawn concept
    directions (with Zipf-drawn frequencies, so concepts range from common to
    rare), then add independent surface noise.
    """
    concept_ids = torch.multinomial(
        concept_probs, n_pairs * active_concepts, replacement=True
    ).view(n_pairs, active_concepts)
    magnitudes = torch.rand(n_pairs, active_concepts) + 0.5
    shared = (directions[concept_ids] * magnitudes[..., None]).sum(dim=1)
    members = shared.unsqueeze(0) + torch.randn(
        2, n_pairs, directions.shape[1]
    ) * noise_scale
    return members.permute(1, 0, 2).reshape(-1, directions.shape[1])


def paired_batch_plan(
    n_pairs: int, n_steps: int, batch_pairs: int
) -> list[torch.Tensor]:
    """Row indices of n_steps paired mini-batches (rows (2i, 2i+1) stay together)."""
    batch_plan = []
    for _ in range(n_steps):
        pair_ids = torch.randint(n_pairs, (batch_pairs,))
        batch_plan.append(
            torch.stack([pair_ids * 2, pair_ids * 2 + 1], dim=1).reshape(-1)
        )
    return batch_plan


def train_on_paired_batches(
    sae: BatchTopKTrainingSAE,
    activations: torch.Tensor,
    batch_plan: list[torch.Tensor],
    lr: float,
) -> None:
    """Train the SAE with its own training_forward_pass on paired mini-batches."""
    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr)
    for step, rows in enumerate(batch_plan):
        step_input = TrainStepInput(
            sae_in=activations[rows],
            coefficients={},
            dead_neuron_mask=None,
            n_training_steps=step,
            is_logging_step=False,
        )
        output = sae.training_forward_pass(step_input)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()


def evaluate_on_paired_batches(
    sae: BatchTopKTrainingSAE, activations: torch.Tensor
) -> dict[str, float]:
    """Measure feature sensitivity, boundary geometry, and reconstruction."""
    with torch.no_grad():
        feature_acts, hidden_pre = sae.encode_with_hidden_pre(activations)
        flat_hidden_pre = hidden_pre.reshape(-1, sae.cfg.d_sae)
        cutoff = batchtopk_cutoff(flat_hidden_pre, sae.cfg.k)
        reconstruction = sae.decode(feature_acts)
        mse = ((reconstruction - activations) ** 2).sum(dim=-1).mean()
    return {
        "feature_sensitivity": feature_sensitivity(feature_acts).item(),
        "mean_active_margin": mean_active_margin(flat_hidden_pre, cutoff).item(),
        "active_cutoff": cutoff.item(),
        "mse_loss": mse.item(),
    }


def boundary_flip_rates(
    sae: BatchTopKTrainingSAE, activations: torch.Tensor
) -> tuple[float, float]:
    """
    Fraction of features kept in the first member of a pair that get dropped
    in the second member, split by whether their active margin was below or
    above the median kept margin. This is the paper's diagnosis: the active
    margin predicts feature loss, with ordering failures concentrated at the
    selection boundary.
    """
    with torch.no_grad():
        _, hidden_pre = sae.encode_with_hidden_pre(activations)
        flat_hidden_pre = hidden_pre.reshape(-1, sae.cfg.d_sae)
        cutoff = batchtopk_cutoff(flat_hidden_pre, sae.cfg.k)
        margins = active_margin(flat_hidden_pre, cutoff).reshape(-1, 2, sae.cfg.d_sae)
        kept_in_first = margins[:, 0] > 0
        dropped_in_second = margins[:, 1] <= 0
        median_margin = margins[:, 0][kept_in_first].median()
        small_margin = kept_in_first & (margins[:, 0] <= median_margin)
        large_margin = kept_in_first & (margins[:, 0] > median_margin)
        flip_rate = (
            (dropped_in_second & small_margin).sum() / small_margin.sum(),
            (dropped_in_second & large_margin).sum() / large_margin.sum(),
        )
    return flip_rate[0].item(), flip_rate[1].item()


def test_typical_paired_training_workflow():
    if "rank_stabilized_batchtopk" not in SAE_TRAINING_CLASS_REGISTRY:
        register_sae_training_class(
            "rank_stabilized_batchtopk",
            RankStabilizedBatchTopKTrainingSAE,
            RankStabilizedBatchTopKTrainingSAEConfig,
        )
    sae_class, _ = get_sae_training_class("rank_stabilized_batchtopk")
    assert sae_class is RankStabilizedBatchTopKTrainingSAE

    concept_probs = 1.0 / torch.arange(1, N_CONCEPTS + 1).float()
    concept_probs /= concept_probs.sum()
    directions = torch.randn(N_CONCEPTS, D_IN)
    train_activations = generate_paired_activations(
        directions,
        concept_probs,
        N_TRAIN_PAIRS,
        active_concepts=4,
        noise_scale=NOISE_SCALE,
    )
    eval_activations = generate_paired_activations(
        directions,
        concept_probs,
        N_EVAL_PAIRS,
        active_concepts=4,
        noise_scale=NOISE_SCALE,
    )
    batch_plan = paired_batch_plan(N_TRAIN_PAIRS, N_STEPS, BATCH_PAIRS)

    baseline = BatchTopKTrainingSAE(
        BatchTopKTrainingSAEConfig(d_in=D_IN, d_sae=D_SAE, k=K, decoder_init_norm=0.1)
    )
    # the stabilized SAE is built through from_dict, which resolves the
    # architecture via the training-class registry like the standard runner
    stabilized = RankStabilizedBatchTopKTrainingSAE.from_dict(
        RankStabilizedBatchTopKTrainingSAEConfig(
            d_in=D_IN,
            d_sae=D_SAE,
            k=K,
            decoder_init_norm=0.1,
            pairwise_rank_loss_coefficient=RANK_LOSS_COEFFICIENT,
        ).to_dict()
    )
    # identical starting weights and identical batches, so the only difference
    # between the two runs is the pairwise rank stabilization loss
    stabilized.load_state_dict(baseline.state_dict())

    initial_metrics = evaluate_on_paired_batches(baseline, eval_activations)
    train_on_paired_batches(baseline, train_activations, batch_plan, LR)
    train_on_paired_batches(stabilized, train_activations, batch_plan, LR)

    baseline_metrics = evaluate_on_paired_batches(baseline, eval_activations)
    stabilized_metrics = evaluate_on_paired_batches(stabilized, eval_activations)
    baseline_flips = boundary_flip_rates(baseline, eval_activations)
    stabilized_flips = boundary_flip_rates(stabilized, eval_activations)

    print(f"\ninitial:     {initial_metrics}")
    print(f"baseline:    {baseline_metrics}")
    print(f"stabilized:  {stabilized_metrics}")
    print(f"baseline flip rates (small-margin, large-margin):    {baseline_flips}")
    print(f"stabilized flip rates (small-margin, large-margin):  {stabilized_flips}")

    # both runs trained end-to-end through their own training_forward_pass
    assert baseline_metrics["mse_loss"] < 0.5 * initial_metrics["mse_loss"]
    assert stabilized_metrics["mse_loss"] < 0.5 * initial_metrics["mse_loss"]
    for metrics in (baseline_metrics, stabilized_metrics):
        assert 0.0 < metrics["feature_sensitivity"] <= 1.0
        assert metrics["mean_active_margin"] > 0.0
    # the paper's diagnosis: features kept by a small active margin are the
    # ones that flip across paraphrase pairs; features kept comfortably stay
    for flips in (baseline_flips, stabilized_flips):
        assert flips[0] > 0.05
        assert flips[0] > 10 * flips[1]
