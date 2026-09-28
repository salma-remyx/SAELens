"""
End-to-end analysis workflow: set-level stability of SAE latent sets.

Trains a small BatchTopK SAE on synthetic features with known ground truth,
then measures how much the SAE's active latent set changes when a controlled
number of ground-truth features are swapped out of each input. Active-set
overlap (Jaccard) between matched inputs replaces cosine similarity over
dense representations, following "Beyond a Bag of Features: Set-Level
Instability in Sparse Autoencoders" (arXiv:2608.11197).

Run with: poetry run pytest benchmark/test_typical_analysis_workflow.py -v -s
"""

import pytest
import torch

from sae_lens.analysis.latent_set_stability import latent_set_stability
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.saes.sae import TrainStepInput
from sae_lens.synthetic import ActivationGenerator, FeatureDictionary


def train_batchtopk_sae_on_synthetic_features(
    num_features: int = 100,
    hidden_dim: int = 64,
    d_sae: int = 256,
    k: float = 10,
    firing_probability: float = 0.1,
    n_training_batches: int = 400,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> tuple[BatchTopKTrainingSAE, FeatureDictionary]:
    """
    Train a BatchTopK SAE to reconstruct hidden activations of synthetic features.
    """
    generator = ActivationGenerator(
        num_features=num_features,
        firing_probabilities=firing_probability,
        device=device,
    )
    feature_dictionary = FeatureDictionary(
        num_features=num_features,
        hidden_dim=hidden_dim,
        bias=False,
        device=device,
    )
    sae = BatchTopKTrainingSAE(
        BatchTopKTrainingSAEConfig(
            d_in=hidden_dim,
            d_sae=d_sae,
            k=k,
            apply_b_dec_to_input=False,
            device=device,
        )
    )

    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)
    torch.set_grad_enabled(True)
    for step in range(n_training_batches):
        step_input = TrainStepInput(
            sae_in=feature_dictionary(generator.sample(batch_size)),
            coefficients={},
            dead_neuron_mask=None,
            n_training_steps=step,
            is_logging_step=False,
        )
        output = sae.training_forward_pass(step_input)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
    return sae, feature_dictionary


def sample_with_fixed_active_sets(
    num_samples: int, num_features: int, active_per_sample: int, device: str
) -> torch.Tensor:
    """
    Random samples with exactly active_per_sample features active (value 1.0).
    """
    active_indices = (
        torch.rand(num_samples, num_features, device=device)
        .topk(active_per_sample, dim=-1)
        .indices
    )
    samples = torch.zeros(num_samples, num_features, device=device)
    return samples.scatter(-1, active_indices, 1.0)


def _lowest_ranked_within(mask: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """
    Rank the entries selected by mask by score, 0 being the lowest score.

    Entries not selected by mask get rank num_features, so they never rank
    below a selected entry.
    """
    num_features = scores.shape[-1]
    masked_scores = scores.masked_fill(~mask, float("inf"))
    ranks = masked_scores.argsort(dim=-1).argsort(dim=-1)
    return ranks.masked_fill(~mask, num_features)


def swap_active_features(samples: torch.Tensor, n_swapped: int) -> torch.Tensor:
    """
    Swap n_swapped active features for inactive ones in every row.

    Each row keeps its number of active features: the n_swapped lowest-ranked
    active features (by a fresh random draw) are zeroed and the same number of
    previously inactive features are activated. This is the controlled
    modification the set-stability analysis is applied to.
    """
    active = samples > 0
    scores = torch.rand_like(samples)
    turn_off = active & (_lowest_ranked_within(active, scores) < n_swapped)
    turn_on = ~active & (_lowest_ranked_within(~active, scores) < n_swapped)
    modified = samples.clone()
    modified[turn_off] = 0.0
    modified[turn_on] = 1.0
    return modified


def test_latent_set_stability_under_controlled_feature_swaps():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_features = 100
    active_per_sample = 10

    sae, feature_dictionary = train_batchtopk_sae_on_synthetic_features(
        num_features=num_features,
        hidden_dim=64,
        d_sae=256,
        k=active_per_sample,
        device=device,
    )

    num_probes = 256
    base_samples = sample_with_fixed_active_sets(
        num_probes, num_features, active_per_sample, device
    )
    base_hidden = feature_dictionary(base_samples)

    n_swapped_values = [0, 1, 2, 4, 8]
    mean_overlaps = []
    for n_swapped in n_swapped_values:
        modified_hidden = feature_dictionary(
            swap_active_features(base_samples, n_swapped)
        )
        metrics = latent_set_stability(sae, base_hidden, modified_hidden)
        mean_overlaps.append(metrics["mean_jaccard_overlap"])
        print(
            f"swapped {n_swapped}/{active_per_sample} features -> "
            f"mean active-set overlap {metrics['mean_jaccard_overlap']:.3f} "
            f"(identical sets: {metrics['identical_set_fraction']:.3f})"
        )

    assert metrics["mean_active_set_size_base"] == pytest.approx(active_per_sample)
    assert mean_overlaps[0] == pytest.approx(1.0)
    assert mean_overlaps[4] < mean_overlaps[2] < mean_overlaps[1] < 1.0
    assert mean_overlaps[4] < 0.5
