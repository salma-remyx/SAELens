"""Typical analysis workflow: FRU fidelity and stability for a JumpReLU TSAE.

Scores the two representational-atom criteria from "Representational Atoms
in Neural Networks" (arXiv:2611.14168) on a threshold-activated SAE:

1. Build a ground-truth FRU dictionary with FeatureDictionary and generate
   K-sparse representations with ActivationGenerator.
2. Wire a JumpReLUSAE with the paper's Theorem 11 construction
   (W_dec = D, W_enc = (D D^T)^-1 D), which provably identifies the
   ground-truth atoms.
3. Score atom recovery with evaluate_atom_recovery, then sweep the JumpReLU
   threshold to show how threshold control trades off fidelity, stability,
   and sparsity.
"""

import pytest
import torch

from sae_lens.analysis.atom_recovery import evaluate_atom_recovery
from sae_lens.saes.jumprelu_sae import JumpReLUSAE, JumpReLUSAEConfig
from sae_lens.synthetic import (
    ActivationGenerator,
    FeatureDictionary,
    zipfian_firing_probabilities,
)

NUM_FEATURES = 64
HIDDEN_DIM = 64
NUM_SAMPLES = 16_384


def _build_ground_truth_fru_dataset() -> tuple[FeatureDictionary, torch.Tensor]:
    feature_dict = FeatureDictionary(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM)
    generator = ActivationGenerator(
        num_features=NUM_FEATURES,
        firing_probabilities=zipfian_firing_probabilities(NUM_FEATURES, min_prob=0.05),
        std_firing_magnitudes=0.2,
        mean_firing_magnitudes=1.0,
    )
    feature_acts = generator.sample(NUM_SAMPLES)
    activations = feature_dict(feature_acts)
    return feature_dict, activations


def _build_jumprelu_tsae(
    feature_vectors: torch.Tensor, threshold: float
) -> JumpReLUSAE:
    sae = JumpReLUSAE(
        JumpReLUSAEConfig(
            d_in=feature_vectors.shape[1],
            d_sae=feature_vectors.shape[0],
            dtype="float32",
            device="cpu",
            normalize_activations="none",
        )
    )
    with torch.no_grad():
        sae.W_dec.copy_(feature_vectors)
        sae.W_enc.copy_(
            torch.linalg.solve(feature_vectors.T @ feature_vectors, feature_vectors.T)
        )
        sae.b_enc.zero_()
        sae.b_dec.zero_()
        sae.threshold.fill_(threshold)
    return sae


# Run with: poetry run pytest benchmark/test_typical_analysis_workflow.py -v -s
def test_typical_fru_recovery_workflow():
    feature_dict, activations = _build_ground_truth_fru_dataset()
    sae = _build_jumprelu_tsae(feature_dict.feature_vectors, threshold=0.05)
    metrics = evaluate_atom_recovery(sae, activations)
    print(
        f"\nfidelity={metrics.fidelity:.4f} stability={metrics.stability:.4f} "
        f"coherence={metrics.stability_coherence:.4f} "
        f"sparsity={metrics.stability_sparsity:.1f} mean_l0={metrics.mean_l0:.2f}"
    )

    assert metrics.fidelity == pytest.approx(1.0, abs=1e-3)
    assert metrics.stability > 0.9
    assert 1.0 < metrics.mean_l0 < NUM_FEATURES
    assert metrics.stability_coherence < 1.0 / (2.0 * metrics.stability_sparsity - 1.0)


def test_fru_criteria_respond_to_threshold_control():
    feature_dict, activations = _build_ground_truth_fru_dataset()
    thresholds = [0.05, 0.5, 1.5]
    all_metrics = []
    for threshold in thresholds:
        sae = _build_jumprelu_tsae(feature_dict.feature_vectors, threshold)
        metrics = evaluate_atom_recovery(sae, activations)
        all_metrics.append(metrics)
        print(
            f"threshold={threshold:.2f} fidelity={metrics.fidelity:+.4f} "
            f"stability={metrics.stability:.4f} mean_l0={metrics.mean_l0:.2f}"
        )

    # Raising the threshold can only shrink the active set, so fidelity and
    # L0 are monotonically non-increasing in the threshold.
    for above, below in zip(all_metrics[:-1], all_metrics[1:]):
        assert above.fidelity >= below.fidelity
        assert above.mean_l0 >= below.mean_l0
    assert all_metrics[0].fidelity == pytest.approx(1.0, abs=1e-3)
    assert all_metrics[1].fidelity < all_metrics[0].fidelity
    # A threshold above every firing magnitude silences the SAE entirely,
    # which is worse than predicting the activation mean (negative R^2).
    assert all_metrics[2].fidelity < 0.0
