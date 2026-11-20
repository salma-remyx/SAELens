"""
End-to-end comparison of SAE mode recovery against classical baselines.

Port of the synthetic validation experiment from "User Embeddings are
Superpositions of Interpretable Behavioral Modes"
(https://arxiv.org/abs/2611.08487): dense embeddings are generated as sparse
superpositions of ground-truth behavioral modes (more modes than embedding
dimensions), a BatchTopK SAE is trained on them with train_toy_sae, and its
mode recovery (Mean Correlation Coefficient against the ground truth modes)
is compared against classical dictionary-learning baselines fit on the same
distribution.
"""

import pytest

from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.synthetic import (
    ActivationGenerator,
    FeatureDictionary,
    eval_sae_on_synthetic_data,
    train_toy_sae,
)
from sae_lens.synthetic.classical_baselines import evaluate_classical_baselines

NUM_MODES = 48
HIDDEN_DIM = 32
TRUE_L0 = 4.0


def test_batchtopk_sae_recovers_superposed_modes_better_than_classical_baselines():
    feature_dict = FeatureDictionary(num_features=NUM_MODES, hidden_dim=HIDDEN_DIM)
    activations_gen = ActivationGenerator(
        num_features=NUM_MODES,
        firing_probabilities=TRUE_L0 / NUM_MODES,
    )
    sae = BatchTopKTrainingSAE(
        BatchTopKTrainingSAEConfig(
            d_in=HIDDEN_DIM,
            d_sae=128,
            k=TRUE_L0,
            dtype="float32",
            device="cpu",
        )
    )
    train_toy_sae(
        sae=sae,
        feature_dict=feature_dict,
        activations_generator=activations_gen,
        training_samples=2_000_000,
        batch_size=1024,
        lr=1e-3,
    )

    sae_result = eval_sae_on_synthetic_data(
        sae,
        feature_dict,
        activations_gen,
        num_samples=8192,
        batch_size=2048,
    )
    baseline_activations = feature_dict(activations_gen.sample(16_384))
    baseline_scores = evaluate_classical_baselines(
        baseline_activations,
        feature_dict.feature_vectors,
        num_components=NUM_MODES,
    )

    # The SAE must genuinely recover modes, not merely beat weak baselines.
    assert sae_result.mcc > 0.9
    assert sae_result.sae_l0 == pytest.approx(sae_result.true_l0, abs=0.25)
    for method, score in baseline_scores.items():
        assert sae_result.mcc > score, (
            f"SAE mode recovery ({sae_result.mcc:.3f}) did not exceed the "
            f"{method} baseline ({score:.3f})"
        )
