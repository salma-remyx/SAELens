"""
Typical analysis workflow: do an SAE's learned atoms form a stable unit set?

Trains threshold-activated (JumpReLU) SAEs on synthetic data whose ground-truth
feature dictionary is a set of fundamental representational units, then measures
whether the learned dictionary recovers them under the two criteria of
"Representational Atoms in Neural Networks" (https://arxiv.org/abs/2611.14168):

- fidelity: reconstruction R^2 of the SAE.
- stability: the largest quantile at which the learned dictionary still satisfies
  the K-sparse uniqueness condition mu_q < 1 / (2 K_q - 1), where K_q is the
  q-quantile of active latents per sample and mu_q the q-quantile of the absolute
  normalized coherence between decoder atoms.

The contrast the workflow is after: a plain ReLU SAE reconstructs the data best
but its dense codes break uniqueness, while thresholding trades a little fidelity
for a dictionary in which (nearly) every sample has a unique sparse decomposition.

Run with: poetry run pytest benchmark/test_typical_analysis_workflow.py -v -s
"""

from typing import Literal

from sae_lens.analysis.dictionary_stability import (
    RepresentationalAtomResult,
    evaluate_representational_atoms,
)
from sae_lens.saes.jumprelu_sae import JumpReLUTrainingSAE, JumpReLUTrainingSAEConfig
from sae_lens.saes.standard_sae import StandardTrainingSAE, StandardTrainingSAEConfig
from sae_lens.synthetic.activation_generator import ActivationGenerator
from sae_lens.synthetic.feature_dictionary import FeatureDictionary
from sae_lens.synthetic.training import train_toy_sae

NUM_FEATURES = 128
HIDDEN_DIM = 64
TRUE_L0 = 2.0
TRAINING_SAMPLES = 1_000_000
TRAIN_BATCH_SIZE = 1024
LEARNING_RATE = 3e-3
EVAL_SAMPLES = 50_000


def create_ground_truth_fru_setup() -> tuple[FeatureDictionary, ActivationGenerator]:
    """
    Build a synthetic dataset whose features are fundamental representational units.

    The feature dictionary is overcomplete (128 atoms in 64 dimensions), so atoms
    are only approximately orthogonal, matching the regime the paper analyses.
    """
    feature_dict = FeatureDictionary(
        num_features=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
        bias=False,
        initializer=None,
    )
    generator = ActivationGenerator(
        num_features=NUM_FEATURES,
        firing_probabilities=TRUE_L0 / NUM_FEATURES,
        std_firing_magnitudes=0.2,
        mean_firing_magnitudes=1.0,
    )
    return feature_dict, generator


def build_jumprelu_tsae(
    sparsity_loss_mode: Literal["step", "quadratic"], ste_to_input: bool
) -> JumpReLUTrainingSAE:
    """Build a JumpReLU SAE with the given sparsity loss and STE configuration."""
    return JumpReLUTrainingSAE(
        JumpReLUTrainingSAEConfig(
            d_in=HIDDEN_DIM,
            d_sae=NUM_FEATURES,
            dtype="float32",
            device="cpu",
            normalize_activations="none",
            apply_b_dec_to_input=False,
            decoder_init_norm=1.0,
            jumprelu_init_threshold=0.1,
            jumprelu_bandwidth=0.05,
            jumprelu_sparsity_loss_mode=sparsity_loss_mode,
            target_l0=TRUE_L0 if sparsity_loss_mode == "quadratic" else None,
            jumprelu_ste_to_input=ste_to_input,
            l0_coefficient=1e-2,
        )
    )


def report(label: str, result: RepresentationalAtomResult) -> None:
    print(
        f"{label:>24}: fidelity={result.fidelity:.3f} "
        f"stability={result.stability.score:.3f} "
        f"l0={result.stability.mean_sparsity:.2f} "
        f"coherence={result.stability.mean_coherence:.3f}"
    )


def test_typical_representational_atom_workflow() -> None:
    feature_dict, generator = create_ground_truth_fru_setup()
    eval_hidden_acts = feature_dict(generator.sample(EVAL_SAMPLES))

    untrained = build_jumprelu_tsae("quadratic", ste_to_input=True)
    untrained_result = evaluate_representational_atoms(untrained, eval_hidden_acts)

    quadratic = build_jumprelu_tsae("quadratic", ste_to_input=True)
    train_toy_sae(
        quadratic,
        feature_dict,
        generator,
        training_samples=TRAINING_SAMPLES,
        batch_size=TRAIN_BATCH_SIZE,
        lr=LEARNING_RATE,
    )
    quadratic_result = evaluate_representational_atoms(quadratic, eval_hidden_acts)

    step = build_jumprelu_tsae("step", ste_to_input=False)
    train_toy_sae(
        step,
        feature_dict,
        generator,
        training_samples=TRAINING_SAMPLES,
        batch_size=TRAIN_BATCH_SIZE,
        lr=LEARNING_RATE,
    )
    step_result = evaluate_representational_atoms(step, eval_hidden_acts)

    relu = StandardTrainingSAE(
        StandardTrainingSAEConfig(
            d_in=HIDDEN_DIM,
            d_sae=NUM_FEATURES,
            dtype="float32",
            device="cpu",
            normalize_activations="none",
            apply_b_dec_to_input=False,
            decoder_init_norm=1.0,
            l1_coefficient=2e-3,
        )
    )
    train_toy_sae(
        relu,
        feature_dict,
        generator,
        training_samples=TRAINING_SAMPLES,
        batch_size=TRAIN_BATCH_SIZE,
        lr=1e-3,
    )
    relu_result = evaluate_representational_atoms(relu, eval_hidden_acts)

    print("\nRepresentational atom criteria on synthetic FRUs:")
    report("jumprelu untrained", untrained_result)
    report("jumprelu quadratic + ste", quadratic_result)
    report("jumprelu step", step_result)
    report("relu", relu_result)

    # a trained threshold-activated SAE recovers a usable unit set: it reconstructs
    # the data and its codes stay sparse enough for uniqueness to hold
    assert quadratic_result.fidelity > 0.8
    assert quadratic_result.stability.score > 0.5
    assert quadratic_result.fidelity > untrained_result.fidelity
    assert quadratic_result.stability.score > untrained_result.stability.score

    # anchoring the sparsity loss on a target L0 keeps codes tight, which the
    # stability score rewards even though fidelity is unchanged
    assert quadratic_result.stability.score > step_result.stability.score

    # a plain ReLU SAE reconstructs best of all but its dense codes break the
    # uniqueness condition, so its atoms are not stable units despite the fidelity
    assert relu_result.fidelity > 0.9
    assert relu_result.stability.score < 0.2
