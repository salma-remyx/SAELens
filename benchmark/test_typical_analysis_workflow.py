"""
Typical analysis workflow: score an SAE dictionary as a set of representational atoms.

Trains a JumpReLU (threshold-activated) SAE on synthetic data with a known
feature dictionary, then reads the learned decoder as a candidate set of
fundamental representational units and scores it with the two criteria from
"Representational Atoms in Neural Networks" (https://arxiv.org/abs/2611.14168):

- fidelity, reported as the R^2 of the SAE reconstruction by
  `eval_sae_on_synthetic_data`
- stability, reported by `eval_sae_atom_stability` as the largest quantile of
  codes whose coherence mu_q and sparsity K_q jointly clear the
  distinguishability bound mu_q < 1 / (2 K_q - 1)

Sweeping the threshold moves along the fidelity/stability frontier the paper
reports: looser thresholds keep more atoms per code, which reconstructs better
but leaves the codes too dense to be uniquely decodable.

Run with: poetry run pytest benchmark/test_typical_analysis_workflow.py -v -s
"""

import torch

from sae_lens.saes.jumprelu_sae import JumpReLUTrainingSAE, JumpReLUTrainingSAEConfig
from sae_lens.synthetic import (
    ActivationGenerator,
    FeatureDictionary,
    eval_sae_on_synthetic_data,
    init_sae_to_match_feature_dict,
    train_toy_sae,
)
from sae_lens.synthetic.representational_atoms import eval_sae_atom_stability

HIDDEN_DIM = 16
NUM_FEATURES = 24
FIRING_PROBABILITY = 0.1
NUM_EVAL_SAMPLES = 50_000
TRUE_L0 = NUM_FEATURES * FIRING_PROBABILITY
THRESHOLDS = (0.0, 0.25, 0.5, 1.0)


def test_typical_representational_atom_workflow():
    feature_dict = FeatureDictionary(
        num_features=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
        bias=False,
        initializer=None,
    )
    activations_gen = ActivationGenerator(
        num_features=NUM_FEATURES,
        firing_probabilities=FIRING_PROBABILITY,
    )

    # A threshold-activated SAE, set up the way SAELens trains them: directly
    # parameterized threshold, STE to the input, GemmaScope-2 quadratic sparsity
    # loss targeting the true L0 of the generator.
    cfg = JumpReLUTrainingSAEConfig(
        d_in=HIDDEN_DIM,
        d_sae=NUM_FEATURES,
        jumprelu_sparsity_loss_mode="quadratic",
        target_l0=TRUE_L0,
        jumprelu_ste_to_input=True,
        l0_coefficient=1e-3,
    )
    sae = JumpReLUTrainingSAE(cfg)

    # Start from the ground truth dictionary, and use the least-squares encoder
    # for it, so the decoder really is usable as a sparse codebook.
    init_sae_to_match_feature_dict(sae, feature_dict)
    with torch.no_grad():
        sae.W_enc.data = torch.linalg.pinv(sae.W_dec.data)
        sae.b_dec.zero_()

    train_toy_sae(
        sae,
        feature_dict,
        activations_gen,
        training_samples=200_000,
        batch_size=1024,
        lr=1e-3,
    )

    print(f"\nGround truth L0: {TRUE_L0:.2f}")
    print(
        f"{'threshold':>9} {'fidelity':>9} {'stability':>9} "
        f"{'retention':>9} {'coherence':>9} {'mean L0':>7}"
    )

    fidelities = []
    stabilities = []
    coherences = []
    sparsities = []
    for threshold in THRESHOLDS:
        with torch.no_grad():
            sae.threshold.data.fill_(threshold)

        eval_result = eval_sae_on_synthetic_data(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=NUM_EVAL_SAMPLES,
            batch_size=10_000,
        )
        atoms = eval_sae_atom_stability(
            sae=sae,
            feature_dict=feature_dict,
            activations_generator=activations_gen,
            num_samples=NUM_EVAL_SAMPLES,
            batch_size=10_000,
        )

        fidelities.append(eval_result.explained_variance)
        stabilities.append(atoms.stability)
        coherences.append(atoms.coherence)
        sparsities.append(atoms.mean_atom_sparsity)
        print(
            f"{threshold:>9.2f} {eval_result.explained_variance:>9.4f} "
            f"{atoms.stability:>9.4f} {atoms.retention:>9.4f} "
            f"{atoms.coherence:>9.4f} {atoms.mean_atom_sparsity:>7.3f}"
        )

    # The dictionary is fixed across the sweep, so its coherence cannot move.
    assert max(coherences) - min(coherences) < 1e-9

    # Raising the threshold only ever zeroes more latents, so the codes get
    # sparser, which relaxes the distinguishability bound while throwing away
    # more of the reconstruction: stability climbs, fidelity falls.
    for looser, tighter in zip(sparsities, sparsities[1:]):
        assert tighter <= looser
    for looser, tighter in zip(stabilities, stabilities[1:]):
        assert tighter >= looser
    for looser, tighter in zip(fidelities, fidelities[1:]):
        assert tighter <= looser

    assert fidelities[0] > 0.5
    assert stabilities[-1] > 0.9
    assert stabilities[0] < stabilities[-1]
