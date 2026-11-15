"""
Typical analysis workflow: atom recovery metrics for a trained JumpReLU SAE.

Builds a synthetic model with a known feature dictionary, attaches a JumpReLU
SAE (a threshold-activated SAE) whose decoder either is the true atom set or a
superposed variant of it, and scores both with
`sae_lens.analysis.atom_recovery.evaluate_atom_recovery`:

- Fidelity (R^2 of the atom reconstruction) catches dictionaries that miss or
  merge features, whatever the SAE's own encoder scores.
- Stability (the quantile coherence criterion) catches dictionaries that
  reconstruct fine but whose atoms are too coherent for their codes to be
  unique -- the polysemantic case.

The threshold sweep at the end pins down the squeeze condition of the paper:
the reconstruction is exact for any gate below the smallest true coefficient
and collapses once the gate rises above it.
"""

import torch

from sae_lens.analysis.atom_recovery import evaluate_atom_recovery
from sae_lens.saes.jumprelu_sae import JumpReLUSAE
from sae_lens.synthetic.activation_generator import ActivationGenerator
from sae_lens.synthetic.feature_dictionary import FeatureDictionary
from sae_lens.synthetic.initialization import init_sae_to_match_feature_dict
from tests.helpers import build_jumprelu_sae_cfg

NUM_FEATURES = 8
HIDDEN_DIM = 8
NUM_SAMPLES = 4096
# firing magnitudes are deterministic, so every true coefficient is exactly 1.0
COEFFICIENT_MAGNITUDE = 1.0
JUMPRELU_THRESHOLD = 0.1


def build_synthetic_setup() -> tuple[FeatureDictionary, ActivationGenerator]:
    feature_dict = FeatureDictionary(
        num_features=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
        initializer=None,
    )
    orthonormal, _ = torch.linalg.qr(torch.randn(NUM_FEATURES, HIDDEN_DIM))
    feature_dict.feature_vectors.data = orthonormal
    generator = ActivationGenerator(
        num_features=NUM_FEATURES,
        firing_probabilities=0.5,
        mean_firing_magnitudes=COEFFICIENT_MAGNITUDE,
        std_firing_magnitudes=0.0,
    )
    return feature_dict, generator


def build_jumprelu_sae(num_latents: int) -> JumpReLUSAE:
    sae = JumpReLUSAE(build_jumprelu_sae_cfg(d_in=HIDDEN_DIM, d_sae=num_latents))
    with torch.no_grad():
        sae.threshold.fill_(JUMPRELU_THRESHOLD)
    return sae


def superposed_dictionary(feature_vectors: torch.Tensor) -> torch.Tensor:
    mixtures = []
    for parent_a, parent_b in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        mixture = feature_vectors[parent_a] + feature_vectors[parent_b]
        mixtures.append(mixture / mixture.norm())
    return torch.cat([feature_vectors, torch.stack(mixtures)])


def test_typical_atom_recovery_workflow():
    feature_dict, generator = build_synthetic_setup()
    hidden_acts = feature_dict(generator.sample(NUM_SAMPLES))

    sae = build_jumprelu_sae(NUM_FEATURES)
    init_sae_to_match_feature_dict(sae, feature_dict)

    true_atoms = evaluate_atom_recovery(sae, hidden_acts)
    assert true_atoms.fidelity > 0.99
    assert true_atoms.stability > 0.99
    assert true_atoms.quantile_sparsity == NUM_FEATURES

    superposed = build_jumprelu_sae(NUM_FEATURES + NUM_FEATURES // 2)
    with torch.no_grad():
        superposed.W_dec.data = superposed_dictionary(sae.W_dec.data)
        superposed.W_enc.data = superposed.W_dec.data.T.clone()

    superposed_metrics = evaluate_atom_recovery(superposed, hidden_acts)
    assert superposed_metrics.fidelity > 0.99
    assert superposed_metrics.stability < true_atoms.stability

    mixtures_only = build_jumprelu_sae(NUM_FEATURES // 2)
    with torch.no_grad():
        mixtures_only.W_dec.data = superposed_dictionary(sae.W_dec.data)[NUM_FEATURES:]
        mixtures_only.W_enc.data = mixtures_only.W_dec.data.T.clone()

    mixtures_metrics = evaluate_atom_recovery(mixtures_only, hidden_acts)
    assert mixtures_metrics.stability > 0.99
    assert mixtures_metrics.fidelity < 0.6


def test_typical_atom_recovery_threshold_sweep_workflow():
    feature_dict, generator = build_synthetic_setup()
    hidden_acts = feature_dict(generator.sample(NUM_SAMPLES))

    sae = build_jumprelu_sae(NUM_FEATURES)
    init_sae_to_match_feature_dict(sae, feature_dict)

    below_smallest_coefficient = [
        evaluate_atom_recovery(sae, hidden_acts, threshold=gate).fidelity
        for gate in (0.1, 0.5, 0.99)
    ]
    assert all(
        fidelity > 0.99 for fidelity in below_smallest_coefficient
    ), below_smallest_coefficient

    above_smallest_coefficient = evaluate_atom_recovery(
        sae, hidden_acts, threshold=1.5
    ).fidelity
    assert above_smallest_coefficient < 0.0, above_smallest_coefficient
