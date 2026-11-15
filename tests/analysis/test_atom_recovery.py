import math

import pytest
import torch

from sae_lens.analysis.atom_recovery import (
    atom_coherence,
    atom_fidelity,
    evaluate_atom_recovery,
    quantile_coherence,
    quantile_sparsity,
    recovery_stability,
)
from sae_lens.saes.jumprelu_sae import JumpReLUSAE
from sae_lens.synthetic.activation_generator import ActivationGenerator
from sae_lens.synthetic.feature_dictionary import FeatureDictionary
from tests.helpers import build_jumprelu_sae_cfg


def orthonormal_dictionary(num_atoms: int, d_in: int) -> torch.Tensor:
    dictionary, _ = torch.linalg.qr(torch.randn(num_atoms, d_in))
    return dictionary


def unit_vectors_at(angles: list[float]) -> torch.Tensor:
    return torch.stack(
        [torch.tensor([math.cos(angle), math.sin(angle)]) for angle in angles]
    )


def test_atom_coherence_is_zero_for_an_orthonormal_dictionary():
    coherence = atom_coherence(orthonormal_dictionary(8, 8))

    assert coherence.shape == (8, 8)
    assert coherence.diagonal().abs().max().item() == pytest.approx(0.0, abs=1e-6)
    assert coherence.abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_atom_coherence_flags_duplicated_and_parallel_atoms():
    dictionary = orthonormal_dictionary(8, 8)
    duplicated = dictionary.clone()
    duplicated[3] = dictionary[5]

    coherence = atom_coherence(duplicated)

    assert coherence[3, 5].item() == pytest.approx(1.0)
    others = coherence.clone()
    others[3, 5] = 0.0
    others[5, 3] = 0.0
    assert others.abs().max().item() == pytest.approx(0.0, abs=1e-6)

    parallel = dictionary[0].unsqueeze(0).repeat(4, 1)
    parallel_coherence = atom_coherence(parallel)
    off_diagonal = parallel_coherence[~torch.eye(4, dtype=torch.bool)]
    assert off_diagonal.min().item() == pytest.approx(1.0)


def test_quantile_coherence_is_the_lower_quantile_of_off_diagonal_entries():
    coherence = torch.tensor(
        [
            [0.0, 0.1, 0.9],
            [0.1, 0.0, 0.5],
            [0.9, 0.5, 0.0],
        ]
    )

    # off-diagonal entries sorted: [0.1, 0.1, 0.5, 0.5, 0.9, 0.9]
    assert quantile_coherence(coherence, 0.0) == pytest.approx(0.1)
    assert quantile_coherence(coherence, 0.33) == pytest.approx(0.1)
    assert quantile_coherence(coherence, 0.34) == pytest.approx(0.5)
    assert quantile_coherence(coherence, 1.0) == pytest.approx(0.9)


def test_quantile_sparsity_tracks_the_l0_distribution_and_never_reaches_zero():
    codes = torch.zeros(100, 4)
    codes[:40, 0] = 1.0
    codes[40:, :3] = 1.0

    assert quantile_sparsity(codes, 0.0) == 1
    assert quantile_sparsity(codes, 0.4) == 1
    assert quantile_sparsity(codes, 0.41) == 3
    assert quantile_sparsity(codes, 1.0) == 3
    # a sample where nothing fires is 0-sparse, but K_q is counted over the
    # positive integers, so it still contributes a sparsity of 1
    assert quantile_sparsity(torch.zeros(10, 4), 1.0) == 1


def test_recovery_stability_stops_where_coherence_breaks_uniqueness():
    # three atoms in 2 dimensions: the only incoherent pair is (0, 2), so 2 of
    # the 6 off-diagonal entries sit below the coherence of the other pairs
    dictionary = unit_vectors_at([0.0, 0.5, 1.0])

    two_sparse = torch.tensor([[1.0, 0.0, 1.0]] * 100)
    metrics = recovery_stability(dictionary, two_sparse)

    assert metrics.stability == pytest.approx(0.33)
    assert metrics.quantile_sparsity == 2
    assert metrics.quantile_coherence == pytest.approx(
        quantile_coherence(atom_coherence(dictionary), 0.33)
    )
    assert metrics.quantile_coherence < 1.0 / 3

    # 1-sparse codes are unique for any coherence below 1
    one_sparse = torch.tensor([[1.0, 0.0, 0.0]] * 100)
    assert recovery_stability(dictionary, one_sparse).stability == pytest.approx(1.0)

    # 3-sparse codes need coherence below 1/5, which even the best pair exceeds
    three_sparse = torch.ones(100, 3)
    assert recovery_stability(dictionary, three_sparse).stability == pytest.approx(0.0)


def test_atom_fidelity_is_exact_for_a_true_atom_dictionary():
    dictionary = orthonormal_dictionary(8, 8)
    codes = (torch.rand(4096, 8) < 0.5).float()
    samples = codes @ dictionary

    assert atom_fidelity(dictionary, samples) == pytest.approx(1.0, abs=1e-6)


def test_atom_fidelity_drops_when_the_dictionary_merges_or_misses_atoms():
    dictionary = orthonormal_dictionary(8, 8)
    codes = (torch.rand(8192, 8) < 0.5).float()
    samples = codes @ dictionary

    # two of the eight atoms are missing, so their share of the data is not
    # representable whatever the readout
    missing = dictionary.clone()
    missing[6] = dictionary[0]
    missing[7] = dictionary[1]
    assert atom_fidelity(missing, samples) == pytest.approx(0.5, abs=0.05)

    # one feature is split across two coherent atoms, so the readout splits its
    # coefficient across both and the gate drops it
    split = dictionary.clone()
    split[0] = dictionary[0] + 0.6 * dictionary[2]
    split[0] /= split[0].norm()
    split[7] = dictionary[0] - 0.6 * dictionary[2]
    split[7] /= split[7].norm()
    assert atom_fidelity(split, samples, threshold=0.5) < 0.9
    assert atom_fidelity(dictionary, samples, threshold=0.5) == pytest.approx(
        1.0, abs=1e-6
    )


def test_evaluate_atom_recovery_separates_faithful_from_stable_dictionaries():
    num_features, d_in = 8, 8
    feature_dict = FeatureDictionary(
        num_features=num_features, hidden_dim=d_in, initializer=None
    )
    feature_dict.feature_vectors.data = orthonormal_dictionary(num_features, d_in)
    mixtures = []
    for parent_a, parent_b in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        mixture = (
            feature_dict.feature_vectors[parent_a]
            + feature_dict.feature_vectors[parent_b]
        )
        mixtures.append(mixture / mixture.norm())

    generator = ActivationGenerator(num_features=num_features, firing_probabilities=0.5)
    hidden_acts = feature_dict(generator.sample(4096))

    def jumprelu_with_dictionary(dictionary: torch.Tensor) -> JumpReLUSAE:
        sae = JumpReLUSAE(build_jumprelu_sae_cfg(d_in=d_in, d_sae=dictionary.shape[0]))
        with torch.no_grad():
            sae.W_dec.data = dictionary.clone()
            sae.W_enc.data = dictionary.T.clone()
            sae.threshold.fill_(0.1)
        return sae

    true_atoms = evaluate_atom_recovery(
        jumprelu_with_dictionary(feature_dict.feature_vectors.data), hidden_acts
    )
    assert true_atoms.fidelity == pytest.approx(1.0, abs=1e-4)
    assert true_atoms.stability == pytest.approx(1.0)
    assert true_atoms.quantile_coherence == pytest.approx(0.0, abs=1e-6)
    assert true_atoms.quantile_sparsity == num_features

    # adding superposed atoms leaves the reconstruction exact but makes the
    # codes non-unique, which only the stability criterion catches
    superposed = evaluate_atom_recovery(
        jumprelu_with_dictionary(
            torch.cat([feature_dict.feature_vectors.data, torch.stack(mixtures)])
        ),
        hidden_acts,
    )
    assert superposed.fidelity == pytest.approx(1.0, abs=1e-4)
    assert 0.5 < superposed.stability < 0.95

    # keeping only the mixtures spans half the space: stable, but not faithful
    mixtures_only = evaluate_atom_recovery(
        jumprelu_with_dictionary(torch.stack(mixtures)), hidden_acts
    )
    assert mixtures_only.stability == pytest.approx(1.0)
    assert mixtures_only.fidelity == pytest.approx(0.5, abs=0.05)
