import pytest
import torch

from sae_lens.synthetic import ActivationGenerator, FeatureDictionary
from sae_lens.synthetic.classical_baselines import (
    evaluate_classical_baselines,
    ica_atoms,
    kmeans_atoms,
    nmf_atoms,
    pca_atoms,
)
from sae_lens.synthetic.evals import mean_correlation_coefficient


def _orthonormal_modes(num_modes: int, hidden_dim: int) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(hidden_dim, num_modes))
    return q.T


def test_pca_atoms_recovers_orthogonal_modes_with_distinct_variances():
    num_modes = 12
    modes = _orthonormal_modes(num_modes, num_modes)
    # Distinct variance per mode keeps the eigenvalues non-degenerate, so the
    # eigenvectors are stable rather than free to rotate within eigenspaces.
    stds = torch.linspace(0.5, 2.0, num_modes)
    fires = (torch.rand(50_000, num_modes) < 0.4).float()
    feature_acts = fires * torch.rand(50_000, num_modes) * stds
    activations = feature_acts @ modes

    mcc = mean_correlation_coefficient(pca_atoms(activations, num_modes), modes)

    assert mcc == pytest.approx(1.0, abs=2e-3)


def test_ica_atoms_recovers_sparse_sources_mixed_by_orthogonal_matrix():
    num_sources = 8
    modes = _orthonormal_modes(num_sources, num_sources)
    sources = (torch.rand(60_000, num_sources) < 0.1).float()
    activations = sources @ modes

    mcc = mean_correlation_coefficient(ica_atoms(activations, num_sources), modes)

    assert mcc > 0.99


def test_kmeans_atoms_recovers_axis_aligned_cluster_directions():
    num_samples, hidden_dim, num_clusters = 30_000, 16, 6
    cluster_ids = torch.randint(0, num_clusters, (num_samples,))
    radii = torch.rand(num_samples) * 3 + 0.5
    one_hot = torch.nn.functional.one_hot(cluster_ids, hidden_dim).float()
    activations = radii.unsqueeze(1) * one_hot
    basis = torch.eye(num_clusters, hidden_dim)

    mcc = mean_correlation_coefficient(kmeans_atoms(activations, num_clusters), basis)

    assert mcc > 0.99


def test_nmf_atoms_recovers_separable_nonnegative_dictionary():
    num_modes, hidden_dim, num_samples = 8, 12, 30_000
    dictionary = torch.rand(num_modes, hidden_dim) * 2
    mode_ids = torch.randint(0, num_modes, (num_samples,))
    magnitudes = torch.rand(num_samples) * 3 + 0.5
    one_hot = torch.nn.functional.one_hot(mode_ids, num_modes).float()
    activations = magnitudes.unsqueeze(1) * one_hot @ dictionary
    ground_truth = dictionary / dictionary.norm(dim=1, keepdim=True)

    mcc = mean_correlation_coefficient(nmf_atoms(activations, num_modes), ground_truth)

    assert mcc > 0.99


def test_evaluate_classical_baselines_reports_all_methods():
    num_features, hidden_dim = 24, 8
    feature_dict = FeatureDictionary(
        num_features=num_features, hidden_dim=hidden_dim, initializer=None
    )
    activations_gen = ActivationGenerator(
        num_features=num_features, firing_probabilities=0.3
    )
    activations = feature_dict(activations_gen.sample(4096))

    scores = evaluate_classical_baselines(activations, feature_dict.feature_vectors)

    assert set(scores) == {"pca", "ica", "kmeans", "nmf"}
    assert all(0.0 <= score <= 1.0 for score in scores.values())


def test_pca_and_ica_atoms_reject_more_components_than_dimensions():
    activations = torch.randn(1000, 8)

    with pytest.raises(ValueError):
        pca_atoms(activations, 9)
    with pytest.raises(ValueError):
        ica_atoms(activations, 9)
