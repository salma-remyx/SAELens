import pytest
import torch

from sae_lens.analysis.atom_recovery import (
    atom_coherence_samples,
    compute_fidelity,
    compute_stability,
    estimate_atomic_metric,
    evaluate_atom_recovery,
)
from sae_lens.saes.jumprelu_sae import JumpReLUSAE
from tests.helpers import build_jumprelu_sae_cfg


def _orthonormal_atoms(num_atoms: int) -> torch.Tensor:
    return torch.linalg.qr(torch.randn(num_atoms, num_atoms, dtype=torch.float64)).Q


def _orthonormal_atoms_32(num_atoms: int) -> torch.Tensor:
    return _orthonormal_atoms(num_atoms).to(torch.float32)


def _sparse_coefficients(
    num_samples: int,
    num_features: int,
    firing_prob: float,
    min_magnitude: float,
    max_magnitude: float,
) -> torch.Tensor:
    firing = torch.rand(num_samples, num_features) < firing_prob
    magnitudes = min_magnitude + (max_magnitude - min_magnitude) * torch.rand(
        num_samples, num_features
    )
    return (firing * magnitudes).to(torch.float32)


def _build_threshold_sae(atoms: torch.Tensor, threshold: float) -> JumpReLUSAE:
    sae = JumpReLUSAE(build_jumprelu_sae_cfg(d_in=atoms.shape[1], d_sae=atoms.shape[0]))
    with torch.no_grad():
        sae.W_dec.copy_(atoms)
        sae.W_enc.copy_(torch.linalg.solve(atoms.T @ atoms, atoms.T))
        sae.b_enc.zero_()
        sae.b_dec.zero_()
        sae.threshold.fill_(threshold)
    return sae


def test_estimate_atomic_metric_inverts_second_moment():
    scales = torch.tensor([0.5, 1.0, 2.0, 4.0])
    activations = torch.randn(200_000, 4) * scales
    metric = estimate_atomic_metric(activations)
    expected = torch.diag(1.0 / scales**2)
    for i in range(4):
        for j in range(4):
            assert metric[i, j] == pytest.approx(expected[i, j], rel=0.05, abs=0.02)


def test_atom_coherence_samples_zero_for_orthonormal_atoms():
    atoms = _orthonormal_atoms(12)
    coherences = atom_coherence_samples(atoms, torch.eye(12, dtype=torch.float64))
    assert coherences.shape == (66,)
    assert coherences.max().item() < 1e-10


def test_atom_coherence_samples_one_for_identical_atoms():
    atoms = torch.randn(1, 8).repeat(200, 1)
    coherences = atom_coherence_samples(
        atoms, torch.eye(8, dtype=torch.float64), max_pairs=1_000
    )
    assert coherences.numel() <= 1_000
    assert coherences.mean().item() == pytest.approx(1.0, abs=1e-9)


def test_atom_coherence_samples_isolates_duplicated_atom():
    atoms = _orthonormal_atoms(12)
    atoms[11] = atoms[0]
    coherences = atom_coherence_samples(atoms, torch.eye(12, dtype=torch.float64))
    assert coherences.max().item() == pytest.approx(1.0, abs=1e-8)
    assert coherences.sum().item() == pytest.approx(1.0, abs=1e-8)


def test_compute_fidelity_is_one_for_exact_fru_dictionary():
    atoms = _orthonormal_atoms_32(16)
    coefficients = _sparse_coefficients(4_096, 16, 0.25, 0.5, 1.5)
    activations = coefficients @ atoms
    sae = _build_threshold_sae(atoms, threshold=0.1)
    assert compute_fidelity(sae, activations) == pytest.approx(1.0, abs=1e-5)


def test_compute_fidelity_is_negative_when_nothing_fires():
    atoms = _orthonormal_atoms_32(16)
    coefficients = _sparse_coefficients(4_096, 16, 0.75, 3.0, 4.0)
    activations = coefficients @ atoms
    sae = _build_threshold_sae(atoms, threshold=10.0)
    assert compute_fidelity(sae, activations) < -1.0


def test_compute_stability_is_maximal_for_distinguishable_dictionary():
    atoms = _orthonormal_atoms_32(16)
    activations = torch.randn(2_048, 16)
    sae = _build_threshold_sae(atoms, threshold=0.1)
    stability = compute_stability(
        sae, activations, metric=torch.eye(16, dtype=torch.float64)
    )
    assert stability == pytest.approx(0.999999)


def test_compute_stability_drops_when_atoms_are_duplicated():
    atoms = _orthonormal_atoms_32(8)
    duplicated = torch.cat([atoms, atoms], dim=0)
    activations = torch.randn(2_048, 8)
    sae = _build_threshold_sae(duplicated, threshold=0.1)
    stability = compute_stability(
        sae, activations, metric=torch.eye(8, dtype=torch.float64)
    )
    assert 0.85 < stability < 0.96


def test_evaluate_atom_recovery_on_exact_fru_dictionary():
    atoms = _orthonormal_atoms_32(24)
    coefficients = _sparse_coefficients(20_000, 24, 0.125, 0.5, 1.5)
    activations = coefficients @ atoms
    sae = _build_threshold_sae(atoms, threshold=0.1)
    metrics = evaluate_atom_recovery(sae, activations)
    assert metrics.fidelity == pytest.approx(1.0, abs=1e-4)
    assert metrics.stability > 0.99
    assert metrics.mean_l0 == pytest.approx(3.0, abs=0.05)
    assert 1.0 <= metrics.stability_sparsity <= 24.0
    assert metrics.stability_coherence < 1.0 / (2.0 * metrics.stability_sparsity - 1.0)
