import math

import pytest
import torch

from sae_lens.analysis.dictionary_stability import (
    atom_pair_coherences,
    evaluate_representational_atoms,
    quantile_stability,
)
from sae_lens.saes.jumprelu_sae import JumpReLUSAE
from tests.helpers import build_jumprelu_sae_cfg


def test_quantile_stability_matches_a_brute_force_quantile_scan() -> None:
    # cast to double so the expectations below see the same quantile arithmetic
    # as quantile_stability, which accumulates in float64
    coherences = torch.rand(20_000).double()
    sparsity_levels = (1 + (torch.rand(20_000) * 6).floor()).double()

    result = quantile_stability(coherences, sparsity_levels)

    grid = torch.linspace(0.0, 0.999999, 2_000)
    best = 0.0
    for quantile in grid.tolist():
        sparsity = torch.quantile(sparsity_levels, quantile).item()
        coherence = torch.quantile(coherences, quantile).item()
        if coherence < 1.0 / max(2.0 * sparsity - 1.0, 1.0):
            best = quantile

    assert result.score == pytest.approx(best, abs=1 / 2_000)
    assert result.sparsity_at_score == pytest.approx(
        torch.quantile(sparsity_levels, result.score).item()
    )
    assert result.coherence_at_score == pytest.approx(
        torch.quantile(coherences, result.score).item()
    )
    assert result.incoherence_bound_at_score == pytest.approx(
        1.0 / (2.0 * result.sparsity_at_score - 1.0)
    )


def test_quantile_stability_pinpoints_the_coherence_tail() -> None:
    # sparsity is fixed at 3, so the uniqueness bound is 1 / (2 * 3 - 1) = 0.2 for
    # every quantile; the coherence distribution jumps from 0.1 to 0.9 at its 90th
    # percentile, so the condition mu_q < 0.2 breaks just past q = 0.9
    coherences = torch.where(torch.arange(10_000) < 9_000, 0.1, 0.9).double()
    sparsity_levels = torch.full((10_000,), 3.0)

    result = quantile_stability(coherences, sparsity_levels)

    assert result.score == pytest.approx(0.9, abs=0.01)
    assert result.sparsity_at_score == pytest.approx(3.0)
    assert result.incoherence_bound_at_score == pytest.approx(0.2)
    assert result.mean_sparsity == pytest.approx(3.0)
    assert result.mean_coherence == pytest.approx(0.18)


def test_quantile_stability_is_maximal_for_incoherent_atoms() -> None:
    coherences = torch.zeros(10_000)
    sparsity_levels = torch.randint(1, 6, (10_000,)).double()

    result = quantile_stability(coherences, sparsity_levels)

    assert result.score == pytest.approx(0.999999, abs=1e-6)
    assert result.coherence_at_score == pytest.approx(0.0)


def test_atom_pair_coherences_match_the_analytic_mean_for_random_atoms() -> None:
    # two random unit vectors in hidden_dim dimensions have |cos| with mean
    # Gamma(hidden_dim / 2) / (sqrt(pi) Gamma((hidden_dim + 1) / 2)); a sampler
    # that ever pairs an atom with itself shifts the mean up by ~1 / num_atoms
    num_atoms, hidden_dim = 256, 64
    atoms = torch.randn(num_atoms, hidden_dim)

    coherences = atom_pair_coherences(atoms, num_pairs=200_000)

    expected_mean = math.exp(
        math.lgamma(hidden_dim / 2)
        - 0.5 * math.log(math.pi)
        - math.lgamma((hidden_dim + 1) / 2)
    )
    assert coherences.shape == (200_000,)
    assert coherences.min() >= 0.0
    assert coherences.max() <= 1.0
    assert coherences.mean().item() == pytest.approx(expected_mean, abs=2e-3)


def test_atom_pair_coherences_rejects_a_single_atom() -> None:
    with pytest.raises(ValueError):
        atom_pair_coherences(torch.randn(1, 8))


def _ground_truth_setup(
    num_atoms: int,
    hidden_dim: int,
    num_samples: int,
    true_l0: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an orthonormal atom set and K-sparse coefficients drawn from it."""
    atoms = torch.linalg.qr(torch.randn(hidden_dim, num_atoms).double()).Q.T.float()

    coefficients = torch.zeros(num_samples, num_atoms)
    chosen = torch.rand(num_samples, num_atoms).argsort(dim=-1)[:, :true_l0]
    magnitudes = torch.empty(num_samples, true_l0).uniform_(0.5, 2.0)
    coefficients.scatter_(1, chosen, magnitudes)

    hidden_acts = coefficients @ atoms
    return atoms, hidden_acts


def _jumprelu_sae_on_ground_truth(
    atoms: torch.Tensor, threshold: float
) -> JumpReLUSAE:
    sae = JumpReLUSAE(build_jumprelu_sae_cfg(d_in=atoms.shape[1], d_sae=atoms.shape[0]))
    with torch.no_grad():
        sae.W_enc.copy_(atoms.T)
        sae.W_dec.copy_(atoms)
        sae.b_enc.zero_()
        sae.b_dec.zero_()
        sae.threshold.fill_(threshold)
    return sae


def test_jumprelu_sae_recovering_the_atoms_scores_perfectly() -> None:
    atoms, hidden_acts = _ground_truth_setup(
        num_atoms=32, hidden_dim=64, num_samples=20_000, true_l0=3
    )
    # 0.1 sits inside the paper's admissible interval (tau < min coefficient 0.5)
    sae = _jumprelu_sae_on_ground_truth(atoms, threshold=0.1)

    result = evaluate_representational_atoms(sae, hidden_acts)

    assert result.fidelity == pytest.approx(1.0, abs=1e-4)
    assert result.stability.score == pytest.approx(0.999999, abs=1e-6)
    assert result.stability.sparsity_at_score == pytest.approx(3.0)
    assert result.stability.mean_sparsity == pytest.approx(3.0)
    assert result.stability.mean_coherence == pytest.approx(0.0, abs=1e-5)
    assert result.stability.coherence_at_score == pytest.approx(0.0, abs=1e-5)


def test_threshold_above_the_minimum_coefficient_costs_fidelity_and_sparsity() -> None:
    atoms, hidden_acts = _ground_truth_setup(
        num_atoms=32, hidden_dim=64, num_samples=20_000, true_l0=3
    )
    # 1.5 sits above many of the coefficients in [0.5, 2.0], so the gate drops
    # atoms the representation actually used
    sae = _jumprelu_sae_on_ground_truth(atoms, threshold=1.5)

    result = evaluate_representational_atoms(sae, hidden_acts)

    assert result.fidelity < 0.9
    assert result.stability.mean_sparsity < 2.0

    reference = evaluate_representational_atoms(
        _jumprelu_sae_on_ground_truth(atoms, threshold=0.1), hidden_acts
    )
    assert result.fidelity < reference.fidelity
    assert result.stability.mean_sparsity < reference.stability.mean_sparsity
