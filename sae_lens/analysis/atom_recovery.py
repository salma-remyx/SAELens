"""
Fidelity and stability metrics for the atoms of a learned SAE dictionary.

A dictionary of decoder directions is a set of *atoms* (fundamental
representational units) for a collection of representations when every
representation is a sparse combination of the atoms and the atoms are
approximately orthogonal under the atomic inner product. Two criteria decide
whether a learned dictionary actually recovers them:

- Fidelity: the coefficient of determination of the reconstruction built from
  the atoms with the paper's threshold encoder, `W_enc = D^T (D D^T)^{-1}` for a
  dictionary D whose columns are the atoms, so a dictionary that really is an
  atom set reconstructs the data exactly (R^2 = 1) regardless of how good the
  SAE's own encoder is.
- Stability: the largest quantile level `q*` at which the quantile coherence of
  the dictionary still guarantees uniqueness of the sparse codes,
  `mu_q < 1 / (2 K_q - 1)`, where `K_q` is the quantile sparsity of the codes
  and `mu_q` the quantile coherence of the atoms. Coherent (duplicated,
  polysemantic) atoms or denser codes both push `q*` down.

Adapted from "Representational Atoms in Neural Networks"
(https://arxiv.org/abs/2611.14168), Definitions 5-6 and Theorem 11. Quantiles
are empirical lower quantiles (order statistics), so `q*` is reported on the
grid of quantile levels that is evaluated.
"""

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch

from sae_lens.saes.sae import SAE


def atom_coherence(dictionary: torch.Tensor) -> torch.Tensor:
    """
    Pairwise normalized atomic inner products (NAIP) of a dictionary.

    The atomic inner product is `<d_i, d_j>_(D D^T)^-1`, under which atoms share
    a common norm; its Gram matrix is the orthogonal projector onto the span of
    the dictionary expressed in atom coordinates, here computed through the SVD
    so that rank-deficient dictionaries are still defined. The returned matrix
    holds the absolute normalized inner products off-diagonal (the coherence the
    stability criterion thresholds on) and zeros on the diagonal.

    Args:
        dictionary: Decoder directions of shape (num_atoms, d_in).

    Returns:
        Tensor of shape (num_atoms, num_atoms). An orthonormal dictionary gives
        an all-zero matrix; duplicated atoms give 1 on their pair.
    """
    u, singular_values, _ = torch.linalg.svd(dictionary.detach(), full_matrices=False)
    tolerance = (
        singular_values.max()
        * max(dictionary.shape)
        * torch.finfo(singular_values.dtype).eps
    )
    u = u[:, singular_values > tolerance]
    gram = u @ u.T
    atom_norms = gram.diagonal().clamp(min=torch.finfo(gram.dtype).eps).sqrt()
    coherence = (gram / torch.outer(atom_norms, atom_norms)).abs()
    return coherence.fill_diagonal_(0.0)


def quantile_coherence(coherence: torch.Tensor, q: float) -> float:
    """
    Quantile coherence `mu_q` of a dictionary.

    `mu_q` is the smallest coherence that at least a fraction `q` of atom pairs
    stay below, i.e. the empirical lower quantile of the off-diagonal entries of
    the coherence matrix.

    Args:
        coherence: Matrix from `atom_coherence` of shape (num_atoms, num_atoms).
        q: Quantile level in [0, 1].

    Returns:
        The quantile coherence, 0.0 for a single-atom dictionary.
    """
    num_atoms = coherence.shape[0]
    not_diagonal = ~torch.eye(num_atoms, dtype=torch.bool, device=coherence.device)
    off_diagonal = coherence[not_diagonal]
    if off_diagonal.numel() == 0:
        return 0.0
    return float(_lower_quantile(off_diagonal, q))


def quantile_sparsity(codes: torch.Tensor, q: float) -> int:
    """
    Quantile sparsity `K_q` of a distribution of codes.

    `K_q` is the smallest number of active atoms that at least a fraction `q` of
    the samples stay below, i.e. the empirical lower quantile of the L0 norm of
    the codes, counted over the positive integers as in the paper so that
    samples where nothing fires still count as 1-sparse.

    Args:
        codes: Coefficient matrix of shape (num_samples, num_atoms).
        q: Quantile level in [0, 1].

    Returns:
        The quantile sparsity, at least 1.
    """
    return max(1, int(_lower_quantile((codes != 0).sum(dim=-1), q)))


@dataclass
class StabilityMetrics:
    """Stability of a dictionary as an atom set for a distribution of codes."""

    stability: float
    """Largest quantile level q at which the uniqueness criterion holds (q*)"""

    quantile_coherence: float
    """Coherence mu_q at that level"""

    quantile_sparsity: int
    """Sparsity K_q at that level"""


@dataclass
class AtomRecoveryMetrics:
    """Fidelity and stability of a learned dictionary as an atom set."""

    fidelity: float
    """R^2 of the atom reconstruction with the threshold encoder (1 = exact)"""

    stability: float
    """Largest quantile level q at which the uniqueness criterion holds (1 = ideal)"""

    quantile_coherence: float
    """Coherence mu_q of the dictionary at that level"""

    quantile_sparsity: int
    """Sparsity K_q of the codes at that level"""

    def to_log_dict(self, prefix: str = "") -> dict[str, float | int]:
        """Convert to a flat dictionary for logging (e.g. to wandb)."""
        return {
            f"{prefix}{field.name}": getattr(self, field.name)
            for field in fields(self)
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return asdict(self)


def recovery_stability(
    dictionary: torch.Tensor,
    codes: torch.Tensor,
    num_quantiles: int = 101,
) -> StabilityMetrics:
    """
    Stability `q* = sup{ q | mu_q < 1 / (2 K_q - 1) }` of a dictionary.

    The criterion is the quantile form of the classical coherence condition for
    uniqueness of K-sparse representations: below it, no two sparse codes
    explain the same representation, so the atoms the SAE reports are the atoms
    that generated the data. `mu_q` and `K_q` both grow with `q` while the bound
    shrinks, so the admissible levels form an interval that is scanned from 0
    upwards and cut at the first failure; the reported `q*` is the largest grid
    level that still holds, a lower bound on the supremum.

    Args:
        dictionary: Decoder directions of shape (num_atoms, d_in).
        codes: Coefficient matrix of shape (num_samples, num_atoms), e.g. the
            SAE's latent activations on the same data.
        num_quantiles: Number of quantile levels in [0, 1] to evaluate.

    Returns:
        StabilityMetrics with `q*` and the (mu_q, K_q) pair at that level.
    """
    coherence = atom_coherence(dictionary)
    levels = torch.linspace(0.0, 1.0, num_quantiles).tolist()

    metrics = StabilityMetrics(
        stability=0.0,
        quantile_coherence=quantile_coherence(coherence, levels[0]),
        quantile_sparsity=quantile_sparsity(codes, levels[0]),
    )
    for level in levels:
        mu_q = quantile_coherence(coherence, level)
        k_q = quantile_sparsity(codes, level)
        if mu_q >= 1.0 / (2 * k_q - 1):
            break
        metrics = StabilityMetrics(
            stability=level, quantile_coherence=mu_q, quantile_sparsity=k_q
        )
    return metrics


def atom_fidelity(
    dictionary: torch.Tensor,
    samples: torch.Tensor,
    threshold: float | torch.Tensor | None = None,
) -> float:
    """
    Fidelity (R^2) of a dictionary as an atom set for a collection of samples.

    Codes are read out with the paper's threshold encoder: the minimum-norm
    coefficients of each sample in the span of the dictionary, then gated by the
    bidirectional threshold activation, which zeroes every coefficient whose
    magnitude falls below the threshold and keeps the rest. A dictionary that
    really is an atom set of the data reconstructs it exactly (R^2 = 1); one
    that misses atoms cannot, whatever the SAE's own encoder scores, and one
    whose atoms split a feature into coherent halves loses it as soon as the
    gate is raised above the interference between them.

    Args:
        dictionary: Decoder directions of shape (num_atoms, d_in).
        samples: Representations of shape (num_samples, d_in).
        threshold: Gate magnitude, either a scalar or a per-atom tensor of shape
            (num_atoms,). If None, uses the smallest nonzero coefficient
            magnitude observed per atom, the empirical estimate of the paper's
            delta_min bound on the true coefficients (coefficients at the bound
            are kept, so the gate only removes readout below it).

    Returns:
        The coefficient of determination of the atom reconstruction.
    """
    codes = samples @ torch.linalg.pinv(dictionary.detach())
    if threshold is None:
        threshold = _minimum_coefficient_magnitudes(codes)
    gated = torch.where(codes.abs() < threshold, torch.zeros_like(codes), codes)
    residual = samples - gated @ dictionary
    centered = samples - samples.mean(dim=0)
    return float(1.0 - residual.pow(2).sum() / centered.pow(2).sum())


@torch.no_grad()
def evaluate_atom_recovery(
    sae: SAE[Any],
    hidden_acts: torch.Tensor,
    num_quantiles: int = 101,
    threshold: float | torch.Tensor | None = None,
) -> AtomRecoveryMetrics:
    """
    Evaluate how faithfully and stably an SAE's dictionary recovers the atoms.

    Fidelity is measured on the SAE's decoder directions alone, while stability
    is measured on those directions together with the codes the SAE actually
    produces on `hidden_acts`, so the two criteria can be read against each
    other: a dictionary can be faithful yet unstable when its atoms are too
    coherent for the sparsity of its own codes.

    Args:
        sae: The SAE to evaluate.
        hidden_acts: Representations of shape (num_samples, d_in).
        num_quantiles: Number of quantile levels for the stability scan.
        threshold: Gate magnitude for the fidelity reconstruction. If None, the
            per-atom minimum coefficient magnitude is estimated from the data.

    Returns:
        AtomRecoveryMetrics with fidelity, stability and the (mu_q, K_q) pair.
    """
    dictionary = sae.W_dec.detach()
    codes = sae.encode(hidden_acts).detach()
    stability = recovery_stability(dictionary, codes, num_quantiles=num_quantiles)
    fidelity = atom_fidelity(dictionary, hidden_acts, threshold=threshold)
    return AtomRecoveryMetrics(
        fidelity=fidelity,
        stability=stability.stability,
        quantile_coherence=stability.quantile_coherence,
        quantile_sparsity=stability.quantile_sparsity,
    )


def _lower_quantile(values: torch.Tensor, q: float) -> float:
    # The smallest order statistic that a fraction q of the samples stays below.
    ordered, _ = torch.sort(values.flatten())
    count = max(1, math.ceil(q * ordered.numel()))
    return ordered[count - 1].item()


def _minimum_coefficient_magnitudes(codes: torch.Tensor) -> torch.Tensor:
    # delta_min per atom: the smallest nonzero coefficient magnitude, so the
    # gate removes readout noise without touching a real coefficient. Atoms that
    # are never used get 0, which only gates exact zeros.
    magnitudes = codes.abs()
    nonzero = magnitudes.where(magnitudes > 0, torch.full_like(magnitudes, math.inf))
    minima = nonzero.min(dim=0).values
    return torch.where(torch.isinf(minima), torch.zeros_like(minima), minima)
