"""
Representational-atom stability metrics for SAE dictionaries.

"Representational Atoms in Neural Networks" (https://arxiv.org/abs/2611.14168)
defines fundamental representational units (FRUs) as dictionary atoms that are
jointly representable, sparse and separable, and scores a candidate atom set
along two criteria: fidelity (how well sparse codes over the atoms reconstruct
the representations) and stability (how much of the code distribution respects
the coherence bound that makes a sparse code uniquely decodable).

This module implements the dictionary-level half of that framework:

- `normalized_atom_inner_products`: the atomic inner product induced by
  S~ = (DD^T)^-1, which scores separability in the geometry the dictionary
  defines rather than the ambient hidden space.
- `AtomStabilityCalculator`: a streaming calculator for the quantile coherence
  mu_q, the quantile sparsity K_q, and the stability score q* found by
  bisecting over quantiles.
- `eval_sae_atom_stability`: mirrors `eval_sae_on_synthetic_data` for scoring
  an SAE decoder as a candidate FRU dictionary.

The fidelity criterion is already produced by `ExplainedVarianceCalculator`, so
it is deliberately not duplicated here.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import torch

from sae_lens.saes.sae import SAE
from sae_lens.synthetic.activation_generator import ActivationGenerator
from sae_lens.synthetic.feature_dictionary import FeatureDictionary
from sae_lens.training.activation_scaler import ActivationScaler

DEFAULT_MAX_COHERENCE_PAIRS = 10_000_000
"""Upper bound on atom pairs sampled for the coherence distribution."""

_QUANTILE_SEARCH_TOLERANCE = 1e-4


def normalized_atom_inner_products(sae_decoder: torch.Tensor) -> torch.Tensor:
    """
    Compute the normalized atomic inner product (NAIP) matrix of a dictionary.

    Treating the rows of `sae_decoder` as candidate atoms, the paper defines the
    atomic inner product through S~ = (DD^T)^-1, so NAIP[i, j] = d_i^T S~ d_j.
    That matrix is exactly the orthogonal projector onto the row space of D, so
    it is computed here with a QR factorization rather than by inverting DD^T.
    Unlike cosine similarity between decoder rows, the result is invariant to
    any invertible reparameterization of the hidden space, and it accounts for
    the whole dictionary at once: two atoms only get a large NAIP if they
    genuinely compete for the same direction once every other atom is accounted
    for.

    Args:
        sae_decoder: Decoder weights of shape (num_atoms, hidden_dim). Must be
            overcomplete (num_atoms >= hidden_dim), since an undercomplete
            dictionary makes DD^T singular and the atomic inner product
            undefined.

    Returns:
        Tensor of shape (num_atoms, num_atoms) with entries in [-1, 1].
    """
    atoms = sae_decoder.detach().to(dtype=torch.float64)
    num_atoms, hidden_dim = atoms.shape
    if hidden_dim > num_atoms:
        raise ValueError(
            f"Expected an overcomplete dictionary (num_atoms >= hidden_dim), got "
            f"{num_atoms} atoms in {hidden_dim} dimensions; DD^T is singular."
        )
    # atoms is D^T, so its Q factor spans the row space of D inside R^num_atoms.
    basis, _ = torch.linalg.qr(atoms)
    return basis @ basis.T


def _off_diagonal_coherence_values(
    naip: torch.Tensor, max_pairs: int | None
) -> torch.Tensor:
    """Absolute NAIP values over unordered pairs of distinct atoms."""
    rows, cols = torch.triu_indices(naip.shape[0], naip.shape[0], offset=1)
    values = naip[rows, cols].abs()
    if max_pairs is not None and values.numel() > max_pairs:
        # Deterministic thinning rather than sampling, so a score is reproducible.
        values = values[:: math.ceil(values.numel() / max_pairs)]
    return values


def _empirical_quantile(sorted_values: torch.Tensor, q: float) -> float:
    """
    Smallest observed value v with P(X <= v) >= q.

    This inverse-CDF form is the one the paper's quantile coherence and quantile
    sparsity are defined with, rather than an interpolated quantile.
    """
    if sorted_values.numel() == 0:
        raise ValueError("Cannot take a quantile of an empty set of values.")
    num_values = sorted_values.numel()
    index = min(math.ceil(q * num_values) - 1, num_values - 1)
    return sorted_values[max(index, 0)].item()


def _distinguishability_condition(mu: float, sparsity: float) -> bool:
    """Stability condition mu_q < 1 / (2 K_q - 1) (unique K-sparse recovery)."""
    return mu < 1.0 / (2.0 * sparsity - 1.0)


def _coherence_condition(mu: float, sparsity: float) -> bool:
    """Retention condition mu_q < 1 / (K_q - 1), vacuous while K_q <= 1."""
    return sparsity <= 1.0 or mu < 1.0 / (sparsity - 1.0)


def _largest_feasible_quantile(
    sorted_coherence: torch.Tensor,
    sorted_sparsity: torch.Tensor,
    condition: Callable[[float, float], bool],
) -> float:
    """
    Bisect for the largest quantile whose coherence and sparsity jointly hold.

    Both mu_q and K_q are non-decreasing in q, so the bound 1 / (2 K_q - 1) is
    non-increasing and the condition can only flip from satisfied to violated,
    which is what makes a bisection valid.
    """

    def holds_at(q: float) -> bool:
        mu = _empirical_quantile(sorted_coherence, q)
        sparsity = max(1.0, _empirical_quantile(sorted_sparsity, q))
        return condition(mu, sparsity)

    if holds_at(1.0):
        return 1.0
    lower, upper = 0.0, 1.0
    while upper - lower > _QUANTILE_SEARCH_TOLERANCE:
        mid = (lower + upper) / 2
        if holds_at(mid):
            lower = mid
        else:
            upper = mid
    return lower


@dataclass
class AtomStabilityMetrics:
    """Stability of an SAE dictionary viewed as a set of representational atoms."""

    stability: float
    """Largest quantile q with mu_q < 1 / (2 K_q - 1); 1.0 is ideal."""

    retention: float
    """Largest quantile q with mu_q < 1 / (K_q - 1); always >= stability."""

    coherence: float
    """Max absolute NAIP mu over pairs of distinct atoms (lower is better)."""

    mean_atom_sparsity: float
    """Mean L0 of the latent codes, i.e. how many atoms a code uses on average."""

    def to_log_dict(self, prefix: str = "") -> dict[str, float]:
        """Convert to a flat dictionary for logging (e.g. to wandb)."""
        return {f"{prefix}{f.name}": getattr(self, f.name) for f in fields(self)}


class AtomStabilityCalculator:
    """
    Streaming calculator for the stability of a decoder as an atom set.

    The coherence half of the score depends only on the decoder dictionary and is
    computed once at construction. `add_batch` accumulates the per-sample L0 of
    the latent codes, and `compute` bisects over quantiles for the largest one at
    which coherence and sparsity jointly meet the paper's bounds.

    Args:
        sae_decoder: Decoder weights of shape (num_latents, hidden_dim).
        max_pairs: Cap on the number of atom pairs used for the coherence
            distribution. Excess pairs are deterministically thinned. None
            disables the cap.
    """

    def __init__(
        self,
        sae_decoder: torch.Tensor,
        max_pairs: int | None = DEFAULT_MAX_COHERENCE_PAIRS,
    ) -> None:
        naip = normalized_atom_inner_products(sae_decoder)
        coherence = _off_diagonal_coherence_values(naip, max_pairs)
        self._sorted_coherence = torch.sort(coherence).values
        self._l0_values: list[torch.Tensor] = []

    def add_batch(self, sae_latents: torch.Tensor) -> None:
        """Add a batch of latent activations. Shape: (batch_size, num_latents)."""
        active = (sae_latents > 0).sum(dim=1)
        self._l0_values.append(active.to(device="cpu", dtype=torch.float64))

    def compute(self) -> AtomStabilityMetrics:
        """Return the stability metrics over all batches seen so far."""
        if not self._l0_values:
            raise ValueError("AtomStabilityCalculator saw no batches.")
        sorted_sparsity = torch.sort(torch.cat(self._l0_values)).values
        stability = _largest_feasible_quantile(
            self._sorted_coherence, sorted_sparsity, _distinguishability_condition
        )
        retention = _largest_feasible_quantile(
            self._sorted_coherence, sorted_sparsity, _coherence_condition
        )
        return AtomStabilityMetrics(
            stability=stability,
            retention=retention,
            coherence=float(self._sorted_coherence[-1]),
            mean_atom_sparsity=sorted_sparsity.mean().item(),
        )


@torch.no_grad()
def eval_sae_atom_stability(
    sae: SAE[Any],
    feature_dict: FeatureDictionary,
    activations_generator: ActivationGenerator,
    num_samples: int = 100_000,
    batch_size: int | None = None,
    activation_scaler: ActivationScaler | None = None,
) -> AtomStabilityMetrics:
    """
    Score an SAE's decoder as a candidate set of representational atoms.

    Mirrors `eval_sae_on_synthetic_data`: the decoder rows are the candidate
    atoms, the latent activations are their coefficients, and the samples come
    from the same synthetic generator so the two evaluations can be logged side
    by side.

    Args:
        sae: The SAE whose decoder is scored
        feature_dict: The feature dictionary used to generate activations
        activations_generator: Generator that produces feature activations
        num_samples: Number of samples to use for the code distribution
        batch_size: Batch size for processing. If None, processes all at once.
        activation_scaler: Optional scaler applied to activations before encoding.
            If None, no scaling is applied.

    Returns:
        AtomStabilityMetrics with the stability, retention, coherence and mean
        sparsity of the atom set
    """
    sae.eval()

    if batch_size is None:
        batch_size = num_samples

    if activation_scaler is None:
        activation_scaler = ActivationScaler()

    calculator = AtomStabilityCalculator(sae.W_dec)

    num_processed = 0
    while num_processed < num_samples:
        current_batch_size = min(batch_size, num_samples - num_processed)
        feature_acts = activations_generator.sample(current_batch_size)
        hidden_acts = feature_dict(feature_acts)
        sae_latents = sae.encode(activation_scaler.scale(hidden_acts))
        calculator.add_batch(sae_latents)
        num_processed += current_batch_size

    return calculator.compute()
