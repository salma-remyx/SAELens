"""Fidelity and stability evaluation of SAE features as representational atoms.

Adapted from "Representational Atoms in Neural Networks" (arXiv:2611.14168),
which defines fundamental representational units (FRUs) and two criteria for
judging whether a dictionary of candidate units (here, the SAE decoder)
recovers them:

- Fidelity: how well the candidate units reconstruct the original
  representations, measured as the R^2 of the SAE reconstruction.
- Stability: whether the candidate dictionary is distinguishable enough that
  sparse decompositions in it are unique, measured as the largest quantile
  level q* such that mu_q < 1 / (2 * K_q - 1), where mu_q is the q-quantile of
  pairwise normalized atomic coherences and K_q is the q-quantile of
  per-sample L0 sparsity.

Coherences are computed under the normalized atomic inner product, whose
metric is estimated from the second-order statistics of the representations
themselves: S~ = (E[x x^T])^-1.

The entry points follow the same contract as the metric buckets in
``sae_lens.evals.run_evals`` (an SAE plus a batch of activations in, scores
out), with threshold-activated SAEs such as ``JumpReLUSAE`` as the intended
input.
"""

from dataclasses import dataclass
from typing import Any

import torch

from sae_lens.saes.sae import SAE

# Upper bound of the quantile search for the stability criterion.
_MAX_QUANTILE = 0.999999


@dataclass(frozen=True)
class AtomRecoveryMetrics:
    """Fidelity and stability of an SAE's decoder as candidate FRUs.

    Attributes:

        fidelity: R^2 of the SAE reconstruction of the evaluated activations.
        stability: Largest quantile level q* satisfying the paper's
            distinguishability condition mu_q < 1 / (2 * K_q - 1).
        stability_coherence: Coherence quantile mu_{q*} at the reported
            stability level.
        stability_sparsity: Sparsity quantile K_{q*} at the reported stability
            level.
        mean_l0: Mean number of active features per sample.
    """

    fidelity: float
    stability: float
    stability_coherence: float
    stability_sparsity: float
    mean_l0: float


def estimate_atomic_metric(
    activations: torch.Tensor, ridge: float = 0.0
) -> torch.Tensor:
    """Estimate the normalized atomic inner product metric from representations.

    The metric is S~ = (E[x x^T] + ridge * I)^-1, estimated from the
    second-order statistics of the provided activations. The estimate
    requires at least ``d_in`` linearly independent samples; ``ridge`` can be
    used to regularize the second-moment matrix when it is near-singular.

    Args:

        activations: Tensor of shape (n_samples, d_in).
        ridge: Optional diagonal regularization added to the second-moment
            matrix before inversion.

    Returns:

        Tensor of shape (d_in, d_in) containing S~ in float64.
    """
    acts = activations.detach().to(torch.float64)
    second_moment = acts.T @ acts / acts.shape[0]
    if ridge > 0.0:
        second_moment = second_moment + ridge * torch.eye(
            acts.shape[1], dtype=torch.float64, device=acts.device
        )
    return torch.linalg.inv(second_moment)


def atom_coherence_samples(
    decoder_weights: torch.Tensor,
    metric: torch.Tensor,
    max_pairs: int = 100_000,
) -> torch.Tensor:
    """Sample absolute normalized atomic coherences between distinct atoms.

    Each sample is |<d_i~, d_j~>_S~| for an unordered pair of distinct decoder
    atoms, normalized so that identical atoms have coherence 1. All pairs are
    enumerated when there are at most ``max_pairs`` of them; otherwise
    ``max_pairs`` pairs are sampled uniformly at random, following the
    paper's procedure for large dictionaries.

    Args:

        decoder_weights: Tensor of shape (d_sae, d_in) whose rows are atoms.
        metric: Normalized atomic inner product metric of shape (d_in, d_in),
            e.g. from ``estimate_atomic_metric``.
        max_pairs: Maximum number of atom pairs to evaluate.

    Returns:

        Tensor of shape (num_pairs,) of nonnegative coherence samples in
        float64.
    """
    decoder = decoder_weights.detach().to(torch.float64)
    weighted = decoder @ metric
    squared_norms = torch.einsum("fi,fi->f", decoder, weighted)
    # Drop zero atoms, which have no well-defined normalized direction.
    keep_mask = squared_norms > 0
    decoder = decoder[keep_mask]
    weighted = weighted[keep_mask]
    squared_norms = squared_norms[keep_mask]
    num_atoms = decoder.shape[0]
    num_pairs = num_atoms * (num_atoms - 1) // 2
    if num_pairs <= max_pairs:
        gram = weighted @ decoder.T
        rows, cols = torch.triu_indices(num_atoms, num_atoms, offset=1)
        coherences = gram[rows, cols] / torch.sqrt(
            squared_norms[rows] * squared_norms[cols]
        )
    else:
        pairs = torch.randint(0, num_atoms, (max_pairs, 2), device=decoder.device)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        coherences = torch.einsum(
            "fi,fi->f", weighted[pairs[:, 0]], decoder[pairs[:, 1]]
        ) / torch.sqrt(squared_norms[pairs[:, 0]] * squared_norms[pairs[:, 1]])
    return coherences.abs()


def per_sample_sparsity(sae: SAE[Any], activations: torch.Tensor) -> torch.Tensor:
    """Count active (nonzero) SAE features per sample.

    For threshold-activated SAEs this is the L0 sparsity of the coefficients
    under the SAE's own thresholds.

    Args:

        sae: SAE used to encode the activations.
        activations: Tensor of shape (batch_size, d_in) on the SAE's device.

    Returns:

        Integer tensor of shape (batch_size,) of per-sample feature counts.
    """
    with torch.no_grad():
        feature_acts = sae.encode(activations)
    return (feature_acts != 0).sum(dim=-1)


def compute_fidelity(sae: SAE[Any], activations: torch.Tensor) -> float:
    """Compute the paper's fidelity criterion: R^2 of the SAE reconstruction.

    Args:

        sae: SAE used to encode and reconstruct the activations.
        activations: Tensor of shape (batch_size, d_in) on the SAE's device.

    Returns:

        The R^2 of the reconstruction, where 1.0 is perfect fidelity and 0.0
        is no better than predicting the activation mean.
    """
    with torch.no_grad():
        recons = sae.decode(sae.encode(activations))
    return _r2_score(activations, recons)


def compute_stability(
    sae: SAE[Any],
    activations: torch.Tensor,
    metric: torch.Tensor | None = None,
    max_pairs: int = 100_000,
) -> float:
    """Compute the paper's stability criterion q* for the SAE's decoder.

    Args:

        sae: SAE whose decoder atoms are evaluated; threshold-activated SAEs
            (e.g. ``JumpReLUSAE``) are the intended input.
        activations: Tensor of shape (batch_size, d_in) on the SAE's device,
            used for the sparsity quantiles and, if no metric is passed, for
            estimating the atomic inner product metric.
        metric: Precomputed normalized atomic inner product metric; if None,
            estimated from ``activations``.
        max_pairs: Maximum number of atom pairs for the coherence quantiles.

    Returns:

        The largest quantile level q* in [0, 0.999999] such that
        mu_q < 1 / (2 * K_q - 1); 1.0-ish values indicate stable atoms.
    """
    if metric is None:
        metric = estimate_atomic_metric(activations)
    coherences = atom_coherence_samples(sae.W_dec, metric, max_pairs=max_pairs)
    l0_counts = per_sample_sparsity(sae, activations)
    return _search_stability_quantile(coherences, l0_counts)[0]


def evaluate_atom_recovery(
    sae: SAE[Any],
    activations: torch.Tensor,
    metric_activations: torch.Tensor | None = None,
    max_pairs: int = 100_000,
) -> AtomRecoveryMetrics:
    """Compute both FRU recovery criteria for an SAE over a batch of activations.

    Args:

        sae: SAE whose decoder is evaluated as candidate FRUs;
            threshold-activated SAEs (e.g. ``JumpReLUSAE``) are the intended
            input.
        activations: Tensor of shape (batch_size, d_in) on the SAE's device.
        metric_activations: Independently collected representations used to
            estimate the atomic inner product metric; defaults to
            ``activations``.
        max_pairs: Maximum number of atom pairs for the coherence quantiles.

    Returns:

        AtomRecoveryMetrics with the fidelity, stability, and their boundary
        values.
    """
    with torch.no_grad():
        feature_acts = sae.encode(activations)
        recons = sae.decode(feature_acts)
    l0_counts = (feature_acts != 0).sum(dim=-1)
    fidelity = _r2_score(activations, recons)
    metric = estimate_atomic_metric(
        activations if metric_activations is None else metric_activations
    )
    coherences = atom_coherence_samples(sae.W_dec, metric, max_pairs=max_pairs)
    stability, coherence_q, sparsity_q = _search_stability_quantile(
        coherences, l0_counts
    )
    return AtomRecoveryMetrics(
        fidelity=fidelity,
        stability=stability,
        stability_coherence=coherence_q,
        stability_sparsity=sparsity_q,
        mean_l0=l0_counts.to(torch.float64).mean().item(),
    )


def _r2_score(activations: torch.Tensor, recons: torch.Tensor) -> float:
    acts = activations.detach().to(torch.float64)
    recons = recons.detach().to(torch.float64)
    ss_res = (acts - recons).pow(2).sum()
    ss_tot = (acts - acts.mean(dim=0, keepdim=True)).pow(2).sum()
    return (1.0 - ss_res / ss_tot).item()


def _distinguishability(
    coherences: torch.Tensor, l0_counts: torch.Tensor, quantile: float
) -> tuple[float, float, bool]:
    coherence_q = torch.quantile(coherences, quantile).item()
    # The paper defines K_q over positive integers, so floor the sparsity
    # quantile at 1 (the condition is vacuous when only one atom is active).
    sparsity_q = max(torch.quantile(l0_counts.to(torch.float64), quantile).item(), 1.0)
    return coherence_q, sparsity_q, coherence_q < 1.0 / (2.0 * sparsity_q - 1.0)


def _search_stability_quantile(
    coherences: torch.Tensor, l0_counts: torch.Tensor
) -> tuple[float, float, float]:
    """Find the largest quantile satisfying mu_q < 1 / (2 * K_q - 1).

    Uses the paper's bisection procedure on [0, 0.999999]. Returns the
    reported quantile along with the coherence and sparsity quantiles at it.
    """
    if not _distinguishability(coherences, l0_counts, 0.0)[2]:
        q_star = 0.0
    elif _distinguishability(coherences, l0_counts, _MAX_QUANTILE)[2]:
        q_star = _MAX_QUANTILE
    else:
        low, high = 0.0, _MAX_QUANTILE
        for _ in range(60):
            mid = (low + high) / 2
            if _distinguishability(coherences, l0_counts, mid)[2]:
                low = mid
            else:
                high = mid
        q_star = low
    coherence_q, sparsity_q, _ = _distinguishability(coherences, l0_counts, q_star)
    return q_star, coherence_q, sparsity_q
