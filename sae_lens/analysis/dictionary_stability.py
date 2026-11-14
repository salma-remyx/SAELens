"""
Fidelity and stability of a learned SAE dictionary as a set of representational atoms.

"Representational Atoms in Neural Networks" (https://arxiv.org/abs/2611.14168)
defines fundamental representational units (FRUs) and two criteria for judging
whether a candidate unit set deserves the name. Both are computed here with the
decoder directions of an SAE as the candidate atoms and its latent activations as
their coefficients:

- Fidelity: how exactly the atoms reconstruct the representation they were trained
  on, measured as reconstruction R^2 (reusing ExplainedVarianceCalculator).
- Stability: whether the atom set still supports a unique sparse decomposition at
  the top of the activation distribution. With K_q the q-quantile of the
  per-sample number of active latents and mu_q the q-quantile of the absolute
  normalized coherence |<d_i / ||d_i||, d_j / ||d_j||>| between distinct atom
  pairs, the score is the largest quantile at which the distinguishability
  condition for K-sparse uniqueness still holds:

      q* = sup { q : mu_q < 1 / (2 K_q - 1) }

  q* near 1 means essentially every sample has a unique sparse decomposition in
  the learned dictionary; a q* well below 1 pinpoints the tail (denser-than-typical
  samples, or correlated atom pairs) where uniqueness is no longer guaranteed.
  mu_q is non-decreasing in q while 1 / (2 K_q - 1) is non-increasing, so the
  feasible set is an interval and q* is located by binary search.

The paper also defines a weaker retention condition mu_q < 1 / (K_q - 1); it is
deliberately not computed here.
"""

from dataclasses import dataclass, fields
from typing import Any

import torch

from sae_lens.evals import ExplainedVarianceCalculator
from sae_lens.saes.sae import SAE
from sae_lens.training.activation_scaler import ActivationScaler


@dataclass
class StabilityResult:
    """Quantile stability of a dictionary against an activation distribution."""

    score: float
    """Largest quantile q at which mu_q < 1 / (2 K_q - 1) holds (1.0 is ideal)."""

    sparsity_at_score: float
    """K_{q*}, the q*-quantile of the number of active latents per sample."""

    coherence_at_score: float
    """mu_{q*}, the q*-quantile of the absolute normalized atom-pair coherence."""

    incoherence_bound_at_score: float
    """1 / (2 K_{q*} - 1), the uniqueness bound the coherence must stay under."""

    mean_sparsity: float
    """Mean number of active latents per sample."""

    mean_coherence: float
    """Mean absolute normalized atom-pair coherence."""

    def to_log_dict(self, prefix: str = "") -> dict[str, float]:
        """Convert to a flat dictionary for logging (e.g. to wandb)."""
        return {f"{prefix}{f.name}": getattr(self, f.name) for f in fields(self)}


@dataclass
class RepresentationalAtomResult:
    """Fidelity and stability of an SAE's decoder atoms as candidate FRUs."""

    fidelity: float
    """Reconstruction R^2 on the evaluated activations (1.0 is ideal)."""

    stability: StabilityResult
    """Quantile stability of the decoder atoms as a candidate unit set."""

    def to_log_dict(self, prefix: str = "") -> dict[str, float]:
        """Convert to a flat dictionary for logging (e.g. to wandb)."""
        return {
            f"{prefix}fidelity": self.fidelity,
            **self.stability.to_log_dict(f"{prefix}stability/"),
        }


def atom_pair_coherences(
    decoder: torch.Tensor,
    num_pairs: int = 100_000,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample the absolute normalized coherence between distinct decoder atoms.

    Pairs are drawn uniformly at random (with replacement) over the
    num_atoms * (num_atoms - 1) ordered distinct pairs; coherence is symmetric,
    so this matches the distribution over unordered distinct pairs used by the
    paper, and never pairs an atom with itself.

    Args:
        decoder: Atom directions of shape (num_atoms, hidden_dim), e.g. sae.W_dec.
        num_pairs: Number of atom pairs to sample.
        generator: Optional generator for reproducible sampling.

    Returns:
        Tensor of shape (num_pairs,) with |<d_i / ||d_i||, d_j / ||d_j||>| in [0, 1].
    """
    atoms = decoder.detach().to(device="cpu", dtype=torch.float32)
    if atoms.shape[0] < 2:
        raise ValueError("decoder must contain at least two atoms to form a pair")
    atoms = atoms / atoms.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    num_atoms = atoms.shape[0]
    first = torch.randint(num_atoms, (num_pairs,), generator=generator)
    # a uniform offset in [1, num_atoms - 1] makes every non-self partner of
    # `first` equally likely, so pairs stay uniform over distinct atoms
    second = (
        first + torch.randint(1, num_atoms, (num_pairs,), generator=generator)
    ) % num_atoms
    return (atoms[first] * atoms[second]).sum(dim=-1).abs()


def quantile_stability(
    coherences: torch.Tensor,
    sparsity_levels: torch.Tensor,
    max_quantile: float = 0.999999,
) -> StabilityResult:
    """Compute the quantile stability score of a dictionary.

    Args:
        coherences: Atom-pair coherences, e.g. from atom_pair_coherences.
        sparsity_levels: Number of active latents per sample, shape (num_samples,).
        max_quantile: Upper end of the searched quantile interval.

    Returns:
        StabilityResult with the score and the quantiles attained at the score.
    """
    coherences = coherences.detach().flatten().to(device="cpu", dtype=torch.float64)
    sparsity_levels = sparsity_levels.detach().flatten().to(
        device="cpu", dtype=torch.float64
    )
    if coherences.numel() == 0 or sparsity_levels.numel() == 0:
        raise ValueError("coherences and sparsity_levels must both be non-empty")

    def satisfies(quantile: float) -> bool:
        sparsity = torch.quantile(sparsity_levels, quantile).item()
        coherence = torch.quantile(coherences, quantile).item()
        # samples activating no latents are clamped to the K = 1 bound of 1
        return coherence < 1.0 / max(2.0 * sparsity - 1.0, 1.0)

    # the feasible set is an interval starting at 0, so bisect for its right end
    lower, upper = 0.0, max_quantile
    score = 0.0
    if satisfies(lower):
        for _ in range(24):
            middle = (lower + upper) / 2
            if satisfies(middle):
                lower = middle
            else:
                upper = middle
        score = lower

    sparsity_at_score = torch.quantile(sparsity_levels, score).item()
    coherence_at_score = torch.quantile(coherences, score).item()
    return StabilityResult(
        score=score,
        sparsity_at_score=sparsity_at_score,
        coherence_at_score=coherence_at_score,
        incoherence_bound_at_score=1.0 / max(2.0 * sparsity_at_score - 1.0, 1.0),
        mean_sparsity=sparsity_levels.mean().item(),
        mean_coherence=coherences.mean().item(),
    )


@torch.no_grad()
def evaluate_representational_atoms(
    sae: SAE[Any],
    hidden_acts: torch.Tensor,
    num_atom_pairs: int = 100_000,
    eval_batch_size: int = 8192,
    activation_scaler: ActivationScaler | None = None,
) -> RepresentationalAtomResult:
    """Evaluate an SAE's decoder atoms as candidate fundamental representational units.

    Encodes and decodes hidden_acts with the SAE to measure fidelity (R^2), then
    scores the dictionary's quantile stability against the per-sample sparsity of
    the latent activations it produces.

    Args:
        sae: The SAE to evaluate.
        hidden_acts: Activations of shape (num_samples, d_in) in the SAE's input space.
        num_atom_pairs: Atom pairs sampled from sae.W_dec for the coherence quantiles.
        eval_batch_size: Number of samples encoded per batch.
        activation_scaler: Optional scaler applied to activations before encoding;
            its inverse is applied to the reconstruction before scoring.

    Returns:
        RepresentationalAtomResult with the fidelity and stability of the atoms.
    """
    sae.eval()
    if activation_scaler is None:
        activation_scaler = ActivationScaler()

    fidelity_calculator = ExplainedVarianceCalculator()
    sparsity_levels = []
    for start in range(0, hidden_acts.shape[0], eval_batch_size):
        batch = hidden_acts[start : start + eval_batch_size]
        latents = sae.encode(activation_scaler.scale(batch))
        reconstruction = activation_scaler.unscale(sae.decode(latents))
        fidelity_calculator.add_batch(reconstruction, batch)
        sparsity_levels.append((latents > 0).sum(dim=-1).to("cpu", torch.float64))

    stability = quantile_stability(
        atom_pair_coherences(sae.W_dec, num_atom_pairs),
        torch.cat(sparsity_levels),
    )
    return RepresentationalAtomResult(
        fidelity=fidelity_calculator.compute(),
        stability=stability,
    )
