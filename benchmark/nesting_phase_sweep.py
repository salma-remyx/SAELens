"""
Sweep the recovery / merging / diffuse phase diagram of SAEs trained on
nested synthetic features.

Systematic feature co-occurrence is a known failure mode for feature
recovery: when a "nested" feature only ever fires alongside another
feature, an SAE may absorb both into a single latent (merging), or land in
a diffuse phase where reconstruction is good but no learned atom matches
any true feature. This module builds synthetic models with a controllable
nesting fraction, trains SAEs over grids of nesting fraction, sparsity
penalty and dictionary size via ``SyntheticSAERunner``, and measures which
phase each run lands in.

Adapted from the protocol of "A Dominant Diffuse Phase in the Sparse
Autoencoder Phase Diagram" (https://arxiv.org/abs/2609.10299), which
reports that SAEs trained on nested dictionaries converge to a diffuse
phase (median best cosine 0.5-0.7 against a 0.95 recovery criterion,
codes an order of magnitude denser than the ground truth) rather than to
the merged solution that minimizes the exact sparse-coding objective.

Run as a script:

    poetry run python -m benchmark.nesting_phase_sweep --num-features 16 32
"""

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Literal

import torch

from sae_lens.config import LoggingConfig
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig
from sae_lens.synthetic.activation_generator import (
    ActivationGenerator,
    ActivationsModifier,
)
from sae_lens.synthetic.evals import eval_sae_on_synthetic_data
from sae_lens.synthetic.feature_dictionary import FeatureDictionary
from sae_lens.synthetic.firing_probabilities import ConstantFiringProbabilityConfig
from sae_lens.synthetic.synthetic_model import (
    OrthogonalizationConfig,
    SyntheticModel,
    SyntheticModelConfig,
)
from sae_lens.synthetic.synthetic_sae_runner import (
    SyntheticSAERunner,
    SyntheticSAERunnerConfig,
)
from sae_lens.util import cosine_similarities

Phase = Literal["recovered", "merged", "diffuse"]


def nested_pairs(num_features: int, nesting_fraction: float) -> list[tuple[int, int]]:
    """
    Group features into (child, parent) pairs for nesting.

    The first ``2 * num_pairs`` features form pairs ``(2k, 2k + 1)`` where the
    child ``2k`` only fires when the parent ``2k + 1`` fires. Remaining
    features fire independently. ``nesting_fraction`` is the fraction of
    features participating in nested pairs, so 0.0 gives fully independent
    features and 1.0 nests every feature.

    Args:
        num_features: Total number of ground-truth features.
        nesting_fraction: Fraction of features to nest, in [0, 1].

    Returns:
        List of (child_index, parent_index) pairs.
    """
    if not 0.0 <= nesting_fraction <= 1.0:
        raise ValueError(f"nesting_fraction must be in [0, 1], got {nesting_fraction}")
    num_pairs = int(nesting_fraction * num_features) // 2
    return [(2 * k, 2 * k + 1) for k in range(num_pairs)]


def nested_pairs_modifier(pairs: list[tuple[int, int]]) -> ActivationsModifier:
    """
    Create an activation modifier that enforces nesting.

    The child of each pair is zeroed whenever its parent does not fire, so the
    child's firing support becomes a subset of the parent's (the same
    semantics as hierarchy parent deactivation, applied to flat pairs).

    Args:
        pairs: (child_index, parent_index) pairs from ``nested_pairs``.

    Returns:
        Modifier function suitable for ``ActivationGenerator``.
    """
    child_indices = torch.tensor([child for child, _ in pairs], dtype=torch.long)
    parent_indices = torch.tensor([parent for _, parent in pairs], dtype=torch.long)

    def modifier(activations: torch.Tensor) -> torch.Tensor:
        device = activations.device
        children = child_indices.to(device)
        parents = parent_indices.to(device)
        parents_firing = (activations[:, parents] > 0).to(activations.dtype)
        activations = activations.clone()
        activations[:, children] = activations[:, children] * parents_firing
        return activations

    return modifier


def build_nested_synthetic_model(
    num_features: int,
    hidden_dim: int,
    nesting_fraction: float,
    firing_probability: float = 0.3,
    seed: int | None = None,
) -> SyntheticModel:
    """
    Build a synthetic model whose features nest with the given fraction.

    Ground-truth feature vectors are (approximately) orthogonal unit vectors
    and all features share one firing probability, so any absorption observed
    downstream comes from the nesting structure alone.

    Args:
        num_features: Number of ground-truth features.
        hidden_dim: Dimensionality of the hidden activation space.
        nesting_fraction: Fraction of features arranged into nested pairs.
        firing_probability: Firing probability shared by all features.
        seed: Random seed for the feature dictionary.

    Returns:
        A ``SyntheticModel`` generating nested feature activations.
    """
    cfg = SyntheticModelConfig(
        num_features=num_features,
        hidden_dim=hidden_dim,
        firing_probability=ConstantFiringProbabilityConfig(
            probability=firing_probability
        ),
        orthogonalization=OrthogonalizationConfig(),
        bias=False,
        seed=seed,
    )
    feature_dict = FeatureDictionary(
        num_features=num_features,
        hidden_dim=hidden_dim,
        bias=False,
        seed=seed,
    )
    activation_generator = ActivationGenerator(
        num_features=num_features,
        firing_probabilities=firing_probability,
        modify_activations=nested_pairs_modifier(
            nested_pairs(num_features, nesting_fraction)
        ),
    )
    return SyntheticModel(
        cfg,
        feature_dict=feature_dict,
        activation_generator=activation_generator,
    )


@dataclass
class RecoveryMetrics:
    """How learned atoms recover, merge or miss the ground-truth features."""

    best_cosines: torch.Tensor
    """Absolute cosine of each ground-truth feature to its closest atom, shape (num_gt_features,)"""

    recovered_fraction: float
    """Fraction of ground-truth features with best cosine >= recovery threshold"""

    merged_fraction: float
    """Fraction of ground-truth features sharing their best atom with another feature"""

    median_best_cosine: float
    """Median best cosine across ground-truth features (the diffuse-phase signature)"""

    phase: Phase
    """One of "recovered", "merged" or "diffuse"."""


def compute_recovery_metrics(
    sae_decoder: torch.Tensor,
    gt_features: torch.Tensor,
    recovery_threshold: float = 0.95,
    merge_threshold: float = 0.6,
) -> RecoveryMetrics:
    """
    Measure recovery, merging and diffusion of learned atoms against ground truth.

    Each ground-truth feature is matched to its most similar learned atom by
    absolute cosine similarity. A feature is recovered when that similarity
    reaches ``recovery_threshold`` (the paper's 0.95 criterion). A feature is
    merged when its best atom is also the best atom of another feature and
    stays above ``merge_threshold`` for both: for (near-)orthogonal ground
    truth the canonical merged atom pointing between two features has cosine
    1/sqrt(2) ~ 0.707 to each, so 0.6 separates two-way merges from three-way
    splits (cos 1/sqrt(3) ~ 0.577).

    The phase is "recovered" when every feature is recovered and nothing is
    merged, "merged" when any feature is merged, and "diffuse" otherwise.

    Args:
        sae_decoder: Learned atoms of shape (num_atoms, hidden_dim).
        gt_features: Ground-truth feature vectors of shape (num_gt_features, hidden_dim).
        recovery_threshold: Best cosine at or above which a feature counts as recovered.
        merge_threshold: Best cosine at or above which a shared atom counts as a merge.

    Returns:
        RecoveryMetrics with per-feature best cosines and aggregate fractions.
    """
    cos = cosine_similarities(sae_decoder, gt_features).abs()  # (atoms, gt)
    best_cosines, best_atoms = cos.max(dim=0)  # (num_gt,)
    recovered = best_cosines >= recovery_threshold
    above_merge = best_cosines >= merge_threshold
    atom_load = torch.bincount(best_atoms[above_merge], minlength=cos.shape[0])
    merged = above_merge & (atom_load[best_atoms] > 1)

    recovered_fraction = recovered.float().mean().item()
    merged_fraction = merged.float().mean().item()
    if recovered.all() and merged_fraction == 0.0:
        phase: Phase = "recovered"
    elif merged_fraction > 0.0:
        phase = "merged"
    else:
        phase = "diffuse"

    return RecoveryMetrics(
        best_cosines=best_cosines,
        recovered_fraction=recovered_fraction,
        merged_fraction=merged_fraction,
        median_best_cosine=best_cosines.median().item(),
        phase=phase,
    )


@dataclass
class NestingSweepCell:
    """Phase-diagram measurements for one trained run."""

    num_features: int
    nesting_fraction: float
    l1_coefficient: float
    seed: int
    phase: Phase
    recovered_fraction: float
    merged_fraction: float
    median_best_cosine: float
    true_l0: float
    sae_l0: float
    density_ratio: float
    explained_variance: float
    mcc: float

    def to_dict(self) -> dict[str, float | int | str]:
        """Convert to a flat dictionary for logging or JSON output."""
        return asdict(self)


def run_single_cell(
    num_features: int,
    nesting_fraction: float,
    l1_coefficient: float,
    seed: int = 0,
    firing_probability: float = 0.3,
    d_sae: int | None = None,
    training_samples: int = 1_000_000,
    batch_size: int = 1024,
    lr: float = 3e-4,
    eval_samples: int = 100_000,
    device: str = "cpu",
    recovery_threshold: float = 0.95,
    merge_threshold: float = 0.6,
) -> NestingSweepCell:
    """
    Train one SAE on a nested dictionary and measure its phase.

    The SAE width defaults to the number of ground-truth features, so full
    recovery is always information-theoretically available and any
    merging/diffusion observed is a training outcome, not a capacity limit.

    Args:
        num_features: Number of ground-truth features (the paper's M).
        nesting_fraction: Fraction of features arranged into nested pairs (gamma).
        l1_coefficient: Sparsity penalty of the StandardSAE (the paper's lambda).
        seed: Seed for the ground-truth dictionary.
        firing_probability: Shared firing probability of all features.
        d_sae: Number of SAE latents. Defaults to ``num_features``.
        training_samples: Training activations to generate.
        batch_size: Batch size for training and evaluation.
        lr: Learning rate.
        eval_samples: Samples used for the final evaluation.
        device: Device to train on.
        recovery_threshold: Cosine at or above which a feature is recovered.
        merge_threshold: Cosine at or above which a shared atom is a merge.

    Returns:
        A NestingSweepCell summarizing the run.
    """
    model = build_nested_synthetic_model(
        num_features=num_features,
        hidden_dim=num_features,
        nesting_fraction=nesting_fraction,
        firing_probability=firing_probability,
        seed=seed,
    )
    if d_sae is None:
        d_sae = num_features

    runner_cfg = SyntheticSAERunnerConfig(
        synthetic_model=model.cfg,
        sae=StandardTrainingSAEConfig(
            d_in=num_features,
            d_sae=d_sae,
            l1_coefficient=l1_coefficient,
        ),
        training_samples=training_samples,
        batch_size=batch_size,
        lr=lr,
        device=device,
        checkpoint_path=None,
        output_path=None,
        eval_samples=0,
        run_final_eval=False,
        logger=LoggingConfig(log_to_wandb=False),
    )
    result = SyntheticSAERunner(runner_cfg, override_synthetic_model=model).run()
    sae = result.sae

    eval_result = eval_sae_on_synthetic_data(
        sae=sae,
        feature_dict=model.feature_dict,
        activations_generator=model.activation_generator,
        num_samples=eval_samples,
        batch_size=batch_size,
    )
    metrics = compute_recovery_metrics(
        sae_decoder=sae.W_dec,
        gt_features=model.feature_dict.feature_vectors,
        recovery_threshold=recovery_threshold,
        merge_threshold=merge_threshold,
    )
    density_ratio = (
        eval_result.sae_l0 / eval_result.true_l0
        if eval_result.true_l0 > 0
        else 0.0
    )

    return NestingSweepCell(
        num_features=num_features,
        nesting_fraction=nesting_fraction,
        l1_coefficient=l1_coefficient,
        seed=seed,
        phase=metrics.phase,
        recovered_fraction=metrics.recovered_fraction,
        merged_fraction=metrics.merged_fraction,
        median_best_cosine=metrics.median_best_cosine,
        true_l0=eval_result.true_l0,
        sae_l0=eval_result.sae_l0,
        density_ratio=density_ratio,
        explained_variance=eval_result.explained_variance,
        mcc=eval_result.mcc,
    )


@dataclass
class NestingSweepConfig:
    """Grid definition for a nesting phase sweep."""

    nesting_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    """Nesting fractions (gamma) to sweep"""

    l1_coefficients: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1)
    """Sparsity penalties (lambda) to sweep"""

    num_features_grid: tuple[int, ...] = (16, 32, 64)
    """Dictionary sizes (M) to sweep"""

    seeds: tuple[int, ...] = (0,)
    """Seeds for independently initialized fits per grid cell"""

    firing_probability: float = 0.3
    """Shared firing probability of all ground-truth features"""

    training_samples: int = 1_000_000
    """Training activations per run"""

    batch_size: int = 1024
    """Batch size for training and evaluation"""

    lr: float = 3e-4
    """Learning rate"""

    eval_samples: int = 100_000
    """Samples used for the final evaluation of each run"""

    device: str = "cpu"
    """Device to train on"""


def run_nesting_sweep(
    config: NestingSweepConfig,
) -> list[NestingSweepCell]:
    """
    Train and evaluate one SAE per (M, gamma, lambda, seed) grid cell.

    Args:
        config: Grid definition and shared training settings.

    Returns:
        One NestingSweepCell per grid cell, ordered by num_features, then
        nesting fraction, then l1 coefficient, then seed.
    """
    cells: list[NestingSweepCell] = []
    for num_features in config.num_features_grid:
        for nesting_fraction in config.nesting_fractions:
            for l1_coefficient in config.l1_coefficients:
                for seed in config.seeds:
                    cells.append(
                        run_single_cell(
                            num_features=num_features,
                            nesting_fraction=nesting_fraction,
                            l1_coefficient=l1_coefficient,
                            seed=seed,
                            firing_probability=config.firing_probability,
                            training_samples=config.training_samples,
                            batch_size=config.batch_size,
                            lr=config.lr,
                            eval_samples=config.eval_samples,
                            device=config.device,
                        )
                    )
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep the nesting phase diagram of SAEs on synthetic data."
    )
    parser.add_argument("--num-features", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument(
        "--nesting-fractions", type=float, nargs="+", default=[0.0, 0.5, 1.0]
    )
    parser.add_argument(
        "--l1-coefficients", type=float, nargs="+", default=[0.001, 0.01, 0.1]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--training-samples", type=int, default=1_000_000)
    parser.add_argument("--eval-samples", type=int, default=100_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = NestingSweepConfig(
        nesting_fractions=tuple(args.nesting_fractions),
        l1_coefficients=tuple(args.l1_coefficients),
        num_features_grid=tuple(args.num_features),
        seeds=tuple(args.seeds),
        training_samples=args.training_samples,
        eval_samples=args.eval_samples,
        device=args.device,
    )
    cells = run_nesting_sweep(config)

    for cell in cells:
        print(
            f"M={cell.num_features} gamma={cell.nesting_fraction:g} "
            f"lambda={cell.l1_coefficient:g} seed={cell.seed} "
            f"phase={cell.phase} recovered={cell.recovered_fraction:.3f} "
            f"merged={cell.merged_fraction:.3f} "
            f"median_cos={cell.median_best_cosine:.3f} "
            f"sae_l0={cell.sae_l0:.2f} true_l0={cell.true_l0:.2f} "
            f"density_ratio={cell.density_ratio:.2f} "
            f"explained_variance={cell.explained_variance:.3f} "
            f"mcc={cell.mcc:.3f}"
        )

    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump([cell.to_dict() for cell in cells], f, indent=2)
        print(f"Saved sweep results to {args.output}")


if __name__ == "__main__":
    main()
