"""Feature-Effect Geometry Analysis (FEGA).

Removes a single SAE feature across many contexts and scores the geometry of
the resulting cloud of logit-change vectors, following "Sparse Autoencoders
Encode Both Concepts and Functions: The Downstream Geometry of Feature
Effects" (arXiv:2607.24645).

Each context contributes one effect vector: the change in final-position
logits between an ablated forward pass (the feature zeroed at
``hook_sae_acts_post``) and a reconstruction baseline (the SAE attached,
unmodified), so the two passes differ only in whether the feature is
retained. Geometry is scored through the cosine kernel of the unit effect
directions: directed-ray consistency, antipodal (axis) statistics, the
eigenvalue spectrum of the span, participation ratio and entropy rank, and
the centered residual energy. Consistent one-dimensional clouds indicate a
reusable steering direction; diffuse clouds indicate context-dependent
effects.

Adapted from the paper with two substitutions: the frozen-tail linear logit
readout is replaced by full forward passes through the attached SAE, and the
von Mises-Fisher mixture analysis is left out (ray, axis, and span
statistics already separate reusable directions from diffuse effects). The
paper's value-like / pointer-like populations come from task-specific
feature selection rather than geometry thresholds, so compare geometry
distributions across feature populations rather than thresholding a single
feature's scores.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformer_lens.hook_points import HookPoint

from sae_lens import logger
from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from sae_lens.saes.sae import SAE


def ablate_feature_hook(
    feature_idx: int,
) -> Callable[[torch.Tensor, HookPoint], torch.Tensor]:
    """Return a forward hook that zeroes one SAE feature.

    The hook acts on ``hook_sae_acts_post`` activations, removing the feature
    wherever it fires in the context, which is the feature-removal
    intervention FEGA scores.

    Args:
        feature_idx: Index of the feature to ablate.
    """

    def hook_fn(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
        ablated = acts.clone()
        ablated[..., feature_idx] = 0.0
        return ablated

    return hook_fn


def feature_effect_vectors(
    model: HookedSAETransformer,
    sae: SAE[Any],
    prompts: Sequence[str],
    feature_idx: int,
    *,
    max_contexts: int = 64,
) -> torch.Tensor:
    """Collect the cloud of logit-change vectors for one SAE feature.

    For every prompt, the baseline is a forward pass with the SAE attached
    (plain reconstruction, as in the paper) and the ablated pass adds a
    forward hook zeroing ``feature_idx`` at ``hook_sae_acts_post``, so the
    two passes differ only in whether the feature is retained. The effect
    vector is the difference of final-position logits. Both passes use the
    SAE's own ``use_error_term`` setting, so an SAE attached with the error
    term on measures effects relative to the model's own activations.

    Contexts whose effect is zero (the feature never fires) or non-finite
    are dropped, and at most ``max_contexts`` valid contexts are kept.

    Args:
        model: Model to run the interventions on.
        sae: SAE to attach during both passes.
        prompts: Prompts defining the contexts.
        feature_idx: Index of the feature to ablate.
        max_contexts: Maximum number of valid contexts to keep.

    Returns:
        Tensor of effect vectors with shape (num_valid_contexts, d_vocab).
    """
    output_hook = sae.cfg.metadata.hook_name_out or sae.cfg.metadata.hook_name
    sae_acts_post_hook = f"{output_hook}.hook_sae_acts_post"
    ablation_hook = ablate_feature_hook(feature_idx)
    d_vocab = model.cfg.d_vocab
    effects: list[torch.Tensor] = []
    with torch.no_grad():
        for prompt in prompts:
            tokens = model.to_tokens(prompt)
            baseline = model.run_with_saes(tokens, saes=sae)
            ablated = model.run_with_hooks_with_saes(
                tokens,
                saes=sae,
                fwd_hooks=[(sae_acts_post_hook, ablation_hook)],
            )
            assert isinstance(baseline, torch.Tensor)
            assert isinstance(ablated, torch.Tensor)
            effect = ablated[0, -1] - baseline[0, -1]
            d_vocab = effect.shape[-1]
            if torch.isfinite(effect).all() and effect.norm() > 0:
                effects.append(effect)
            if len(effects) >= max_contexts:
                break
    if not effects:
        return torch.empty((0, d_vocab))
    return torch.stack(effects)


@dataclass
class FeatureEffectGeometry:
    """Geometry of one feature's cloud of logit-change effect vectors.

    All spectral quantities come from the cosine kernel of the unit effect
    directions, so they are invariant to effect magnitude.

    Attributes:
        num_contexts: Number of valid (finite, non-zero) contexts scored.
        ray_consistency: Mean off-diagonal cosine between effect directions;
            1.0 means every context's effect points along one reusable
            directed ray.
        axis_eigenvalue_fraction: Share of kernel energy in the top
            eigenvector; high alongside low ray_consistency indicates
            antipodal (axis) effects rather than a directed ray.
        axis_split_fraction: Balance of positive and negative projections
            onto the top axis (0.0 = one-sided ray, 0.5 = perfectly
            antipodal).
        top2_span_variance: Share of kernel energy in the top two
            eigenvectors.
        participation_ratio: Inverse participation ratio 1 / sum(p**2) of the
            normalized eigenvalue spectrum; 1.0 for a one-dimensional cloud.
        entropy_rank: Exponential of the spectral entropy; sensitive to a
            long tail of weak directions.
        centered_residual_energy: Kernel energy left after removing the mean
            direction; 0.0 for a perfect ray.
        mean_effect_norm: Mean L2 norm of the effect vectors.
        effect_norm_cv: Coefficient of variation of the effect norms.
        sufficient_contexts: Whether at least ``min_contexts`` contexts were
            scored.
    """

    num_contexts: int
    ray_consistency: float
    axis_eigenvalue_fraction: float
    axis_split_fraction: float
    top2_span_variance: float
    participation_ratio: float
    entropy_rank: float
    centered_residual_energy: float
    mean_effect_norm: float
    effect_norm_cv: float
    sufficient_contexts: bool


def _ray_consistency(kernel: torch.Tensor) -> float:
    """Mean off-diagonal entry of the cosine kernel of unit directions."""
    n = kernel.shape[0]
    if n < 2:
        return 1.0
    return ((kernel.sum() - kernel.trace()) / (n * (n - 1))).item()


def effect_geometry(
    effects: torch.Tensor,
    *,
    min_contexts: int = 8,
) -> FeatureEffectGeometry:
    """Score the geometry of a cloud of feature-effect vectors.

    Args:
        effects: Effect vectors with shape (num_contexts, d_vocab), e.g. the
            output of ``feature_effect_vectors``.
        min_contexts: Minimum number of contexts treated as sufficient
            evidence, following the paper.

    Returns:
        Scores summarizing the consistency and dimensionality of the cloud.
        A cloud with no contexts yields all-zero scores and
        ``sufficient_contexts=False``; a single context yields vacuous ray
        scores of 1.0.
    """
    n = effects.shape[0]
    if n == 0:
        logger.warning(
            "FEGA: no valid contexts (feature inactive or effects zero); "
            "geometry scores are all zero."
        )
        return FeatureEffectGeometry(
            num_contexts=0,
            ray_consistency=0.0,
            axis_eigenvalue_fraction=0.0,
            axis_split_fraction=0.0,
            top2_span_variance=0.0,
            participation_ratio=0.0,
            entropy_rank=0.0,
            centered_residual_energy=0.0,
            mean_effect_norm=0.0,
            effect_norm_cv=0.0,
            sufficient_contexts=False,
        )
    if n < min_contexts:
        logger.warning(
            f"FEGA: only {n} valid contexts (< {min_contexts}); "
            "geometry is weak evidence."
        )

    norms = effects.norm(dim=-1)
    unit = effects / norms.unsqueeze(-1)
    # float64 keeps the small trailing eigenvalues meaningful for the
    # spectral statistics below; the kernel is at most (n, n).
    kernel = (unit @ unit.T).double()
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    weights = eigenvalues / eigenvalues.sum()
    # eigh returns ascending order, so the last eigenvector is the top axis;
    # projections onto it have the sign of its entries, up to a global flip
    # that cannot change the split balance.
    top_axis = eigenvectors[:, -1]
    positive = (top_axis > 0).sum().item()
    negative = (top_axis < 0).sum().item()
    nonzero_weights = weights[weights > 0]
    spectral_entropy = -(nonzero_weights * nonzero_weights.log()).sum()
    kernel_trace = kernel.trace().item()
    return FeatureEffectGeometry(
        num_contexts=n,
        ray_consistency=_ray_consistency(kernel),
        axis_eigenvalue_fraction=weights[0].item(),
        axis_split_fraction=min(positive, negative) / n,
        top2_span_variance=weights[:2].sum().item() if n > 1 else 1.0,
        participation_ratio=(1.0 / (weights**2).sum()).item(),
        entropy_rank=torch.exp(spectral_entropy).item(),
        centered_residual_energy=1.0 - kernel.sum().item() / (n * kernel_trace),
        mean_effect_norm=norms.mean().item(),
        effect_norm_cv=(norms.std() / norms.mean()).item() if n > 1 else 0.0,
        sufficient_contexts=n >= min_contexts,
    )


def analyze_feature_effect(
    model: HookedSAETransformer,
    sae: SAE[Any],
    prompts: Sequence[str],
    feature_idx: int,
    *,
    max_contexts: int = 64,
    min_contexts: int = 8,
) -> FeatureEffectGeometry:
    """Ablate one feature across prompts and score its effect geometry.

    Single-call entry point for FEGA on one feature: collects the
    logit-change cloud with ``feature_effect_vectors``, then scores it with
    ``effect_geometry``.

    Args:
        model: Model to run the interventions on.
        sae: SAE to attach during both passes.
        prompts: Prompts defining the contexts.
        feature_idx: Index of the feature to ablate.
        max_contexts: Maximum number of valid contexts to keep.
        min_contexts: Minimum number of contexts treated as sufficient
            evidence.

    Returns:
        The geometry of the feature's effect cloud.
    """
    effects = feature_effect_vectors(
        model,
        sae,
        prompts,
        feature_idx,
        max_contexts=max_contexts,
    )
    return effect_geometry(effects, min_contexts=min_contexts)
