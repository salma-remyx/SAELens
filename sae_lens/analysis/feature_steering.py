"""Steering with SAE features selected by contrasting groups of parallel texts.

Implements the inference-time steering method of "Strengthening Target-Language
Features: SAE-Based Steering for Multilingual Inference" (arXiv:2608.04904).
SAE activations are compared across groups of parallel texts (e.g. the same
sentences in several languages), the few features whose mean activation most
distinguishes a target group from the others are decoded into a steering
vector through the SAE decoder, and the vector is added to the model's hidden
states at inference time. No parameters are updated.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from transformer_lens.hook_points import HookPoint

if TYPE_CHECKING:
    from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
    from sae_lens.saes.sae import SAE

SteeringHook = Callable[[torch.Tensor, HookPoint], torch.Tensor]


@dataclass
class ContrastiveFeatures:
    """SAE features that distinguish one group of texts from the rest.

    Attributes:
        feature_indices: (num_features,) indices of the selected SAE features,
            ranked by the size of their activation contrast.
        contrasts: (num_features,) signed difference between the target
            group's mean activation and the other groups' mean at each
            selected feature.
    """

    feature_indices: torch.Tensor
    contrasts: torch.Tensor


@dataclass
class SteeringVector:
    """A residual-stream steering vector decoded from selected SAE features.

    Attributes:
        hook_name: name of the hook point the SAE operates on. Apply the
            steering hook at this name on a model without the SAE attached,
            or at ``hook_name + ".hook_sae_output"`` while the SAE is
            attached.
        vector: (d_in,) steering vector in model activation space.
        features: the selected features the vector was decoded from.
    """

    hook_name: str
    vector: torch.Tensor
    features: ContrastiveFeatures


def _mean_over_leading(activations: torch.Tensor) -> torch.Tensor:
    """Mean over every dimension except the trailing feature dimension."""
    return activations.mean(dim=tuple(range(activations.dim() - 1)))


def sentence_mean_activations(
    model: HookedSAETransformer,
    sae: SAE[Any],
    texts_by_group: Mapping[str, Sequence[str]],
) -> dict[str, torch.Tensor]:
    """Mean SAE feature activations per sentence for each group of texts.

    Each text is run through the model with the SAE attached (via
    ``run_with_cache_with_saes``) and its SAE activations are averaged over
    positions, giving one (n_features,) vector per sentence. All positions are
    included in the mean (BOS included); every group is tokenized the same
    way, so the shared special-token contribution cancels in the contrast.

    Args:
        model: model to run the texts through.
        sae: SAE to collect feature activations for.
        texts_by_group: texts per group, e.g. one entry per language holding
            that language's side of a set of parallel sentences.

    Returns:
        Mapping from group name to a (num_texts, n_features) tensor of
        per-sentence mean feature activations.
    """
    acts_hook = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"
    activations: dict[str, torch.Tensor] = {}
    for group, texts in texts_by_group.items():
        sentence_means = []
        for text in texts:
            _, cache = model.run_with_cache_with_saes(text, saes=[sae])
            sentence_means.append(cache[acts_hook].mean(dim=1)[0])
        activations[group] = torch.stack(sentence_means)
    return activations


def select_contrastive_features(
    activations_by_group: Mapping[str, torch.Tensor],
    target_group: str,
    num_features: int = 3,
) -> ContrastiveFeatures:
    """Select the features that most distinguish the target group.

    The mean activation per feature is computed for each group, the mean over
    the non-target groups forms a reference, and the features with the largest
    absolute target-minus-reference contrast are selected. Contrasts are kept
    signed, so features that are suppressed for the target group are selected
    too.

    Args:
        activations_by_group: mapping from group name to per-sentence mean
            feature activations of shape (..., n_features); leading dimensions
            are averaged over.
        target_group: the group to select features for.
        num_features: number of features to select. The paper selects three
            features per layer.

    Returns:
        The selected feature indices and their signed contrasts, ranked by
        absolute contrast.

    Raises:
        ValueError: If the target group is missing or no other group is
            available to contrast against.
    """
    if target_group not in activations_by_group:
        raise ValueError(
            f"Target group {target_group!r} not in groups: "
            f"{sorted(activations_by_group)}"
        )
    reference_groups = [
        group for group in activations_by_group if group != target_group
    ]
    if not reference_groups:
        raise ValueError(
            "At least one non-target group is required to contrast against."
        )
    target_mean = _mean_over_leading(activations_by_group[target_group])
    reference_mean = torch.stack(
        [_mean_over_leading(activations_by_group[group]) for group in reference_groups]
    ).mean(dim=0)
    contrast = target_mean - reference_mean
    num_selected = min(num_features, contrast.numel())
    feature_indices = contrast.abs().topk(num_selected).indices
    return ContrastiveFeatures(
        feature_indices=feature_indices,
        contrasts=contrast[feature_indices],
    )


def steering_vector_from_features(
    sae: SAE[Any],
    features: ContrastiveFeatures,
) -> SteeringVector:
    """Decode selected features into a steering vector.

    Each selected feature contributes its decoder direction (a row of
    ``sae.W_dec``) weighted by its signed contrast, following the paper's
    decoding of a sparse contrast code. The decoder bias is excluded: it is
    shared by every group's steering vector and would inject a constant
    offset unrelated to the target group.

    Args:
        sae: SAE the features were selected from.
        features: features selected for the target group.

    Returns:
        The steering vector, anchored at the SAE's hook point.
    """
    return SteeringVector(
        hook_name=sae.cfg.metadata.hook_name,
        vector=features.contrasts @ sae.W_dec[features.feature_indices],
        features=features,
    )


def steering_hook(
    steering_vector: SteeringVector,
    coefficient: float = 1.0,
) -> SteeringHook:
    """Create a forward hook that applies the steering vector.

    The hook adds ``coefficient * steering_vector.vector`` at the final
    position of the sequence it sees, the position that determines the first
    response token in the paper's intervention. Single-token sequences (the
    per-token steps of autoregressive generation) are left untouched, so the
    vector is applied once at the final prompt position instead of at every
    generated position.

    Use it at ``steering_vector.hook_name`` on a model without the SAE
    attached (``model.run_with_hooks``), or at
    ``steering_vector.hook_name + ".hook_sae_output"`` while the SAE is
    attached (``model.run_with_hooks_with_saes``).

    Args:
        steering_vector: the steering vector to apply.
        coefficient: strength of the intervention (alpha in the paper, which
            tunes it per target language and reports 0.6 as a shared value).

    Returns:
        A forward hook adding the steering vector at the final position.
    """

    def steer(hidden_states: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
        if hidden_states.shape[1] == 1:
            return hidden_states
        hidden_states[:, -1, :] += coefficient * steering_vector.vector
        return hidden_states

    return steer
