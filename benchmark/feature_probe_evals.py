"""Contrastive-probe evaluation of SAE features for concept discovery.

Deterministic evaluation core adapted from SAEScientist-Bench
(arXiv:2609.09113), which scores a candidate SAE feature for a target
concept along three axes:

- feature discovery: rank features by how much more they activate on
  positive probe texts than on control probe texts
- concept selectivity: AUROC of a single feature's probe activations at
  separating positive from control texts
- causal steering: change in a target-vs-baseline logit difference when
  the feature's decoder direction is added to the residual stream,
  relative to the unmodified model and to a norm-matched random
  direction control

The agentic probe-design loop from the paper is out of scope: probe
texts are provided by the caller, and no LLM judge is involved. The
steering hook targets residual-stream SAEs; head-dim hooks
(hook_q/hook_k/hook_v/hook_z) are not supported.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch

from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from sae_lens.saes.sae import SAE
from sae_lens.util import get_special_token_ids


def auroc(positive_scores: torch.Tensor, control_scores: torch.Tensor) -> float:
    """Area under the ROC curve for one feature's probe activations.

    Computed as the Mann-Whitney U statistic: the fraction of
    (positive, control) pairs where the positive activation exceeds the
    control one, with ties counted as half wins. 1.0 means every
    positive probe activates above every control probe; 0.5 is chance.

    Args:
        positive_scores: shape (n_positive_probes,)
        control_scores: shape (n_control_probes,)
    """
    diffs = positive_scores[:, None] - control_scores[None, :]
    wins = (diffs > 0).sum() + 0.5 * (diffs == 0).sum()
    n_pairs = positive_scores.numel() * control_scores.numel()
    return wins.item() / n_pairs


def contrastive_feature_scores(
    positive_acts: torch.Tensor, control_acts: torch.Tensor
) -> torch.Tensor:
    """Per-feature contrast score between two probe activation sets.

    The score is the gap in mean max-activation between the positive and
    control probes, i.e. how much more strongly a feature fires on one
    contrast than the other.

    Args:
        positive_acts: shape (n_positive_probes, d_sae)
        control_acts: shape (n_control_probes, d_sae)

    Returns:
        shape (d_sae,)
    """
    return positive_acts.mean(dim=0) - control_acts.mean(dim=0)


def feature_ranks(scores: torch.Tensor) -> torch.Tensor:
    """One-based rank of each feature when scores are sorted descending.

    Args:
        scores: shape (d_sae,)

    Returns:
        shape (d_sae,), where rank 1 is the highest-scoring feature
    """
    order = scores.argsort(descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device)
    return ranks


@torch.no_grad()
def feature_activations_from_cache(
    sae: SAE[Any],
    cache_acts: torch.Tensor,
    tokens: torch.Tensor,
    special_token_ids: Sequence[int] = (),
) -> torch.Tensor:
    """Max feature activation per probe from cached hook activations.

    Args:
        sae: SAE whose encoder produces the feature activations
        cache_acts: shape (n_probes, seq_len, d_in), activations at the
            SAE's hook point
        tokens: shape (n_probes, seq_len), token ids matching cache_acts
        special_token_ids: token ids excluded from the max (e.g. BOS)

    Returns:
        shape (n_probes, d_sae)
    """
    feature_acts = sae.encode(cache_acts)
    if special_token_ids:
        keep = ~torch.isin(
            tokens, torch.as_tensor(list(special_token_ids), device=tokens.device)
        )
        # a probe made entirely of special tokens falls back to using
        # every position rather than returning -inf
        keep = keep | ~keep.any(dim=-1, keepdim=True)
        feature_acts = feature_acts.masked_fill(~keep.unsqueeze(-1), float("-inf"))
    return feature_acts.max(dim=1).values


@torch.no_grad()
def probe_feature_activations(
    sae: SAE[Any],
    model: HookedSAETransformer,
    texts: Sequence[str],
    exclude_special_tokens: bool = True,
) -> torch.Tensor:
    """Feature activations for a set of probe texts.

    Each text is run through the model, the activation at the SAE's hook
    point is encoded, and the max activation per feature over (non
    special) sequence positions is kept.

    Args:
        sae: pretrained SAE attached to a residual-stream hook
        model: model the SAE was trained on
        texts: probe texts, one activation row per text
        exclude_special_tokens: drop special token positions (BOS, EOS,
            ...) from the max

    Returns:
        shape (n_texts, d_sae)
    """
    hook_name = sae.cfg.metadata.hook_name
    if hook_name is None:
        raise ValueError("sae.cfg.metadata.hook_name must be set to probe features")
    prepend_bos = sae.cfg.metadata.prepend_bos
    if prepend_bos is None:
        prepend_bos = True
    special_token_ids = (
        get_special_token_ids(model.tokenizer) if exclude_special_tokens else []
    )
    probe_acts = []
    for text in texts:
        tokens = model.to_tokens(text, prepend_bos=prepend_bos)
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        probe_acts.append(
            feature_activations_from_cache(
                sae, cache[hook_name], tokens, special_token_ids
            )
        )
    return torch.cat(probe_acts, dim=0)


def make_steering_hook(
    steering_vector: torch.Tensor, steering_strength: float
) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    """Forward hook adding steering_strength * steering_vector to activations."""

    def steering_hook(activations: torch.Tensor, hook: Any) -> torch.Tensor:  # noqa: ARG001
        return activations + steering_strength * steering_vector.to(
            device=activations.device, dtype=activations.dtype
        )

    return steering_hook


def logit_diff(
    logits: torch.Tensor, target_token_id: int, baseline_token_id: int
) -> float:
    """logit(target) - logit(baseline) at the final sequence position.

    Args:
        logits: shape (batch_size, seq_len, d_vocab); averaged over the
            batch
        target_token_id: token the concept should promote
        baseline_token_id: token to compare against
    """
    final_logits = logits[:, -1, :]
    return (
        final_logits[:, target_token_id] - final_logits[:, baseline_token_id]
    ).mean().item()


def random_control_directions(
    reference_direction: torch.Tensor,
    n_directions: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random directions with norms matched to a reference direction.

    Used as the control condition for causal steering: a random
    direction of the same norm should move the logits far less than the
    feature's decoder direction if the feature causally encodes the
    concept.

    Args:
        reference_direction: shape (d_in,)
        n_directions: number of control directions to sample
        generator: optional torch.Generator; must live on the same
            device as reference_direction

    Returns:
        shape (n_directions, d_in)
    """
    noise = torch.randn(
        n_directions,
        reference_direction.numel(),
        generator=generator,
        device=reference_direction.device,
        dtype=reference_direction.dtype,
    )
    noise = noise / noise.norm(dim=-1, keepdim=True)
    return noise * reference_direction.norm()


@torch.no_grad()
def causal_steering_gain(
    sae: SAE[Any],
    model: HookedSAETransformer,
    tokens: torch.Tensor,
    feature_index: int,
    target_token_id: int,
    baseline_token_id: int,
    steering_strength: float,
    n_random_controls: int = 10,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Causal effect of steering with a feature's decoder direction.

    Compares the target-vs-baseline logit difference at the final
    position between the unmodified model, the model with
    steering_strength * W_dec[feature_index] added at the SAE's hook
    point, and the model steered with norm-matched random directions
    instead.

    Args:
        sae: SAE providing the feature's decoder direction
        model: model to steer
        tokens: shape (batch_size, seq_len) prompt tokens to steer on
        feature_index: index of the feature being evaluated
        target_token_id: token the concept should promote
        baseline_token_id: token to compare against
        steering_strength: multiplier on the (unit-norm) decoder
            direction
        n_random_controls: number of random control directions
        generator: optional torch.Generator for the random controls

    Returns:
        steering_gain is the feature effect minus the unmodified model,
        random_control_gain the mean random-direction effect, and
        steering_gain_over_control their difference.
    """
    hook_name = sae.cfg.metadata.hook_name
    if hook_name is None:
        raise ValueError("sae.cfg.metadata.hook_name must be set to steer features")
    direction = sae.W_dec[feature_index]

    def run_logit_diff(steering_vector: torch.Tensor | None) -> float:
        fwd_hooks: list[tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = []
        if steering_vector is not None:
            fwd_hooks.append(
                (hook_name, make_steering_hook(steering_vector, steering_strength))
            )
        logits = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=fwd_hooks)
        return logit_diff(logits, target_token_id, baseline_token_id)

    baseline = run_logit_diff(None)
    steered = run_logit_diff(direction)
    control_diffs = [
        run_logit_diff(control)
        for control in random_control_directions(
            direction, n_random_controls, generator=generator
        )
    ]
    control_gain = torch.tensor(control_diffs).mean().item()

    return {
        "baseline_logit_diff": baseline,
        "steered_logit_diff": steered,
        "steering_gain": steered - baseline,
        "random_control_gain": control_gain,
        "steering_gain_over_control": (steered - baseline) - control_gain,
    }


@torch.no_grad()
def run_feature_probe_evals(
    sae: SAE[Any],
    model: HookedSAETransformer,
    positive_texts: Sequence[str],
    control_texts: Sequence[str],
    feature_index: int,
    target_token_id: int,
    baseline_token_id: int,
    steering_strength: float = 10.0,
    n_random_controls: int = 10,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Full contrastive-probe evaluation of one candidate feature.

    Combines the three SAEScientist-Bench evaluation axes for a single
    feature: its discovery rank among all features when ranked by the
    positive-vs-control activation gap, its concept selectivity AUROC on
    the probes, and its causal steering gain relative to the unmodified
    model and a norm-matched random direction control. Steering runs on
    the first positive probe text.
    """
    positive_acts = probe_feature_activations(sae, model, positive_texts)
    control_acts = probe_feature_activations(sae, model, control_texts)

    scores = contrastive_feature_scores(positive_acts, control_acts)
    ranks = feature_ranks(scores)

    prepend_bos = sae.cfg.metadata.prepend_bos
    steering_tokens = model.to_tokens(
        positive_texts[0],
        prepend_bos=True if prepend_bos is None else prepend_bos,
    )

    metrics: dict[str, Any] = {
        "feature_index": feature_index,
        "discovery_rank": ranks[feature_index].item(),
        "total_features": scores.numel(),
        "selectivity_auroc": auroc(
            positive_acts[:, feature_index], control_acts[:, feature_index]
        ),
    }
    metrics.update(
        causal_steering_gain(
            sae,
            model,
            steering_tokens,
            feature_index,
            target_token_id,
            baseline_token_id,
            steering_strength,
            n_random_controls,
            generator=generator,
        )
    )
    return metrics
