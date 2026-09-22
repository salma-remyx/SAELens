"""Contrastive projection: reading model internals by differencing logit lenses.

A logit lens applied to a single hidden state is dominated, at intermediate
layers, by the generic tokens the model would predict for almost any input.
Subtracting the hidden states of closely matched prompts (prompts differing
only in the concept of interest) and projecting the difference through the
unembedding cancels the shared component and surfaces what separates the two.
Projecting the same difference onto the decoder directions of an SAE
attributes the contrast to individual SAE features.

The projection skips the final layer norm, which is nonlinear and would break
the exact cancellation of the shared component: (h_a - h_b) @ W_U is exactly
the difference of the two raw logit lenses, making the operation equivalent to
reading a steering vector through a logit lens.

Adapted from "Contrastive Projection: Reading Transformer Internals by
Differencing Logit Lenses" (https://arxiv.org/abs/2609.09902).
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from sae_lens.saes.sae import SAE


def contrastive_logit_lens(
    hidden_contrast: torch.Tensor, w_unembed: torch.Tensor
) -> torch.Tensor:
    """Project hidden-state differences through the unembedding.

    Args:
        hidden_contrast: (..., d_model) differences of matched hidden states.
        w_unembed: (d_model, d_vocab) unembedding matrix (model.W_U).

    Returns:
        (..., d_vocab) contrast in token space. Positive entries are tokens the
        prompt boosts relative to the baselines, negative entries tokens it
        suppresses.
    """
    return hidden_contrast @ w_unembed


def contrastive_feature_attribution(
    hidden_contrast: torch.Tensor, w_dec: torch.Tensor
) -> torch.Tensor:
    """Attribute hidden-state differences to SAE features via decoder directions.

    Each feature scores the dot product of the contrast with its decoder
    direction, i.e. the contribution that feature's decoder makes to the
    contrast.

    Args:
        hidden_contrast: (..., d_in) differences of matched hidden states,
            flattened when the SAE hook has extra dimensions (e.g. heads).
        w_dec: (d_sae, d_in) SAE decoder directions.

    Returns:
        (..., d_sae) contrast scores per feature.
    """
    w_dec = w_dec.to(device=hidden_contrast.device, dtype=hidden_contrast.dtype)
    return hidden_contrast @ w_dec.T


@dataclass
class ContrastiveProjection:
    """Contrast between one prompt and the mean of its baselines, in model space.

    Attributes:
        prompt: The prompt of interest.
        baselines: The baseline prompts the contrast is taken against.
        contrast: Per hook, (n_positions, d_hook) hidden-state differences.
        token_scores: Per hook, (n_positions, d_vocab) contrast in token space.
        model: The model the contrast was traced on.
        feature_scores: (n_positions, d_sae) contrast attributed to SAE
            features, or None when no SAE was given.
        feature_hook: Hook the feature attribution was read at, or None.
    """

    prompt: str
    baselines: list[str]
    contrast: dict[str, torch.Tensor]
    token_scores: dict[str, torch.Tensor]
    model: HookedSAETransformer = field(repr=False)
    feature_scores: torch.Tensor | None = None
    feature_hook: str | None = None

    def top_tokens(
        self, hook: str, position: int = -1, k: int = 10, largest: bool = True
    ) -> list[tuple[str, float]]:
        """Top tokens on one side of the contrast at a hook and position.

        Positive scores are tokens the prompt boosts relative to the mean
        baseline; with largest=False the suppressed side is returned instead.

        Raises:
            KeyError: If the hook was not part of the run.
        """
        scores = self.token_scores[hook][position]
        top = scores.topk(k, largest=largest)
        return [
            (self.model.to_single_str_token(int(index)), float(value))
            for index, value in zip(top.indices.tolist(), top.values.tolist())
        ]

    def top_features(
        self, position: int = -1, k: int = 10, largest: bool = True
    ) -> list[tuple[int, float]]:
        """Top SAE features on one side of the contrast at a position.

        Positive scores are features whose decoder directions the prompt loads
        more than the mean baseline does; with largest=False the suppressed
        side is returned instead.

        Raises:
            ValueError: If no SAE was given to run_contrastive_projection.
        """
        if self.feature_scores is None:
            raise ValueError(
                "Feature attribution requires an SAE in run_contrastive_projection."
            )
        top = self.feature_scores[position].topk(k, largest=largest)
        return list(zip(top.indices.tolist(), top.values.tolist()))


def run_contrastive_projection(
    model: HookedSAETransformer,
    prompt: str,
    baselines: str | list[str],
    sae: SAE[Any] | None = None,
    hook_names: list[str] | None = None,
) -> ContrastiveProjection:
    """Trace the contrast between a prompt and positionally matched baselines.

    Runs the prompt and each baseline through the model cache, differences the
    prompt against the mean baseline hidden state at every hook, and projects
    each difference through the unembedding. When an SAE is given, prompts run
    through the SAE-attached cache path with an error term (leaving the
    residual stream unchanged) and the contrast at the SAE's hook is also
    projected onto the SAE's decoder directions, attributing it to features.

    Args:
        model: The model to trace.
        prompt: The prompt of interest.
        baselines: A prompt or list of prompts differing from prompt only in
            the concept under study.
        sae: Optional SAE used to attribute the contrast to features.
        hook_names: Cache hooks to read the token-space contrast at. Defaults
            to the residual stream after every block.

    Returns:
        The hidden contrasts, token scores and (if an SAE was given) feature
        scores of the run.

    Raises:
        ValueError: If no baseline is given, prompts tokenize to different
            numbers of positions, or the SAE has no hook name.
    """
    if isinstance(baselines, str):
        baselines = [baselines]
    if len(baselines) == 0:
        raise ValueError("At least one baseline prompt is required.")
    if hook_names is None:
        hook_names = [
            f"blocks.{layer}.hook_resid_post" for layer in range(model.cfg.n_layers)
        ]
    sae_hook = _sae_hook_name(sae)

    read_hooks = list(hook_names)
    if sae_hook is not None and sae_hook not in read_hooks:
        read_hooks.append(sae_hook)

    target = _read_hidden_states(model, prompt, read_hooks, sae, sae_hook)
    n_positions = next(iter(target.values())).shape[0]
    baseline_hiddens = []
    for baseline in baselines:
        hiddens = _read_hidden_states(model, baseline, read_hooks, sae, sae_hook)
        baseline_positions = next(iter(hiddens.values())).shape[0]
        if baseline_positions != n_positions:
            raise ValueError(
                "Prompts must be positionally matched: all prompts must "
                f"tokenize to the same number of positions, but {prompt!r} has "
                f"{n_positions} and {baseline!r} has {baseline_positions}."
            )
        baseline_hiddens.append(hiddens)

    contrast = {}
    for hook in read_hooks:
        baseline_mean = torch.stack(
            [hiddens[hook] for hiddens in baseline_hiddens]
        ).mean(dim=0)
        contrast[hook] = target[hook] - baseline_mean

    token_scores = {
        hook: contrastive_logit_lens(contrast[hook], model.W_U) for hook in hook_names
    }
    feature_scores = None
    if sae is not None and sae_hook is not None:
        hidden_contrast = contrast[sae_hook]
        feature_scores = contrastive_feature_attribution(
            hidden_contrast.reshape(hidden_contrast.shape[0], -1), sae.W_dec
        )
    return ContrastiveProjection(
        prompt=prompt,
        baselines=baselines,
        contrast=contrast,
        token_scores=token_scores,
        model=model,
        feature_scores=feature_scores,
        feature_hook=sae_hook,
    )


def _sae_hook_name(sae: SAE[Any] | None) -> str | None:
    """Hook whose activation space the SAE's decoder directions live in.

    For a transcoder this is the output hook (where the wrapper is installed
    and W_dec points); for an SAE the input and output hook coincide.
    """
    if sae is None:
        return None
    hook_name = sae.cfg.metadata.hook_name_out or sae.cfg.metadata.hook_name
    if hook_name is None:
        raise ValueError(
            "The SAE must have a hook name to attribute the contrast to features."
        )
    return hook_name


def _read_hidden_states(
    model: HookedSAETransformer,
    prompt: str,
    hook_names: list[str],
    sae: SAE[Any] | None,
    sae_hook: str | None,
) -> dict[str, torch.Tensor]:
    """Cache hidden states at each hook for one prompt.

    An attached SAE replaces the hook point at its own hook, so the clean
    activation there is read from the wrapper's hook_sae_output instead (with
    an error term the wrapper output equals the original activation).
    """
    if sae is not None:
        _, cache = model.run_with_cache_with_saes(
            prompt, saes=[sae], use_error_term=True, remove_batch_dim=True
        )
    else:
        _, cache = model.run_with_cache(prompt, remove_batch_dim=True)
    hiddens = {}
    for hook in hook_names:
        source = hook
        if sae is not None and hook == sae_hook:
            source = f"{hook}.hook_sae_output"
        hiddens[hook] = cache[source]
    return hiddens
