"""Strength controls for comparing SAE feature interventions.

Changing SAE codes changes both the direction and the magnitude of an
intervention, so two interventions (or the same intervention expressed in two
different SAEs) can only be compared once a *strength control* fixes what is
held equal. This module implements the controls compared in "Strength Controls
Change How Interventions Compare Across Sparse Autoencoder Codes"
(arXiv:2610.26489):

- Norm matching holds the patch norm equal, with one coefficient per input
  (``total_norm_coefficients``) or one per position
  (``positionwise_norm_coefficients``).
- Full-output KL calibration holds the model's output distribution equal:
  ``calibrate_intervention_strength`` picks one coefficient per input so that
  the intervened logits stay closest, in KL, to a reference intervention's
  logits. The residual divergence at the chosen coefficient is returned as the
  *effect gap*, the diagnostic the paper uses to compare controls. Because
  calibration reads the evaluated input's own outputs, it doubles as a
  diagnostic of residual disagreement between codes.

Patches are applied through the existing hook-based intervention path: the
scaled patch is added to ``hook_sae_acts_post`` activations while the SAE is
attached by ``HookedSAETransformer.run_with_hooks_with_saes``.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import torch

FEATURE_ACTS_HOOK = "hook_sae_acts_post"


class _RunsHooksWithSaes(Protocol):
    """The slice of ``HookedSAETransformer`` this module drives."""

    def run_with_hooks_with_saes(
        self, *model_args: Any, **model_kwargs: Any
    ) -> Any: ...


@dataclass
class StrengthCalibration:
    """Per-input strengths chosen by full-output KL calibration.

    Attributes:
        coefficients: (batch,) winning coefficient for each input.
        candidates: (num_candidates,) shared candidate grid that was searched.
        candidate_gaps: (num_candidates, batch) output KL to the reference at
            each candidate coefficient.
        gaps: (batch,) effect gap (output KL to the reference) at the chosen
            coefficient.
    """

    coefficients: torch.Tensor
    candidates: torch.Tensor
    candidate_gaps: torch.Tensor
    gaps: torch.Tensor

    @property
    def mean_gap(self) -> torch.Tensor:
        """Mean effect gap across inputs, the comparison statistic of the paper."""
        return self.gaps.mean()


def _norm_ratio(
    patch_norms: torch.Tensor, reference_norms: torch.Tensor
) -> torch.Tensor:
    """Non-negative reference / patch norm ratio, 0 where the patch has no mass."""
    empty = patch_norms == 0
    safe_patch_norms = torch.where(empty, torch.ones_like(patch_norms), patch_norms)
    return torch.where(
        empty, torch.zeros_like(reference_norms), reference_norms / safe_patch_norms
    )


def total_norm_coefficients(
    patch: torch.Tensor,
    reference_patch: torch.Tensor,
) -> torch.Tensor:
    """One coefficient per input matching the patch's total norm to a reference.

    Args:
        patch: (batch, seq, num_features) intervention patch to scale.
        reference_patch: (batch, seq, num_features) patch whose per-input norms
            are matched.

    Returns:
        (batch,) coefficients so that the scaled patch and the reference patch
        have equal norm for every input; 0 for inputs whose patch is all zeros.
    """
    return _norm_ratio(
        torch.linalg.vector_norm(patch.flatten(1), dim=-1),
        torch.linalg.vector_norm(reference_patch.flatten(1), dim=-1),
    )


def positionwise_norm_coefficients(
    patch: torch.Tensor,
    reference_patch: torch.Tensor,
) -> torch.Tensor:
    """One coefficient per position matching the patch norm to a reference.

    Args:
        patch: (batch, seq, num_features) intervention patch to scale.
        reference_patch: (batch, seq, num_features) patch whose per-position
            norms are matched.

    Returns:
        (batch, seq) coefficients so that scaled and reference patches have
        equal norm at every position; 0 where the patch is all zeros.
    """
    return _norm_ratio(
        torch.linalg.vector_norm(patch, dim=-1),
        torch.linalg.vector_norm(reference_patch, dim=-1),
    )


def scaled_patch(patch: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Scale a patch by scalar, (batch,) or (batch, seq) coefficients."""
    if coefficients.dim() == 1:
        coefficients = coefficients[:, None, None]
    elif coefficients.dim() == 2:
        coefficients = coefficients[:, :, None]
    elif coefficients.dim() > 2:
        raise ValueError(
            f"coefficients must be scalar, (batch,) or (batch, seq), got shape "
            f"{tuple(coefficients.shape)}"
        )
    scaled = patch * coefficients
    return scaled.to(device=patch.device, dtype=patch.dtype)


def output_kl(
    logits: torch.Tensor,
    reference_logits: torch.Tensor,
    exclude: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-input KL(reference || logits) over the model's output distribution.

    Args:
        logits: (batch, seq, d_vocab) logits under the intervention being
            scored.
        reference_logits: (batch, seq, d_vocab) logits under the reference
            intervention.
        exclude: optional (batch, seq) boolean mask where True drops that
            position from the divergence, giving the paper's answer-excluded
            variant.

    Returns:
        (batch,) divergence in nats, summed over positions and vocabulary.
    """
    log_p = torch.log_softmax(logits.float(), dim=-1)
    log_q = torch.log_softmax(reference_logits.float(), dim=-1)
    per_position = (log_q.exp() * (log_q - log_p)).sum(dim=-1)
    if exclude is not None:
        per_position = per_position.masked_fill(exclude, 0.0)
    return per_position.sum(dim=-1)


def run_with_feature_patch(
    model: _RunsHooksWithSaes,
    sae: Any,
    tokens: torch.Tensor,
    patch: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Run one forward pass with a coefficient-scaled patch on SAE feature acts.

    The SAE is attached for a single pass and the scaled patch is added to its
    feature activations, so it is decoded back into the model like any other
    feature intervention.

    Args:
        model: model exposing ``run_with_hooks_with_saes``.
        sae: the SAE to attach; only its hook point is read.
        tokens: (batch, seq) model input.
        patch: (batch, seq, num_features) feature-space intervention patch.
        coefficients: scalar, (batch,) or (batch, seq) strength control
            coefficients.

    Returns:
        (batch, seq, d_vocab) logits under the scaled intervention.
    """
    act_name = sae.cfg.metadata.hook_name
    scaled = scaled_patch(patch, coefficients)

    def add_patch(acts: torch.Tensor, _hook: Any) -> torch.Tensor:
        return acts + scaled.to(device=acts.device, dtype=acts.dtype)

    return model.run_with_hooks_with_saes(
        tokens,
        saes=[sae],
        fwd_hooks=[(f"{act_name}.{FEATURE_ACTS_HOOK}", add_patch)],
    )


def calibrate_intervention_strength(
    model: _RunsHooksWithSaes,
    sae: Any,
    tokens: torch.Tensor,
    patch: torch.Tensor,
    reference_logits: torch.Tensor,
    candidates: torch.Tensor | None = None,
    exclude: torch.Tensor | None = None,
) -> StrengthCalibration:
    """Choose one coefficient per input by full-output KL calibration.

    Every candidate coefficient is applied to the whole batch in a single
    forward pass through the hooked SAE path; per input, the coefficient whose
    logits stay closest to ``reference_logits`` under ``output_kl`` wins.

    Args:
        model: model exposing ``run_with_hooks_with_saes``.
        sae: the SAE to attach for the intervention.
        tokens: (batch, seq) model input shared by all candidates.
        patch: (batch, seq, num_features) feature-space intervention patch.
        reference_logits: (batch, seq, d_vocab) logits under the reference
            intervention the candidate patch is compared against.
        candidates: optional (num_candidates,) shared coefficient grid.
            Defaults to powers of two from 1/16 to 16, which is relative to the
            patch as supplied; pass a grid centred on
            ``total_norm_coefficients`` to search around the norm-matched
            strength.
        exclude: optional (batch, seq) boolean mask where True drops that
            position from the calibration objective.

    Returns:
        A ``StrengthCalibration`` holding the winning per-input coefficients
        and the effect gaps at every candidate.
    """
    if candidates is None:
        candidates = 2.0 ** torch.arange(-4, 5, device=patch.device)
    candidates = candidates.to(device=patch.device, dtype=torch.float32)

    batch_size = patch.shape[0]
    candidate_gaps: list[torch.Tensor] = []
    with torch.no_grad():
        for coefficient in candidates:
            logits = run_with_feature_patch(
                model, sae, tokens, patch, coefficient.expand(batch_size)
            )
            candidate_gaps.append(output_kl(logits, reference_logits, exclude))
    all_gaps = torch.stack(candidate_gaps)

    best = all_gaps.argmin(dim=0)
    batch_index = torch.arange(batch_size, device=all_gaps.device)
    return StrengthCalibration(
        coefficients=candidates[best],
        candidates=candidates,
        candidate_gaps=all_gaps,
        gaps=all_gaps[best, batch_index],
    )
