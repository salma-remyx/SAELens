"""Activation-selection policies compared at a matched activation budget.

Decoding an SAE pays for every feature a selection policy keeps, so the policy
that turns encoder scores into a sparse activation decides both how many
features are read and which ones. This repo already ships the two policies
studied by "Channel Selection for Mixture-of-Experts Decoding Needs No
Calibration Data" (arXiv:2610.37386): per-token top-k (``TopK``, used by
``TopKSAE``) and a global threshold (the EMA threshold ``BatchTopKTrainingSAE``
tracks, exported as a JumpReLU threshold when the trained SAE is saved).

That paper's finding on routed-expert channel reads transfers directly to
feature reads: a threshold calibrated for a target read budget does not deliver
that budget, because the number of scores above a fixed threshold varies from
token to token, while per-token top-k spends exactly k reads on every token and
matches or beats the threshold at the same mean budget. The helpers below
measure this on any SAE: read the shared encoder score once, apply each policy
to it, and report the budget each policy actually spent, how unevenly it spent
it, what reconstruction it bought, and what the decode cost.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch

from sae_lens.saes.sae import SAE
from sae_lens.saes.topk_sae import TopK


@dataclass
class SelectionBudgetReport:
    """How one selection policy spent its activation budget on a batch.

    Attributes:
        policy: Name of the selection policy that was measured.
        d_sae: Number of features the SAE could have selected from.
        num_tokens: Number of tokens the budget was spent over.
        intended_active: Features per token the policy was configured to keep.
        realized_active: Mean features per token that actually contributed to
            the decode.
        active_p10: 10th percentile of per-token active features.
        active_p50: Median per-token active features.
        active_p90: 90th percentile of per-token active features.
        intended_budget: ``intended_active`` as a fraction of ``d_sae``.
        realized_budget: ``realized_active`` as a fraction of ``d_sae``.
        budget_ratio: ``realized_active / intended_active``; 1.0 means the
            policy spent exactly the budget it was given.
        threshold: The global threshold the threshold policy used, if any.
        reconstruction_mse: Mean squared error of the decode against its input.
        decode_seconds: Mean seconds per ``sae.decode`` call.
    """

    policy: str
    d_sae: int
    num_tokens: int
    intended_active: float
    realized_active: float
    active_p10: float
    active_p50: float
    active_p90: float
    intended_budget: float
    realized_budget: float
    budget_ratio: float
    threshold: float | None
    reconstruction_mse: float
    decode_seconds: float


def sae_hidden_pre(sae: SAE[Any], x: torch.Tensor) -> torch.Tensor:
    """Pre-activation feature scores for ``x``, read through the SAE's encoder.

    The score is cached from the SAE's own ``hook_sae_acts_pre`` hook so that
    every policy selects from the same values the SAE's encode step sees,
    before any architecture-specific rescaling of those values.
    """
    cache, fwd_hooks, _ = sae.get_caching_hooks(names_filter="hook_sae_acts_pre")
    with sae.hooks(fwd_hooks=fwd_hooks):
        sae.encode(x)
    return cache["hook_sae_acts_pre"]


def calibrate_threshold(scores: torch.Tensor, k: float) -> float:
    """Threshold keeping an average of ``k`` features per token on ``scores``.

    This is the offline-calibration step a threshold policy needs. The value is
    the boundary of a batch-level top-k over ``scores`` — the quantity
    ``BatchTopKTrainingSAE`` tracks an EMA of — so on the calibration batch
    itself exactly ``int(k * num_tokens)`` scores sit above it. On any other
    batch the count above it is whatever that batch's score distribution
    happens to give, which is the failure mode arXiv:2610.37386 measures.
    """
    num_tokens = scores.shape[:-1].numel()
    n_keep = int(k * num_tokens)
    if not 0 < n_keep < scores.numel():
        raise ValueError(
            f"k={k} over {num_tokens} tokens must keep between 1 and "
            f"{scores.numel() - 1} of the {scores.numel()} scores"
        )
    # the largest score a batch-level top-k of n_keep excludes: strictly greater
    # scores are exactly the n_keep the budget was set for
    boundary = torch.topk(scores.flatten(), n_keep + 1).values.min()
    return float(boundary)


def select_topk(
    scores: torch.Tensor, k: int, use_sparse_activations: bool = False
) -> torch.Tensor:
    """Per-token top-k selection: the ``k`` highest-scoring features per token.

    Uses the repo's ``TopK`` activation, so ``use_sparse_activations=True``
    returns the COO sparse activations the sparse-decoding path gathers from.
    """
    return TopK(k, use_sparse_activations=use_sparse_activations)(scores)


def select_threshold(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Global-threshold selection: keep features scoring strictly above it.

    This is the rule ``JumpReLUSAE.encode`` applies — ReLU activations masked
    by ``hidden_pre > threshold`` — with a single threshold shared by every
    feature, which is what the BatchTopK to JumpReLU conversion exports.
    """
    return (scores > threshold).to(scores.dtype) * scores.relu()


def active_counts(feature_acts: torch.Tensor) -> torch.Tensor:
    """Features per token that contribute to the decode (positive activations)."""
    if feature_acts.is_sparse:
        feature_acts = feature_acts.to_dense()
    return (feature_acts > 0).sum(dim=-1)


def _timed_decode(
    sae: SAE[Any], feature_acts: torch.Tensor, n_repeats: int
) -> tuple[torch.Tensor, float]:
    """Decode ``feature_acts`` and time the decode, after a warmup call."""
    with torch.no_grad():
        sae_out = sae.decode(feature_acts)
        start = perf_counter()
        for _ in range(n_repeats):
            sae_out = sae.decode(feature_acts)
        seconds = (perf_counter() - start) / n_repeats
    return sae_out, seconds


def _build_report(
    policy: str,
    feature_acts: torch.Tensor,
    sae_out: torch.Tensor,
    x: torch.Tensor,
    intended_active: float,
    threshold: float | None,
    decode_seconds: float,
) -> SelectionBudgetReport:
    counts = active_counts(feature_acts).float()
    d_sae = feature_acts.shape[-1]
    realized = counts.mean().item()
    return SelectionBudgetReport(
        policy=policy,
        d_sae=d_sae,
        num_tokens=counts.numel(),
        intended_active=intended_active,
        realized_active=realized,
        active_p10=counts.quantile(0.1).item(),
        active_p50=counts.quantile(0.5).item(),
        active_p90=counts.quantile(0.9).item(),
        intended_budget=intended_active / d_sae,
        realized_budget=realized / d_sae,
        budget_ratio=realized / intended_active,
        threshold=threshold,
        reconstruction_mse=(sae_out - x).pow(2).mean().item(),
        decode_seconds=decode_seconds,
    )


def compare_selection_policies(
    sae: SAE[Any],
    x: torch.Tensor,
    k: float,
    calibration_scores: torch.Tensor | None = None,
    threshold: float | None = None,
    n_decode_repeats: int = 10,
) -> dict[str, SelectionBudgetReport]:
    """Compare per-token top-k against a calibrated threshold on the same scores.

    Every policy selects from one shared encoder score, so the reports isolate
    the effect of the selection policy, which is the comparison
    arXiv:2610.37386 runs over channel selectors. The threshold defaults to one
    calibrated to keep ``k`` features per token on average, which is the budget
    top-k spends by construction.

    Args:
        sae: SAE to encode with and decode through. Its decode must return the
            shape of ``x``.
        x: Activations to select on, shape (batch_size, d_in).
        k: Target features per token. top-k selects ``int(k)`` per token; the
            threshold is calibrated to the same average.
        calibration_scores: Scores the threshold is calibrated on, e.g. from a
            held-out batch via ``sae_hidden_pre``. Defaults to the scores of
            ``x`` itself, which measures the threshold on its calibration data.
        threshold: Use this threshold instead of calibrating one.
        n_decode_repeats: Timed decode calls averaged into each report.
    """
    scores = sae_hidden_pre(sae, x)
    if threshold is None:
        threshold = calibrate_threshold(
            scores if calibration_scores is None else calibration_scores, k
        )
    selections = {
        "topk": select_topk(scores, int(k)),
        "topk_sparse": select_topk(scores, int(k), use_sparse_activations=True),
        "threshold": select_threshold(scores, threshold),
    }
    reports: dict[str, SelectionBudgetReport] = {}
    for policy, feature_acts in selections.items():
        sae_out, decode_seconds = _timed_decode(sae, feature_acts, n_decode_repeats)
        reports[policy] = _build_report(
            policy=policy,
            feature_acts=feature_acts,
            sae_out=sae_out,
            x=x,
            intended_active=k,
            threshold=threshold if policy == "threshold" else None,
            decode_seconds=decode_seconds,
        )
    return reports


def format_reports(reports: dict[str, SelectionBudgetReport]) -> str:
    """One line per policy: budget hit, per-token spread, reconstruction, cost."""
    lines = [
        f"{'policy':<14}{'intended':>9}{'realized':>9}{'p10':>7}{'p90':>7}"
        f"{'budget_ratio':>13}{'mse':>12}{'decode_ms':>11}"
    ]
    for report in reports.values():
        lines.append(
            f"{report.policy:<14}{report.intended_active:>9.1f}"
            f"{report.realized_active:>9.2f}{report.active_p10:>7.1f}"
            f"{report.active_p90:>7.1f}{report.budget_ratio:>13.3f}"
            f"{report.reconstruction_mse:>12.4f}{report.decode_seconds * 1000:>11.3f}"
        )
    return "\n".join(lines)
