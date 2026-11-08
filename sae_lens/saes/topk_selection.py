"""Budget-matched, calibration-free top-k feature selection for threshold SAEs.

A BatchTopK SAE is saved as a JumpReLU SAE whose threshold is an exponential
moving average learned from training data, so at inference the feature budget is
only known after the fact and varies from sample to sample. This module ports the
selection result from "Channel Selection for Mixture-of-Experts Decoding Needs No
Calibration Data" (arXiv:2610.37386) onto that policy choice: keeping each
sample's k highest-scoring features needs no calibration data, caps the budget
per sample, and at a matched average budget is a drop-in alternative to the
threshold policy on the same score.

The paper's MoE channel score (a learned router / ternary sketch reader) is
replaced by the SAE's own pre-activation, which is the score the JumpReLU
threshold already selects on; its kernels and benchmark suites are out of scope.
"""

from dataclasses import dataclass

import torch

from sae_lens.saes.jumprelu_sae import JumpReLUSAE
from sae_lens.saes.topk_sae import TopK


def l0_per_sample(feature_acts: torch.Tensor) -> torch.Tensor:
    """
    Number of active features per sample.

    Args:
        feature_acts: Activations of shape (..., d_sae); every leading dimension
            is treated as a sample dimension.

    Returns:
        Integer counts of shape feature_acts.shape[:-1].
    """
    return (feature_acts != 0).sum(dim=-1)


def hidden_pre_from_sae(sae: JumpReLUSAE, x: torch.Tensor) -> torch.Tensor:
    """
    Pre-activations of a JumpReLU SAE, i.e. the score its threshold selects on.

    This is the first step of JumpReLUSAE.encode, of shape (..., d_sae), so it
    fires the same pre-activation hook and sees any hooks encode itself would.
    """
    sae_in = sae.process_sae_in(x)
    return sae.hook_sae_acts_pre(sae_in @ sae.W_enc + sae.b_enc)


def select_topk_acts(hidden_pre: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keep each sample's k highest pre-activations, zero the rest.

    This is the calibration-free selection policy: the number of features read is
    capped at exactly k for every sample, with no threshold to calibrate on
    held-out data. The number that end up non-zero is min(k, the count of
    positive pre-activations), since the kept values are ReLU'd.
    """
    return TopK(k)(hidden_pre)


@dataclass
class SelectionBudget:
    """
    How many features a selection policy actually keeps, per sample.

    Args:
        mean_l0: Average number of active features per sample, i.e. the budget
            the policy delivers rather than the one it was calibrated for.
        std_l0: Standard deviation of the per-sample count. A threshold policy
            has non-zero std because the count above a fixed threshold varies per
            sample; a top-k policy has std_l0 == 0 whenever every sample has at
            least k positive pre-activations.
        min_l0: Smallest per-sample count.
        max_l0: Largest per-sample count.
        d_sae: Number of features the policy selects among.
        nominal_l0: Budget the threshold was calibrated to deliver, if known
            (e.g. the k a BatchTopK SAE was trained with).
    """

    mean_l0: float
    std_l0: float
    min_l0: int
    max_l0: int
    d_sae: int
    nominal_l0: float | None = None

    @property
    def read_fraction(self) -> float:
        """Fraction of the features that get read, mean_l0 / d_sae."""
        return self.mean_l0 / self.d_sae

    @property
    def budget_ratio(self) -> float | None:
        """Delivered budget over the nominal one, or None if nominal is unknown."""
        if self.nominal_l0 is None:
            return None
        return self.mean_l0 / self.nominal_l0


def selection_budget(
    per_sample_l0: torch.Tensor, d_sae: int, nominal_l0: float | None = None
) -> SelectionBudget:
    """
    Summarize a policy's delivered budget from per-sample feature counts.

    Args:
        per_sample_l0: Counts from l0_per_sample, of shape (...,).
        d_sae: Number of features the policy selects among.
        nominal_l0: Budget the threshold was calibrated to deliver, if known.
    """
    counts = per_sample_l0.float()
    return SelectionBudget(
        mean_l0=counts.mean().item(),
        std_l0=counts.std().item(),
        min_l0=int(per_sample_l0.min().item()),
        max_l0=int(per_sample_l0.max().item()),
        d_sae=d_sae,
        nominal_l0=nominal_l0,
    )


def matched_budget_k(per_sample_l0: torch.Tensor, d_sae: int) -> int:
    """
    Integer k spending the same average budget as a threshold policy.

    Rounded to the nearest feature (ties up) and clamped to [1, d_sae], since
    top-k must keep at least one feature and cannot keep more than exist.

    Args:
        per_sample_l0: Counts from l0_per_sample, of shape (...,).
        d_sae: Number of features the policy selects among.
    """
    mean_l0 = per_sample_l0.float().mean().item()
    return max(1, min(d_sae, int(mean_l0 + 0.5)))


def reconstruction_mse(sae_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean squared reconstruction error of sae_out against the raw input."""
    return (sae_out.float() - x.float()).pow(2).mean()


@dataclass
class SelectionComparison:
    """
    Threshold and top-k selection run on the same batch.

    Both policies score features identically and decode through the same
    decoder; they differ only in how many features each sample keeps.

    Args:
        k: The top-k budget, matched to the threshold policy's average when not
            given explicitly.
        threshold_budget: Budget the threshold policy delivered.
        topk_budget: Budget the top-k policy delivered.
        threshold_mse: Reconstruction error of the threshold policy.
        topk_mse: Reconstruction error of the top-k policy.
        threshold_explained_variance: 1 - threshold_mse / variance of x.
        topk_explained_variance: 1 - topk_mse / variance of x.
    """

    k: int
    threshold_budget: SelectionBudget
    topk_budget: SelectionBudget
    threshold_mse: torch.Tensor
    topk_mse: torch.Tensor
    threshold_explained_variance: torch.Tensor
    topk_explained_variance: torch.Tensor

    def summary(self) -> str:
        """One line per policy: delivered budget, mse and explained variance."""
        lines = []
        for name, budget, mse, ev in (
            (
                "threshold",
                self.threshold_budget,
                self.threshold_mse,
                self.threshold_explained_variance,
            ),
            (
                "topk",
                self.topk_budget,
                self.topk_mse,
                self.topk_explained_variance,
            ),
        ):
            lines.append(
                f"{name:<9} L0 {budget.mean_l0:6.1f} +/- {budget.std_l0:4.1f} "
                f"(range {budget.min_l0}-{budget.max_l0}) "
                f"mse {mse.item():.5g} explained variance {ev.item():.5g}"
            )
        return "\n".join(lines)


@torch.no_grad()
def compare_threshold_and_topk(
    sae: JumpReLUSAE,
    x: torch.Tensor,
    k: int | None = None,
    nominal_l0: float | None = None,
) -> SelectionComparison:
    """
    Run the threshold and top-k selection policies head to head.

    Args:
        sae: A JumpReLU SAE, e.g. one saved from BatchTopK training.
        x: Activations of shape (..., d_in).
        k: Top-k budget. Defaults to the threshold policy's realized average
            budget on this batch, so the two are compared at a matched budget.
        nominal_l0: Budget the threshold was calibrated to deliver, if known.

    Returns:
        The budget each policy actually delivered plus reconstruction metrics.
    """
    threshold_acts = sae.encode(x)
    threshold_l0 = l0_per_sample(threshold_acts)
    if k is None:
        k = matched_budget_k(threshold_l0, sae.cfg.d_sae)
    topk_acts = select_topk_acts(hidden_pre_from_sae(sae, x), k)

    threshold_mse = reconstruction_mse(sae.decode(threshold_acts), x)
    topk_mse = reconstruction_mse(sae.decode(topk_acts), x)
    variance = x.float().var(unbiased=False)

    return SelectionComparison(
        k=k,
        threshold_budget=selection_budget(
            threshold_l0, sae.cfg.d_sae, nominal_l0=nominal_l0
        ),
        topk_budget=selection_budget(
            l0_per_sample(topk_acts), sae.cfg.d_sae, nominal_l0=nominal_l0
        ),
        threshold_mse=threshold_mse,
        topk_mse=topk_mse,
        threshold_explained_variance=1 - threshold_mse / variance,
        topk_explained_variance=1 - topk_mse / variance,
    )
