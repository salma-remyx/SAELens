"""Pairwise rank stabilization for BatchTopK SAE training.

Wide TopK SAEs lose rare features because the active budget k creates a hard
selection boundary: a feature whose encoder pre-activation sits close to the
cutoff flips in and out of the active set when the same meaning is expressed in
a different surface form. "Active Budget Can Kill Sensitivity: Diagnosing and
Repairing TopK Sparse Autoencoder Reliability" (arxiv:2609.37857) attributes
this failure to the geometry of TopK selection, diagnoses it with the active
margin (the distance from a pre-activation to the cutoff), and repairs it with
pairwise rank stabilization, a regularizer that targets ordering failures at
the cutoff across semantically paired inputs.

This module implements that mechanism for BatchTopK SAEs:

- `batchtopk_cutoff` and `active_margin` expose the selection-boundary geometry.
- `pairwise_rank_stabilization_loss` is the regularizer, computed over encoder
  pre-activations of a batch whose rows are grouped into paraphrase pairs.
- `feature_sensitivity` measures how consistently features stay active across
  the two members of each pair.
- `RankStabilizedBatchTopKTrainingSAE` is a `BatchTopKTrainingSAE` that adds the
  regularizer to its `calculate_aux_loss` dict (the same contract the Matryoshka
  auxiliary loss uses) and logs boundary metrics on every training step.

The regularizer is opt-in: with the default coefficient of 0 the SAE behaves
identically to a plain `BatchTopKTrainingSAE`. When enabled, batches must be
arranged so that rows (2i, 2i+1) of the flattened (num_samples, d_sae) tensor
are paraphrase pairs. To train with the standard runner, register the class
first:

    register_sae_training_class(
        "rank_stabilized_batchtopk",
        RankStabilizedBatchTopKTrainingSAE,
        RankStabilizedBatchTopKTrainingSAEConfig,
    )
"""

from dataclasses import dataclass

import torch
from typing_extensions import override

from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput


def batchtopk_cutoff(pre_acts: torch.Tensor, k: float) -> torch.Tensor:
    """
    Value at the BatchTopK selection boundary: the smallest activation that
    BatchTopK would keep for this batch, i.e. the k*num_samples-th largest of
    the ReLU'd pre-activations over the whole (flattened) batch. This is the
    quantity the `topk_threshold` buffer of `BatchTopKTrainingSAE` tracks with
    an exponential moving average.

    Args:
        pre_acts: Encoder pre-activations of shape (..., d_sae).
        k: Average number of features kept per sample (may be fractional).
    """
    flat_acts = pre_acts.relu().flatten()
    num_selected = int(k * pre_acts.shape[:-1].numel())
    num_selected = max(min(num_selected, flat_acts.numel()), 1)
    return torch.topk(flat_acts, num_selected).values.min()


def active_margin(pre_acts: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
    """
    Distance from each pre-activation to the selection cutoff. Positive for
    features the selection keeps, negative for features it drops; features at
    the cutoff have margin zero. Small positive margins flag features whose
    selection is fragile under semantic variation.

    Args:
        pre_acts: Encoder pre-activations of shape (..., d_sae).
        cutoff: Scalar cutoff value from `batchtopk_cutoff`.
    """
    return pre_acts - cutoff


def mean_active_margin(pre_acts: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
    """
    Mean margin of the features the selection keeps (strictly above the
    cutoff), in units of the pre-activations. Returns 0 if nothing is kept.
    """
    margins = active_margin(pre_acts, cutoff)
    kept = margins > 0
    if not kept.any():
        return pre_acts.new_tensor(0.0)
    return margins[kept].mean()


def pairwise_rank_stabilization_loss(
    pre_acts: torch.Tensor, k: float
) -> torch.Tensor:
    """
    Penalize ordering failures at the selection boundary across paraphrase
    pairs. For each pair and feature, if the feature is kept in one member but
    falls below the cutoff in the other, the deficit (distance below the
    cutoff) is charged, capped by the margin with which the feature was kept in
    its pair-mate. Features kept in both members, or in neither, contribute
    nothing, so the loss only acts on boundary rank instability: gradients push
    dropped features back towards the cutoff — never by more than the
    pair-mate's keeping margin — and never push a kept feature down or touch
    features that are inactive in both members.

    Args:
        pre_acts: Encoder pre-activations of shape (n_pairs, 2, d_sae), where
            [i, 0] and [i, 1] are two surface forms of the same meaning.
        k: Average number of features kept per sample (may be fractional).
    """
    cutoff = batchtopk_cutoff(pre_acts, k).detach()
    margins = active_margin(pre_acts, cutoff)
    # strength with which each member selected the feature (also the cap on
    # the penalty, detached so gradients only flow to the deficient member)
    kept_strength = margins.relu()
    deficit = (-margins).relu()
    dropped_in_second = torch.minimum(kept_strength[:, 0].detach(), deficit[:, 1])
    dropped_in_first = torch.minimum(kept_strength[:, 1].detach(), deficit[:, 0])
    return (dropped_in_second + dropped_in_first).sum(dim=-1).mean()


def feature_sensitivity(feature_acts: torch.Tensor) -> torch.Tensor:
    """
    Fraction of each pair's union of active feature sets that is shared by both
    members, averaged over pairs (1.0 = perfectly stable features, 0.0 = fully
    inconsistent). Pairs where neither member activates any feature are
    excluded from the average.

    Args:
        feature_acts: Feature activations of shape (..., d_sae) with an even
            number of samples, rows (2i, 2i+1) being paraphrase pairs.
    """
    active = (feature_acts > 0).reshape(-1, 2, feature_acts.shape[-1])
    intersection = (active[:, 0] & active[:, 1]).sum(dim=-1).float()
    union = (active[:, 0] | active[:, 1]).sum(dim=-1).float()
    non_empty = union > 0
    if not non_empty.any():
        return feature_acts.new_tensor(0.0)
    return (intersection[non_empty] / union[non_empty]).mean()


def _paired_rows(acts: torch.Tensor) -> torch.Tensor:
    """
    Reshape activations of shape (..., d_sae) into (n_pairs, 2, d_sae) with
    rows (2i, 2i+1) grouped into paraphrase pairs.
    """
    flat_acts = acts.reshape(-1, acts.shape[-1])
    if flat_acts.shape[0] % 2 != 0:
        raise ValueError(
            "Pairwise rank stabilization requires an even number of samples so "
            "that rows (2i, 2i+1) can be grouped into paraphrase pairs, got "
            f"{flat_acts.shape[0]} samples."
        )
    return flat_acts.view(flat_acts.shape[0] // 2, 2, flat_acts.shape[1])


@dataclass
class RankStabilizedBatchTopKTrainingSAEConfig(BatchTopKTrainingSAEConfig):
    """
    Configuration class for training a RankStabilizedBatchTopKTrainingSAE.

    This is a BatchTopK SAE with an optional pairwise rank stabilization loss
    that targets ordering failures at the selection boundary across
    semantically paired inputs, improving the sensitivity of rare features
    (arxiv:2609.37857). Batches must be arranged so that rows (2i, 2i+1) are
    paraphrase pairs.

    After training, RankStabilizedBatchTopK SAEs are saved as JumpReLU SAEs,
    like standard BatchTopK SAEs.

    Args:
        pairwise_rank_loss_coefficient (float): Coefficient on the pairwise
            rank stabilization loss. 0 (the default) disables it, making the
            SAE behave identically to a plain BatchTopKTrainingSAE.
        k (float): Average number of features to keep active across the batch.
            Inherited from BatchTopKTrainingSAEConfig.
        topk_threshold_lr (float): Learning rate for updating the global topk
            threshold. Inherited from BatchTopKTrainingSAEConfig.
        aux_loss_coefficient (float): Coefficient for the auxiliary loss that
            encourages dead neurons to learn useful features. Inherited from
            TopKTrainingSAEConfig.
        rescale_acts_by_decoder_norm (bool): Treat the decoder as if it was
            already normalized. Inherited from TopKTrainingSAEConfig.
        decoder_init_norm (float | None): Norm to initialize decoder weights
            to. Inherited from TrainingSAEConfig.
        d_in (int): Input dimension (dimensionality of the activations being
            encoded). Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from
            SAEConfig.
        device (str): Device to place the SAE on. Inherited from SAEConfig.
    """

    pairwise_rank_loss_coefficient: float = 0.0

    @override
    @classmethod
    def architecture(cls) -> str:
        return "rank_stabilized_batchtopk"


class RankStabilizedBatchTopKTrainingSAE(BatchTopKTrainingSAE):
    """
    BatchTopK SAE with pairwise rank stabilization (arxiv:2609.37857).

    In addition to the standard BatchTopK training losses, this SAE penalizes
    ordering failures at the selection boundary across paraphrase pairs of
    inputs, which stabilizes the ranks of rare features near the cutoff. It
    also logs selection-boundary metrics on every training step:

    - `active_cutoff`: the batch's BatchTopK selection boundary.
    - `mean_active_margin`: mean margin of the kept features.
    - `pairwise_feature_sensitivity`: overlap of active feature sets across
      pair members (only logged when the rank stabilization loss is enabled,
      since it is only meaningful for paired batches).
    """

    cfg: RankStabilizedBatchTopKTrainingSAEConfig  # type: ignore[assignment]

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        aux_losses = super().calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            sae_out=sae_out,
        )
        if self.cfg.pairwise_rank_loss_coefficient > 0:
            paired_pre_acts = _paired_rows(hidden_pre)
            rank_loss = pairwise_rank_stabilization_loss(paired_pre_acts, self.cfg.k)
            aux_losses["pairwise_rank_stabilization_loss"] = (
                self.cfg.pairwise_rank_loss_coefficient * rank_loss
            )
        return aux_losses

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        output = super().training_forward_pass(step_input)
        with torch.no_grad():
            flat_hidden_pre = output.hidden_pre.reshape(-1, self.cfg.d_sae)
            cutoff = batchtopk_cutoff(flat_hidden_pre, self.cfg.k)
            output.metrics["active_cutoff"] = cutoff
            output.metrics["mean_active_margin"] = mean_active_margin(
                flat_hidden_pre, cutoff
            )
            if self.cfg.pairwise_rank_loss_coefficient > 0:
                output.metrics["pairwise_feature_sensitivity"] = feature_sensitivity(
                    output.feature_acts
                )
        return output
