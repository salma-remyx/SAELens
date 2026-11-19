"""Contrastive BatchTopK SAE training.

Adds an InfoNCE contrastive auxiliary loss to BatchTopK SAE training so that
the similarity structure of the input (dense embedding) space is preserved in
the sparse codes. Adapted from "Explanations that Transfer: Interpreting
Medical Retrieval with Sparse AutoEncoders" (arXiv:2611.18179), which combines
an SAE reconstruction objective with an InfoNCE loss so that query-document
relevance survives the sparse bottleneck. The paper derives its positive
pairs from a labelled retrieval corpus, which SAELens does not provide, so
each sample's positive is instead chosen on the fly as its nearest neighbour
in the input space under cosine similarity (the relevance signal of a dense
retrieval model), with the remaining in-batch samples as negatives.

The loss plugs into the standard auxiliary-loss contract:
`TrainingSAE.training_forward_pass` sums it with the MSE and TopK dead-neuron
losses, and `SAETrainer` schedules its weight from `get_coefficients`, so no
trainer changes are needed. To use it with a language model runner, register
the architecture with the public registry first:

    register_sae_training_class(
        "contrastive_batchtopk",
        ContrastiveBatchTopKTrainingSAE,
        ContrastiveBatchTopKTrainingSAEConfig,
    )

After training, ContrastiveBatchTopK SAEs are saved as JumpReLU SAEs, exactly
like BatchTopK SAEs.
"""

from dataclasses import dataclass

import torch
from typing_extensions import override

from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.sae import TrainCoefficientConfig, TrainStepInput


def calculate_infonce_loss(
    sae_in: torch.Tensor,
    feature_acts: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    InfoNCE loss that keeps sparse codes of input-space neighbours close.

    For each sample, the positive is its nearest neighbour in the input space
    under cosine similarity (excluding itself) and every other in-batch sample
    acts as a negative. Code similarities are compared with a
    temperature-scaled cosine, and the cross-entropy of picking each sample's
    positive is averaged over all samples. Batch dimensions are flattened, so
    inputs of shape (batch_size, d_in) with codes of shape (batch_size, d_sae)
    are the common case.

    Args:
        sae_in (torch.Tensor): Input activations of shape (..., d_in). Only
            used to pick positive pairs, never differentiated through.
        feature_acts (torch.Tensor): Feature activations of shape
            (..., d_sae), with batch dimensions matching sae_in.
        temperature (float): InfoNCE temperature applied to code
            similarities. Must be positive.

    Returns:
        torch.Tensor: Scalar InfoNCE loss. Zero for batches with fewer than
            two samples.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if feature_acts.is_sparse:
        feature_acts = feature_acts.to_dense()
    flat_sae_in = sae_in.reshape(-1, sae_in.shape[-1])
    flat_acts = feature_acts.reshape(-1, feature_acts.shape[-1])
    if flat_sae_in.shape[0] < 2:
        return flat_acts.new_tensor(0.0)

    # positives are hard labels picked from the input space, so they carry no
    # gradient into the loss
    with torch.no_grad():
        sae_in_directions = flat_sae_in / flat_sae_in.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        input_similarities = sae_in_directions @ sae_in_directions.T
        input_similarities.fill_diagonal_(float("-inf"))
        positives = input_similarities.argmax(dim=-1)

    act_directions = flat_acts / flat_acts.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    logits = act_directions @ act_directions.T / temperature
    # a sample must not be its own candidate
    logits.fill_diagonal_(float("-inf"))
    return torch.nn.functional.cross_entropy(logits, positives)


@dataclass
class ContrastiveBatchTopKTrainingSAEConfig(BatchTopKTrainingSAEConfig):
    """
    Configuration class for training a ContrastiveBatchTopKTrainingSAE.

    This extends BatchTopK training with an InfoNCE contrastive loss that
    keeps the sparse codes of input-space nearest neighbours close (and other
    in-batch samples far away). This preserves the similarity / relevance
    structure of the input embedding space through the sparse bottleneck, at
    the cost of one extra (batch_size, batch_size) similarity matrix per step.
    Set `infonce_loss_coefficient` to 0 to recover plain BatchTopK training.

    After training, ContrastiveBatchTopK SAEs are saved as JumpReLU SAEs.

    Args:
        infonce_loss_coefficient (float): Weight of the InfoNCE contrastive
            loss relative to the reconstruction objective.
        infonce_warm_up_steps (int): Number of steps over which the InfoNCE
            loss weight is linearly warmed up by the trainer.
        infonce_temperature (float): Temperature for the InfoNCE contrastive
            loss; smaller values sharpen the contrast between the positive
            pair and in-batch negatives.
        k (float): Average number of features to keep active across the
            batch. Inherited from BatchTopKTrainingSAEConfig.
        topk_threshold_lr (float): Learning rate for updating the global topk
            threshold. Inherited from BatchTopKTrainingSAEConfig.
        aux_loss_coefficient (float): Coefficient for the auxiliary loss that
            encourages dead neurons to learn useful features. Inherited from
            TopKTrainingSAEConfig.
        d_in (int): Input dimension (dimensionality of the activations being
            encoded). Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
    """

    infonce_loss_coefficient: float = 0.0
    infonce_warm_up_steps: int = 0
    infonce_temperature: float = 0.1

    @override
    @classmethod
    def architecture(cls) -> str:
        return "contrastive_batchtopk"


class ContrastiveBatchTopKTrainingSAE(BatchTopKTrainingSAE):
    """
    BatchTopK Training SAE with an InfoNCE contrastive auxiliary loss.

    Each training step adds an InfoNCE term on top of the usual BatchTopK
    objectives: the sparse codes of input-space nearest neighbours are pulled
    together while other in-batch samples are pushed apart, so the sparse
    codes preserve the similarity structure (in a retrieval embedding space,
    the relevance structure) of the inputs.
    """

    cfg: ContrastiveBatchTopKTrainingSAEConfig  # type: ignore[assignment]

    def __init__(
        self, cfg: ContrastiveBatchTopKTrainingSAEConfig, use_error_term: bool = False
    ):
        super().__init__(cfg, use_error_term)
        if self.cfg.infonce_temperature <= 0:
            raise ValueError(
                "cfg.infonce_temperature must be positive, got "
                f"{self.cfg.infonce_temperature}"
            )

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        losses = super().calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            sae_out=sae_out,
        )
        # the trainer schedules this coefficient (with warmup) via
        # get_coefficients; fall back to the config value when aux losses are
        # calculated outside SAETrainer
        coefficient = step_input.coefficients.get(
            "infonce", self.cfg.infonce_loss_coefficient
        )
        losses["infonce_loss"] = coefficient * calculate_infonce_loss(
            sae_in=step_input.sae_in,
            feature_acts=feature_acts,
            temperature=self.cfg.infonce_temperature,
        )
        return losses

    @override
    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]:
        return {
            "infonce": TrainCoefficientConfig(
                value=self.cfg.infonce_loss_coefficient,
                warm_up_steps=self.cfg.infonce_warm_up_steps,
            ),
        }
