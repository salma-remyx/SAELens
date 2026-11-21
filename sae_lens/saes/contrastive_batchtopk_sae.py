"""BatchTopK SAE training with an InfoNCE contrastive loss on the sparse codes.

Adapted from "Explanations that Transfer: Interpreting Medical Retrieval with
Sparse AutoEncoders" (medsae), which combines a reconstruction objective with an
InfoNCE contrastive loss so that sparse codes preserve query-document relevance.
The total loss is the BatchTopK loss (MSE + topk auxiliary loss) plus a weighted
InfoNCE term, L = L_rec + lambda * L_NCE.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from typing_extensions import override

from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.sae import TrainStepInput


@dataclass
class ContrastiveBatchTopKTrainingSAEConfig(BatchTopKTrainingSAEConfig):
    """
    Configuration class for training a ContrastiveBatchTopKTrainingSAE.

    Training batches must be laid out as [queries; positive documents]: the first
    half of each batch holds query embeddings and the second half holds the
    positive document embedding for the query at the same offset. Use an even
    train_batch_size_tokens. The InfoNCE loss treats the other documents in the
    batch as negatives for each query.

    After training, ContrastiveBatchTopK SAEs are saved as JumpReLU SAEs.

    Args:
        contrastive_loss_coefficient (float): Weight of the InfoNCE loss relative to
            the reconstruction losses. The paper sweeps values in {0, 5e-6, 1e-5}.
        contrastive_temperature (float): Temperature for the cosine-similarity
            logits of the InfoNCE loss. The paper sweeps values in {0.01, 0.05, 0.1}.
        k (float): Average number of features to keep active across the batch.
            Inherited from BatchTopKTrainingSAEConfig.
        topk_threshold_lr (float): Learning rate for updating the global topk
            threshold. Inherited from BatchTopKTrainingSAEConfig.
        aux_loss_coefficient (float): Coefficient for the auxiliary loss that
            encourages dead neurons to learn useful features. Inherited from
            TopKTrainingSAEConfig.
        rescale_acts_by_decoder_norm (bool): Treat the decoder as if it was already
            normalized. Inherited from TopKTrainingSAEConfig.
        decoder_init_norm (float | None): Norm to initialize decoder weights to.
            Inherited from TrainingSAEConfig.
        d_in (int): Input dimension (dimensionality of the activations being encoded).
            Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
        device (str): Device to place the SAE on. Inherited from SAEConfig.
    """

    contrastive_loss_coefficient: float = 1e-5
    contrastive_temperature: float = 0.05

    @override
    @classmethod
    def architecture(cls) -> str:
        return "contrastive_batchtopk"


class ContrastiveBatchTopKTrainingSAE(BatchTopKTrainingSAE):
    """
    Global Batch TopK Training SAE with an InfoNCE contrastive loss.

    This SAE maintains k active features on average across the batch like a
    BatchTopK SAE, and additionally aligns each query's sparse code with its
    positive document's sparse code (using the other documents in the batch as
    negatives), so that the sparse codes preserve query-document relevance.
    """

    cfg: ContrastiveBatchTopKTrainingSAEConfig  # type: ignore[assignment]

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        losses = dict(
            super().calculate_aux_loss(
                step_input=step_input,
                feature_acts=feature_acts,
                hidden_pre=hidden_pre,
                sae_out=sae_out,
            )
        )
        losses["contrastive_loss"] = (
            self.cfg.contrastive_loss_coefficient
            * self.calculate_contrastive_loss(feature_acts)
        )
        return losses

    def calculate_contrastive_loss(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE loss on the sparse codes of a [queries; positive documents] batch.

        Codes are L2-normalized, so the logits are cosine similarities scaled by
        1 / contrastive_temperature, with the matched query-document pair on the
        diagonal. Rows with all-zero codes normalize to the zero vector and get
        similarity 0 with everything. Requires feature_acts of shape
        (2 * num_pairs, d_sae).
        """
        if feature_acts.shape[0] % 2 != 0:
            raise ValueError(
                "Training batches for ContrastiveBatchTopKTrainingSAE must contain "
                "an even number of rows (queries in the first half, their positive "
                f"documents in the second half), got {feature_acts.shape[0]} rows."
            )
        if feature_acts.is_sparse:
            feature_acts = feature_acts.to_dense()
        num_pairs = feature_acts.shape[0] // 2
        query_codes = F.normalize(feature_acts[:num_pairs], dim=-1)
        doc_codes = F.normalize(feature_acts[num_pairs:], dim=-1)
        logits = query_codes @ doc_codes.T / self.cfg.contrastive_temperature
        labels = torch.arange(num_pairs, device=logits.device)
        return F.cross_entropy(logits, labels)
