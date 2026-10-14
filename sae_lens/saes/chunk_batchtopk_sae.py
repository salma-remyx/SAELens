"""Chunk-level BatchTopK SAEs.

Adapted from "Beyond Token Scale: Chunk-Level Sparse Autoencoders for Reliable
Semantic Feature Discovery" (https://arxiv.org/abs/2609.35521). Token-level
objectives spend the sparse budget on lexical and formatting details alongside
semantic content; encoding mean-pooled activations over contiguous chunks of
tokens instead reserves the budget for higher-level concepts.

To train one with the standard runners, register the architecture once per
process via the public extension point:

    register_sae_training_class(
        "chunk_batchtopk",
        ChunkBatchTopKTrainingSAE,
        ChunkBatchTopKTrainingSAEConfig,
    )

after which `architecture="chunk_batchtopk"` resolves to this class anywhere
training SAEs are built from configs (`TrainingSAE.from_dict`, the config
loaders, and the training runners).

Adaptation note: in the paper the neighbor chunk of a Cross/Joint-Chunk SAE is
produced by an independent forward pass of the model. The activations contract
here (`TrainStepInput.sae_in`) delivers one flat stream of token activations per
step, so the neighbor is approximated by the adjacent contiguous chunk of that
stream. Pooling and the three training objectives are otherwise as described.
"""

from dataclasses import dataclass, replace
from typing import Literal

import torch
from typing_extensions import override

from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.saes.sae import TrainStepInput, TrainStepOutput

ChunkVariant = Literal["mean", "cross", "joint"]


def mean_pool_chunks(sae_in: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Mean-pool a flat stream of token activations into contiguous chunks.

    Takes token activations of shape (n_tokens, d_in) and average-pools every
    contiguous span of `chunk_size` tokens, yielding shape
    (n_tokens // chunk_size, d_in). Trailing tokens that do not fill a whole
    chunk are dropped.
    """
    n_chunks = sae_in.shape[0] // chunk_size
    chunked = sae_in[: n_chunks * chunk_size].reshape(
        n_chunks, chunk_size, sae_in.shape[-1]
    )
    return chunked.mean(dim=1)


@dataclass
class ChunkBatchTopKTrainingSAEConfig(BatchTopKTrainingSAEConfig):
    """
    Configuration class for training a ChunkBatchTopKTrainingSAE.

    Chunk-level SAEs encode mean-pooled activations over contiguous chunks of
    tokens rather than individual token activations, so the limited sparse
    budget is spent on semantic content instead of lexical and formatting
    details (see module docstring for the paper reference and the adaptation
    made to the neighbor-chunk target).

    After training, ChunkBatchTopK SAEs are saved as JumpReLU SAEs, exactly
    like standard BatchTopK SAEs.

    Args:
        chunk_size (int): Number of tokens per chunk. The training stream is
            mean-pooled over contiguous spans of this many tokens before the
            SAE sees it, so the effective number of samples per batch becomes
            train_batch_size_tokens // chunk_size.
        chunk_variant (Literal["mean", "cross", "joint"]): Which chunk-level
            objective to train. "mean" reconstructs the observed chunk itself.
            "cross" encodes each chunk but reconstructs its neighbor chunk,
            keeping only information shared across chunks. "joint" trains both
            targets together.
        k (float): Average number of features to keep active across the batch.
            Inherited from BatchTopKTrainingSAEConfig.
        topk_threshold_lr (float): Learning rate for updating the global topk threshold.
            Inherited from BatchTopKTrainingSAEConfig.
        aux_loss_coefficient (float): Coefficient for the auxiliary loss that encourages
            dead neurons to learn useful features. Inherited from TopKTrainingSAEConfig.
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

    chunk_size: int = 32
    chunk_variant: ChunkVariant = "mean"

    @override
    @classmethod
    def architecture(cls) -> str:
        return "chunk_batchtopk"


class ChunkBatchTopKTrainingSAE(BatchTopKTrainingSAE):
    """
    Chunk-level Batch TopK Training SAE.

    Mean-pools the incoming token-activation stream into contiguous chunks and
    trains a BatchTopK SAE on the pooled chunks. The chunk_variant config
    selects the paper's three objectives: Mean-Chunk (reconstruct the observed
    chunk), Cross-Chunk (reconstruct the neighbor chunk), and Joint-Chunk (both).

    ChunkBatchTopK SAEs are saved as JumpReLU SAEs after training.
    """

    cfg: ChunkBatchTopKTrainingSAEConfig

    def __init__(
        self, cfg: ChunkBatchTopKTrainingSAEConfig, use_error_term: bool = False
    ):
        super().__init__(cfg, use_error_term)
        _validate_chunk_config(cfg)

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        pooled = mean_pool_chunks(step_input.sae_in, self.cfg.chunk_size)
        min_chunks = 1 if self.cfg.chunk_variant == "mean" else 2
        if pooled.shape[0] < min_chunks:
            raise ValueError(
                f"chunk_variant={self.cfg.chunk_variant!r} needs at least "
                f"{min_chunks} pooled chunks per batch, but the batch of "
                f"{step_input.sae_in.shape[0]} tokens with chunk_size="
                f"{self.cfg.chunk_size} only yields {pooled.shape[0]}. "
                "Increase train_batch_size_tokens or decrease chunk_size."
            )

        if self.cfg.chunk_variant == "mean":
            # the observation unit is the only thing that changes: everything
            # downstream (MSE, aux loss, threshold update) runs on pooled chunks
            return super().training_forward_pass(replace(step_input, sae_in=pooled))

        # cross / joint: encode each chunk, predict its neighbor chunk
        source = pooled[:-1]
        target = pooled[1:]
        feature_acts, hidden_pre = self.encode_with_hidden_pre(source)
        sae_out = self.decode(feature_acts)

        losses: dict[str, torch.Tensor] = {}
        cross_chunk_mse_loss = self.mse_loss_fn(sae_out, target).sum(dim=-1).mean()
        losses["cross_chunk_mse_loss"] = cross_chunk_mse_loss
        total_loss = cross_chunk_mse_loss
        if self.cfg.chunk_variant == "joint":
            mean_chunk_mse_loss = self.mse_loss_fn(sae_out, source).sum(dim=-1).mean()
            losses["mean_chunk_mse_loss"] = mean_chunk_mse_loss
            total_loss = total_loss + mean_chunk_mse_loss

        # dead latents chase the residual of the prediction target, matching the
        # objective the live latents are trained on
        aux_losses = self.calculate_aux_loss(
            step_input=replace(step_input, sae_in=target),
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            sae_out=sae_out,
        )
        for name, aux_loss in aux_losses.items():
            losses[name] = aux_loss
            total_loss = total_loss + aux_loss

        output = TrainStepOutput(
            sae_in=source,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=total_loss,
            losses=losses,
        )
        self.update_topk_threshold(feature_acts)
        output.metrics["topk_threshold"] = self.topk_threshold
        return output


def _validate_chunk_config(cfg: ChunkBatchTopKTrainingSAEConfig) -> None:
    if cfg.chunk_size < 1:
        raise ValueError(f"cfg.chunk_size must be positive, got {cfg.chunk_size}.")
    variants: tuple[ChunkVariant, ...] = ("mean", "cross", "joint")
    if cfg.chunk_variant not in variants:
        raise ValueError(
            f"cfg.chunk_variant must be one of {variants}, got {cfg.chunk_variant!r}."
        )
