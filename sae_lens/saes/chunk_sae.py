"""Chunk-level sparse autoencoder (Mean-Chunk variant).

Encodes mean-pooled activations over chunks, each a contiguous span of
tokens, and reconstructs the observed chunk mean. Because the objective
only sees what is shared across a span, the sparse budget is spent on
high-level content rather than per-token lexical and formatting detail.

Adapted from "Beyond Token Scale: Chunk-Level Sparse Autoencoders for
Reliable Semantic Feature Discovery" (https://arxiv.org/abs/2609.35521).
Only the Mean-Chunk variant is implemented here: the Cross-Chunk and
Joint-Chunk variants change the prediction target to a neighbor chunk and
need paired-chunk batches, which the activation store does not provide.

The sparse coding itself is inherited unchanged from BatchTopKTrainingSAE
(batch-level top-k, optional sparse decoding, decoder-norm rescaling), so
chunk_size=1 degenerates exactly to the token-level BatchTopK SAE. After
training, the SAE is saved as a JumpReLU SAE over chunk means; pool inputs
with chunk_means before encoding them with the saved inference SAE.
"""

from dataclasses import dataclass, replace

import einops
import torch
from typing_extensions import override

from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.sae import (
    TrainStepInput,
    TrainStepOutput,
    _disable_hooks,
)


def chunk_means(acts: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Mean-pool activations over contiguous chunks of tokens.

    Takes activations of shape (..., n_tokens, d_in), where the leading
    dimensions index chunks (or batches of sequences), and returns the mean
    over each contiguous span of chunk_size tokens, with shape
    (..., n_tokens // chunk_size, d_in).

    Args:
        acts: Activations of shape (..., n_tokens, d_in) with positions in
            contiguous order.
        chunk_size: Number of contiguous tokens per chunk.

    Returns:
        Chunk mean activations of shape (..., n_tokens // chunk_size, d_in).
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1, got {chunk_size}")
    if acts.ndim < 3:
        raise ValueError(
            "Chunk-level SAEs need a token dimension to pool over: expected "
            "activations of shape (..., n_tokens, d_in) such as "
            "(n_chunks, chunk_size, d_in) or (batch, seq_len, d_in), got "
            f"{tuple(acts.shape)}. The training activation store yields flat "
            "shuffled tokens, so chunked activations must be built upstream."
        )
    n_tokens = acts.shape[-2]
    if n_tokens % chunk_size != 0:
        raise ValueError(
            f"Number of tokens {n_tokens} must be divisible by chunk_size "
            f"{chunk_size} to form complete chunks."
        )
    return einops.rearrange(
        acts, "... (n_chunk c) d -> ... n_chunk c d", c=chunk_size
    ).mean(dim=-2)


@dataclass
class ChunkTrainingSAEConfig(BatchTopKTrainingSAEConfig):
    """
    Configuration class for training a ChunkTrainingSAE (Mean-Chunk).

    A ChunkTrainingSAE mean-pools activations over contiguous spans of
    chunk_size tokens, encodes the chunk mean, and reconstructs it. All
    sparse-coding options are inherited from BatchTopKTrainingSAEConfig.

    Args:
        chunk_size (int): Number of contiguous tokens per chunk. The encoder
            sees one mean-pooled activation per chunk, and 1 recovers the
            token-level BatchTopK SAE exactly.
        k (float): Average number of features to keep active across the
            batch, per chunk. Inherited from BatchTopKTrainingSAEConfig.
        topk_threshold_lr (float): Learning rate for updating the global
            topk threshold. Inherited from BatchTopKTrainingSAEConfig.
        aux_loss_coefficient (float): Coefficient for the auxiliary loss
            that encourages dead neurons to learn useful features.
            Inherited from TopKTrainingSAEConfig.
        rescale_acts_by_decoder_norm (bool): Treat the decoder as if it was
            already normalized. Inherited from TopKTrainingSAEConfig.
        use_sparse_activations (bool): Whether to use sparse tensor
            representations for activations during training. Inherited from
            TopKTrainingSAEConfig.
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

    chunk_size: int = 8

    @override
    @classmethod
    def architecture(cls) -> str:
        return "chunk"


class ChunkTrainingSAE(BatchTopKTrainingSAE):
    """
    Mean-Chunk training SAE: encode chunk means, reconstruct the chunk mean.

    Inputs to forward, encode and training_forward_pass are chunked
    activations of shape (..., n_tokens, d_in) with positions in contiguous
    order, such as (n_chunks, chunk_size, d_in) or (batch, seq_len, d_in).
    They are mean-pooled over spans of chunk_size tokens before the usual
    BatchTopK encode/decode, so the reconstruction target is the chunk mean
    rather than individual token activations.

    With chunk_size=1 this is exactly the token-level BatchTopK SAE.
    """

    cfg: ChunkTrainingSAEConfig  # type: ignore[assignment]

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        chunked_input = replace(
            step_input,
            sae_in=chunk_means(step_input.sae_in, self.cfg.chunk_size),
        )
        return super().training_forward_pass(chunked_input)

    @override
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return super().encode(chunk_means(x, self.cfg.chunk_size))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = chunk_means(x, self.cfg.chunk_size)
        # super().encode instead of self.encode: pooled is already in
        # chunk-mean space, and self.encode would pool it a second time.
        feature_acts = super().encode(pooled)
        sae_out = self.decode(feature_acts)

        if self.use_error_term:
            with torch.no_grad():
                # Recompute without hooks for true error term, in chunk-mean
                # space so the subtraction is well defined.
                with _disable_hooks(self):
                    feature_acts_clean = super().encode(pooled)
                    x_reconstruct_clean = self.decode(feature_acts_clean)
                sae_error = self.hook_sae_error(pooled - x_reconstruct_clean)
            sae_out = sae_out + sae_error

        return self.hook_sae_output(sae_out)
