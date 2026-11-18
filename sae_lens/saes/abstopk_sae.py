"""AbsTopK SAE variant, which selects the k largest-magnitude activations.

Adapted from "AbsTopK: Rethinking Sparse Autoencoders For Bidirectional Features"
(https://arxiv.org/abs/2510.00404). TopK-style SAEs keep the k largest
pre-activations and ReLU them, which forces every feature activation to be
non-negative. This fragments bidirectional concepts (e.g. male vs. female) into
separate features. AbsTopK instead applies the L0 hard-thresholding on the
largest-magnitude pre-activations while preserving their signs, so a single
feature can encode both directions of a concept.
"""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from typing_extensions import override

from sae_lens.saes.topk_sae import (
    TopKSAE,
    TopKSAEConfig,
    TopKTrainingSAE,
    TopKTrainingSAEConfig,
)


class AbsTopK(nn.Module):
    """
    An activation that zeroes out all but the k elements with the largest
    magnitude along the last dimension, keeping their original signs.
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        1) Select the k elements with the largest absolute value along the last dim.
        2) Keep their original (possibly negative) values.
        3) Zero out all other entries.
        """
        topk_indices = torch.topk(x.abs(), k=self.k, dim=-1, sorted=False).indices
        values = torch.gather(x, -1, topk_indices)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_indices, values)
        return result


@dataclass
class AbsTopKSAEConfig(TopKSAEConfig):
    """
    Configuration class for AbsTopKSAE inference.

    Args:
        k (int): Number of top features to keep active during inference. The k
            features with the largest-magnitude pre-activations are kept, with
            their signs.
        rescale_acts_by_decoder_norm (bool): Whether to treat the decoder as if
            it was already normalized. Inherited from TopKSAEConfig.
        d_in (int): Input dimension. Inherited from SAEConfig.
        d_sae (int): SAE latent dimension. Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
        device (str): Device to place the SAE on. Inherited from SAEConfig.
        apply_b_dec_to_input (bool): Whether to apply decoder bias to the input
            before encoding. Inherited from SAEConfig.
        normalize_activations: Normalization strategy for input activations.
            Inherited from SAEConfig.
        reshape_activations: How to reshape activations. Inherited from SAEConfig.
        metadata (SAEMetadata): Metadata about the SAE. Inherited from SAEConfig.
    """

    @override
    @classmethod
    def architecture(cls) -> str:
        return "abstopk"


class AbsTopKSAE(TopKSAE):
    """
    An inference-only sparse autoencoder using an "abstopk" activation function.
    Identical to TopKSAE except that the k largest-magnitude pre-activations are
    kept with their signs, rather than the k largest pre-activations after ReLU.
    """

    cfg: AbsTopKSAEConfig  # type: ignore[assignment]

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return AbsTopK(self.cfg.k)


@dataclass
class AbsTopKTrainingSAEConfig(TopKTrainingSAEConfig):
    """
    Configuration class for training an AbsTopKTrainingSAE.

    Args:
        k (int): Number of top features to keep active. The k features with the
            largest-magnitude pre-activations are kept, with their signs.
        aux_loss_coefficient (float): Coefficient for the auxiliary loss that
            encourages dead neurons to learn useful features. Inherited from
            TopKTrainingSAEConfig.
        rescale_acts_by_decoder_norm (bool): Treat the decoder as if it was
            already normalized. Inherited from TopKTrainingSAEConfig.
        use_sparse_activations (bool): Inherited from TopKTrainingSAEConfig but
            not used by this architecture; activations are always dense.
        decoder_init_norm (float | None): Norm to initialize decoder weights to.
            Inherited from TrainingSAEConfig.
        d_in (int): Input dimension. Inherited from SAEConfig.
        d_sae (int): SAE latent dimension. Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
        device (str): Device to place the SAE on. Inherited from SAEConfig.
        apply_b_dec_to_input (bool): Whether to apply decoder bias to the input
            before encoding. Inherited from SAEConfig.
        normalize_activations: Normalization strategy for input activations.
            Inherited from SAEConfig.
        reshape_activations: How to reshape activations. Inherited from SAEConfig.
        metadata (SAEMetadata): Metadata about the SAE training. Inherited from
            TrainingSAEConfig.
    """

    @override
    @classmethod
    def architecture(cls) -> str:
        return "abstopk"


class AbsTopKTrainingSAE(TopKTrainingSAE):
    """
    Training variant of the AbsTopK SAE. Uses the same dead-neuron auxiliary
    loss as TopK SAEs, except that dead latents are selected by activation
    magnitude (with their signs) to match the AbsTopK activation.
    """

    cfg: AbsTopKTrainingSAEConfig  # type: ignore[assignment]

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return AbsTopK(self.cfg.k)

    @override
    def _select_dead_aux_acts(
        self,
        k_aux: int,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor,
    ) -> torch.Tensor:
        return calculate_abstopk_aux_acts(
            k_aux=k_aux,
            hidden_pre=hidden_pre,
            dead_neuron_mask=dead_neuron_mask,
        )


def calculate_abstopk_aux_acts(
    k_aux: int,
    hidden_pre: torch.Tensor,
    dead_neuron_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Select the top k_aux dead neurons by activation magnitude, keeping signs.

    Args:
        k_aux: Number of top dead neurons to select
        hidden_pre: Pre-activation values from encoder, shape (batch_size, d_sae)
        dead_neuron_mask: Boolean mask indicating which neurons are dead

    Returns:
        Tensor with the signed activations of the k_aux largest-magnitude dead
        neurons, zeros elsewhere
    """
    # Don't include living latents in this loss
    auxk_magnitudes = torch.where(
        dead_neuron_mask[None], hidden_pre.abs(), -torch.inf
    )
    # Top-k_aux dead latents by magnitude
    topk = auxk_magnitudes.topk(k_aux, sorted=False)
    # Keep the original signed values for the selected latents
    auxk_acts = torch.zeros_like(hidden_pre)
    auxk_acts.scatter_(-1, topk.indices, torch.gather(hidden_pre, -1, topk.indices))
    return auxk_acts
