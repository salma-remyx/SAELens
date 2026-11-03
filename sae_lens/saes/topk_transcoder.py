"""TopK transcoders: hook-to-hook adapters with top-k latent selection.

These fill the TopK slot in the transcoder family (ReLU, JumpReLU and Skip
variants already existed), extending the repo's top-k selection chain
(BatchTopK -> TopK SAE -> TopK transcoder). Only the k largest latents are
active per token, so the latent set is exactly sparse and can be ablated
latent-by-latent without changing the behaviour of the survivors.

Adapted from TopKLoRA (Masiak et al., "Interpretable sparse fine-tuning with
TopKLoRA enables faithful circuit discovery", arXiv:2610.32866), whose adapter
update is y + B @ TopK(A @ x): only the k largest adapter latents are active at
each token. Here the skip connection plays the role of the base path and the
adapter is consumed through the standard transcoder load-and-run pipeline
rather than a task-loss fine-tuning loop.
"""

from dataclasses import dataclass
from typing import Any, Callable

import torch
from typing_extensions import override

from sae_lens.saes.sae import SAEMetadata
from sae_lens.saes.topk_sae import TopK
from sae_lens.saes.transcoder import SkipTranscoder, Transcoder, TranscoderConfig
from sae_lens.util import filter_valid_dataclass_fields


@dataclass
class TopKTranscoderConfig(TranscoderConfig):
    """
    Configuration for a TopK transcoder.

    Args:
        k (int): Number of latents kept active per token. Only the k latents
            with the largest pre-activations are non-zero; all others are zero.
        d_out (int): Output dimension (dimensionality of the target hook).
            Inherited from TranscoderConfig.
    """

    k: int = 100

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk_transcoder"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TopKTranscoderConfig":
        """Create a TopKTranscoderConfig from a dictionary."""
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)
        res = cls(**filtered_config_dict)

        if "metadata" in config_dict:
            res.metadata = SAEMetadata(**config_dict["metadata"])

        return res

    def __post_init__(self):
        if not 0 < self.k <= self.d_sae:
            raise ValueError(
                f"k must satisfy 0 < k <= d_sae to select the top k latents, "
                f"got k={self.k} with d_sae={self.d_sae}."
            )
        return super().__post_init__()


class TopKTranscoder(Transcoder):
    """
    A transcoder sparsified by top-k selection.

    Implements y = W_dec @ TopK(W_enc @ x + b_enc) + b_dec, so exactly k latents
    are active per token (fewer only if a selected pre-activation is negative,
    since TopK also applies ReLU) and the rest of the latent vector is zero.
    """

    cfg: TopKTranscoderConfig  # type: ignore[assignment]

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return TopK(self.cfg.k, use_sparse_activations=False)

    @override
    def fold_W_dec_norm(self) -> None:
        raise NotImplementedError(
            "Folding W_dec_norm is not safe for TopK transcoders: rescaling "
            "encoder columns by per-latent decoder norms changes which latents "
            "fall inside the top k, so the folded model is not equivalent."
        )


@dataclass
class TopKSkipTranscoderConfig(TopKTranscoderConfig):
    """Configuration for a TopK transcoder with a learnable skip connection."""

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk_skip_transcoder"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TopKSkipTranscoderConfig":
        """Create a TopKSkipTranscoderConfig from a dictionary."""
        filtered_config_dict = filter_valid_dataclass_fields(config_dict, cls)
        res = cls(**filtered_config_dict)

        if "metadata" in config_dict:
            res.metadata = SAEMetadata(**config_dict["metadata"])

        return res


class TopKSkipTranscoder(TopKTranscoder, SkipTranscoder):
    """
    A TopK transcoder in residual (skip) form.

    Implements y = W_dec @ TopK(W_enc @ x + b_enc) + W_skip @ x + b_dec, the
    transcoder analogue of a sparse adapter riding on a base path: the k active
    latents carry the learned delta on top of the skip connection.
    """

    cfg: TopKSkipTranscoderConfig  # type: ignore[assignment]

    def __init__(self, cfg: TopKSkipTranscoderConfig):
        super().__init__(cfg)
