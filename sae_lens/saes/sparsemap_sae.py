"""TopK SAEs whose feature selection is gated by a sparse probability map.

Adapted from "How Sparse Probability Maps Shape Mixture-of-Experts Routing"
(https://arxiv.org/abs/2610.06677). That paper studies MoE routers that apply a
sparsity-inducing probability map (sparsemax, alpha-entmax, normmax) to the
top-K selected router scores instead of softmax. Such maps can assign exact
zeros to selected experts, so expert participation becomes token-dependent
instead of exactly K per token, and trained routers are far less sensitive to
raising K at inference (sparsemax trained with K=2 loses 0.02 nats when run
with K=8, where softmax loses 0.58).

SparseMapTopK applies the same mechanism to the TopK SAE activation step
(`TopK` in `sae_lens.saes.topk_sae`): the top-k pre-activations are selected as
usual, a sparse probability map is computed over the selected scores, and any
selected feature whose probability is exactly zero is dropped. Surviving
features keep their ReLU pre-activation values, so:

- when the map drops nothing, the activation is identical to plain TopK.
- the number of active features per sample adapts to the score gaps (a feature
  is dropped only when its gap to the higher selected scores reaches the map's
  fixed threshold), so participation depends jointly on the map and the learned
  score distribution. Use `participation_stats` to track this during training.
- raising k at inference leaves the activation unchanged as long as the extra
  selected scores stay below the map's threshold.

To train with the standard SAELens runner, register the classes (see
docs/custom_saes.md) and use the config like any custom architecture:

```python
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    register_sae_class,
    register_sae_training_class,
)
from sae_lens.saes.sparsemap_sae import (
    SparseMapTopKSAE,
    SparseMapTopKSAEConfig,
    SparseMapTopKTrainingSAE,
    SparseMapTopKTrainingSAEConfig,
)

register_sae_class("sparsemap_topk", SparseMapTopKSAE, SparseMapTopKSAEConfig)
register_sae_training_class(
    "sparsemap_topk", SparseMapTopKTrainingSAE, SparseMapTopKTrainingSAEConfig
)

cfg = LanguageModelSAERunnerConfig(
    sae=SparseMapTopKTrainingSAEConfig(
        k=32, probability_map="sparsemax", d_in=768, d_sae=16384
    ),
    model_name="gpt2",
    hook_name="blocks.3.hook_resid_post",
    dataset_path="monology/pile-uncopyrighted",
)
LanguageModelSAETrainingRunner(cfg).run()
```
"""

from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import nn
from typing_extensions import override

from sae_lens.saes.sae import SAEConfig, TrainStepInput, TrainStepOutput
from sae_lens.saes.topk_sae import (
    TopKSAE,
    TopKSAEConfig,
    TopKTrainingSAE,
    TopKTrainingSAEConfig,
)


def sparsemax(v: torch.Tensor) -> torch.Tensor:
    """
    sparsemax probability map over the last dimension of (..., num_entries).

    sparsemax (Martins & Astudillo, 2016) is the Euclidean projection of v onto
    the probability simplex, and assigns exact zeros to every entry whose gap
    to the higher entries reaches the projection threshold.
    """
    v = v - v.max(dim=-1, keepdim=True).values
    sorted_v, _ = torch.sort(v, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_v, dim=-1) - 1
    k = torch.arange(1, v.shape[-1] + 1, device=v.device)
    support = sorted_v * k > cumsum
    k_star = support.sum(dim=-1, keepdim=True)
    tau = cumsum.gather(-1, k_star - 1) / k_star
    return (v - tau).clamp_min(0.0)


def entmax15(v: torch.Tensor) -> torch.Tensor:
    """
    1.5-entmax probability map over the last dimension of (..., num_entries).

    1.5-entmax (Peters et al., 2019) is the Tsallis-entropy regularized
    projection onto the probability simplex with alpha=1.5. It is sparser than
    softmax but keeps more entries (and more probability mass) than sparsemax.
    """
    v = v - v.max(dim=-1, keepdim=True).values
    sorted_v, _ = torch.sort(v, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_v, dim=-1)
    cumsum_sq = torch.cumsum(sorted_v.pow(2), dim=-1)
    k = torch.arange(1, v.shape[-1] + 1, device=v.device, dtype=v.dtype)
    # On the top-k support, the threshold tau solves sum_i (v_i - tau)^2 = 1,
    # whose smaller root is tau = (T - sqrt(T^2 - k * (Q - 1))) / k with
    # T = sum v_i and Q = sum v_i^2 over the support.
    discriminant = (cumsum.pow(2) - k * (cumsum_sq - 1)).clamp_min(0.0)
    tau = (cumsum - discriminant.sqrt()) / k
    support = sorted_v > tau
    k_star = support.sum(dim=-1, keepdim=True)
    tau_star = tau.gather(-1, k_star - 1)
    p = (v - tau_star).clamp_min(0.0)
    return p * p


PROBABILITY_MAPS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "sparsemax": sparsemax,
    "entmax15": entmax15,
}


class SparseMapTopK(nn.Module):
    """
    TopK activation gated by a sparse probability map.

    Selects the top k pre-activations as in TopK, computes the chosen
    probability map over the selected scores, and drops the selected features
    whose probability is exactly zero (the map's adaptive zeros). Surviving
    features keep their ReLU values, so the number of active features per
    sample is adaptive up to k rather than exactly k.
    """

    def __init__(
        self,
        k: int,
        probability_map: Literal["sparsemax", "entmax15"] = "sparsemax",
    ):
        super().__init__()
        self.k = k
        self.probability_map = probability_map

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gate the top-k selection with a sparse probability map.

        1) Select top k elements along the last dimension.
        2) Compute the probability map over the selected scores.
        3) Keep the ReLU values of selected features with nonzero probability,
           zero out everything else.
        """
        topk_values, topk_indices = torch.topk(x, k=self.k, dim=-1, sorted=False)
        probs = PROBABILITY_MAPS[self.probability_map](topk_values)
        values = topk_values.relu() * (probs > 0).to(topk_values.dtype)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_indices, values)
        return result


@dataclass
class SparseMapTopKSAEConfig(TopKSAEConfig):
    """
    Configuration class for SparseMapTopKSAE inference.

    Args:
        probability_map (Literal["sparsemax", "entmax15"]): Sparse probability map
            applied to the top-k selected pre-activations. "sparsemax" drops
            selected features most aggressively; "entmax15" keeps more of them.
        k (int): Number of top features selected before the map drops any.
            Raising this at inference is safe as long as the extra selected
            scores stay below the map's threshold.
        rescale_acts_by_decoder_norm (bool): Whether to rescale pre-activations
            by decoder norms. Inherited from TopKSAEConfig.
        d_in (int): Input dimension. Inherited from SAEConfig.
        d_sae (int): SAE latent dimension. Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
        device (str): Device to place the SAE on. Inherited from SAEConfig.
    """

    probability_map: Literal["sparsemax", "entmax15"] = "sparsemax"

    @override
    @classmethod
    def architecture(cls) -> str:
        return "sparsemap_topk"


class SparseMapTopKSAE(TopKSAE):
    """
    An inference sparse autoencoder using the SparseMapTopK activation.

    Behaves identically to TopKSAE for samples where the probability map
    assigns nonzero probability to every selected feature, and adaptively
    zeroes selected features with larger score gaps otherwise.
    """

    cfg: SparseMapTopKSAEConfig  # type: ignore[assignment]

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return SparseMapTopK(self.cfg.k, self.cfg.probability_map)


@dataclass
class SparseMapTopKTrainingSAEConfig(TopKTrainingSAEConfig):
    """
    Configuration class for training a SparseMapTopKTrainingSAE.

    Args:
        probability_map (Literal["sparsemax", "entmax15"]): Sparse probability map
            applied to the top-k selected pre-activations. "sparsemax" drops
            selected features most aggressively; "entmax15" keeps more of them.
        k (int): Number of top features to select per sample before the map may
            drop any of them. Inherited from TopKTrainingSAEConfig.
        aux_loss_coefficient (float): Coefficient for the dead-neuron auxiliary
            loss. Inherited from TopKTrainingSAEConfig.
        rescale_acts_by_decoder_norm (bool): Treat the decoder as if it was
            already normalized. Inherited from TopKTrainingSAEConfig.
        decoder_init_norm (float | None): Norm to initialize decoder weights to.
            Inherited from TrainingSAEConfig.
        d_in (int): Input dimension. Inherited from SAEConfig.
        d_sae (int): SAE latent dimension. Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
        device (str): Device to place the SAE on. Inherited from SAEConfig.
    """

    probability_map: Literal["sparsemax", "entmax15"] = "sparsemax"

    @override
    @classmethod
    def architecture(cls) -> str:
        return "sparsemap_topk"

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        return SparseMapTopKSAEConfig

    def __post_init__(self):
        super().__post_init__()
        if self.use_sparse_activations:
            raise ValueError(
                "use_sparse_activations is not supported for SparseMapTopK SAEs; "
                "the map gate is always applied to dense top-k values."
            )


class SparseMapTopKTrainingSAE(TopKTrainingSAE):
    """
    Training SAE that selects features with a sparse probability map.

    Inherits the full TopK training machinery (decoder-norm rescaling, the
    dead-neuron auxiliary loss) and replaces the fixed TopK gate with
    SparseMapTopK. The mean number of active features is added to the training
    metrics so participation can be tracked as the learned score distribution
    co-adapts with the map.
    """

    cfg: SparseMapTopKTrainingSAEConfig  # type: ignore[assignment]

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return SparseMapTopK(self.cfg.k, self.cfg.probability_map)

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        output = super().training_forward_pass(step_input)
        output.metrics["mean_active_features"] = (
            (output.feature_acts > 0).sum(dim=-1).float().mean().item()
        )
        return output


def participation_stats(feature_acts: torch.Tensor, k: int) -> dict[str, float]:
    """
    Summarize per-sample feature participation of a (num_samples, d_sae)
    activation tensor, mirroring the expert-participation analysis of the
    sparse-routing paper this module is adapted from.

    - mean_active_features: average number of active features per sample
      (equals k for a plain TopK SAE, and is lower once the map drops features).
    - frac_samples_below_budget: fraction of samples where at least one of the
      k selected features was dropped by the map.
    - frac_single_feature_samples: fraction of samples routed to a single
      feature.

    Args:
        feature_acts: Feature activations of shape (num_samples, d_sae).
        k: The selection budget the SAE was configured with.
    """
    active_per_sample = (feature_acts > 0).sum(dim=-1).float()
    return {
        "mean_active_features": active_per_sample.mean().item(),
        "frac_samples_below_budget": (active_per_sample < k).float().mean().item(),
        "frac_single_feature_samples": (active_per_sample == 1).float().mean().item(),
    }
