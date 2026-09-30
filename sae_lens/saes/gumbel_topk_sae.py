"""TopK SAE training variant with a learned, sample-conditioned selection head.

Adapted from SAMPLESELECT (Sample-Conditioned Representation Selection for
Audio Few-Shot Learning, https://arxiv.org/abs/2609.17076), which predicts a
fixed-budget feature mask independently for each input: training uses
differentiable Gumbel Top-k selection, inference uses deterministic Top-k.

TopK keeps the k largest pre-activations per sample and BatchTopK keeps k
features on average across the batch; this variant instead lets a small MLP
head score every feature per input and learns WHICH k features fire for that
input. The head is trained end-to-end by the SAE's own reconstruction and
auxiliary losses, standing in for the paper's frozen-encoder classification
and cross-background contrastive losses.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from typing_extensions import override

from sae_lens.saes.sae import SAEConfig, TrainStepInput, TrainStepOutput
from sae_lens.saes.topk_sae import TopKSAEConfig, TopKTrainingSAE, TopKTrainingSAEConfig


def _sample_gumbel(x: torch.Tensor) -> torch.Tensor:
    """Draw iid Gumbel(0, 1) noise with the shape (and dtype) of x."""
    uniform = torch.rand(x.shape, device=x.device, dtype=torch.float32)
    uniform.clamp_(1e-10, 1.0 - 1e-7)
    return (-uniform.log()).log().neg().to(x.dtype)


def topk_indicator(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Deterministic fixed-budget mask: 1 for the k largest scores per row."""
    indices = scores.topk(k, dim=-1, sorted=False).indices
    return torch.zeros_like(scores).scatter(-1, indices, 1.0)


def gumbel_topk_selection(
    scores: torch.Tensor, k: int, tau: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a fixed-budget subset mask via sequential Gumbel-softmax.

    Performs k draws without replacement over the last dimension: each draw
    perturbs the scores with Gumbel noise and takes a temperature-softmax, and
    already-selected mass is excluded before the next draw. Returns both a hard
    mask (the exact Gumbel Top-k sample, in {0, 1} with exactly k ones per row)
    and a soft mask in [0, 1] whose entries estimate the probability of each
    feature being selected, carrying gradients back to the scores.

    Args:
        scores: Selection scores of shape (..., d_sae).
        k: Number of features to select per row.
        tau: Gumbel-softmax temperature (lower is closer to hard top-k).
    """
    soft_mask = torch.zeros_like(scores)
    hard_mask = torch.zeros_like(scores)
    for _ in range(k):
        perturbed = scores + _sample_gumbel(scores)
        soft_mask = soft_mask + (1 - soft_mask) * torch.softmax(
            perturbed / tau, dim=-1
        )
        pick = torch.where(hard_mask > 0, -torch.inf, perturbed).argmax(dim=-1)
        hard_mask.scatter_(-1, pick.unsqueeze(-1), 1.0)
    return hard_mask, soft_mask


class GumbelTopKSelection(nn.Module):
    """Fixed-budget feature mask predicted from per-sample selection scores.

    In training mode the mask is a sampled Gumbel Top-k subset: the forward
    value is the exact-budget hard sample, and gradients flow through the soft
    Gumbel-softmax relaxation via a straight-through estimator (the same trick
    the JumpReLU SAE uses to train through its step activation). In eval mode
    the mask is the deterministic top-k of the scores.
    """

    def __init__(self, k: int, tau: float = 1.0):
        super().__init__()
        self.k = k
        self.tau = tau

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Return a mask of shape (..., d_sae) with an exact budget of k per row."""
        if not self.training:
            return topk_indicator(scores, self.k)
        hard_mask, soft_mask = gumbel_topk_selection(scores, self.k, self.tau)
        # Straight-through: keep the hard sample's values, the soft mask's gradients.
        return soft_mask + (hard_mask - soft_mask).detach()


@dataclass
class GumbelTopKTrainingSAEConfig(TopKTrainingSAEConfig):
    """
    Configuration class for training a GumbelTopKTrainingSAE.

    Like a TopK SAE, exactly k features are active per sample, but the active
    set is predicted per input by a learned selection head rather than being
    the k largest pre-activations.

    Args:
        selection_tau (float): Temperature of the Gumbel-softmax relaxation used
            to train the selection head. Lower temperatures track hard top-k
            more closely but give sparser gradients.
        selection_hidden_dim (int): Hidden width of the MLP selection head.
        k (int): Number of features to keep active per sample. Inherited from
            TopKTrainingSAEConfig.
        aux_loss_coefficient (float): Coefficient for the dead-neuron auxiliary
            loss. Inherited from TopKTrainingSAEConfig.
        rescale_acts_by_decoder_norm (bool): Treat the decoder as if it was
            already normalized. Inherited from TopKTrainingSAEConfig.
        use_sparse_activations (bool): Ignored; the selection path is always
            dense. Inherited from TopKTrainingSAEConfig.
        d_in (int): Input dimension (dimensionality of the activations being
            encoded). Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
        device (str): Device to place the SAE on. Inherited from SAEConfig.
    """

    selection_tau: float = 1.0
    selection_hidden_dim: int = 128

    @override
    @classmethod
    def architecture(cls) -> str:
        return "gumbel_topk"

    @override
    def get_inference_config_class(self) -> type[SAEConfig]:
        # The deterministic analog of this variant is a TopK SAE, but a learned
        # selection head cannot be round-tripped through that format (see
        # GumbelTopKTrainingSAE.save_inference_model). This mapping exists so
        # training metadata can be constructed for unregistered architectures.
        return TopKSAEConfig


class GumbelTopKTrainingSAE(TopKTrainingSAE):
    """
    TopK Training SAE whose active features are selected per sample by a
    learned head, trained with differentiable Gumbel Top-k selection.

    The parent class's magnitude-based TopK activation is replaced by
    relu(hidden_pre) * mask, where the mask comes from a small MLP scoring each
    feature for the current input. The inherited dead-neuron auxiliary loss and
    decoder-norm rescaling behave as in TopKTrainingSAE.

    This class is not registered in the SAE training registry by default. To
    drive it from the standard training runners, register it first:

        from sae_lens.registry import register_sae_training_class

        register_sae_training_class(
            "gumbel_topk", GumbelTopKTrainingSAE, GumbelTopKTrainingSAEConfig
        )
    """

    cfg: GumbelTopKTrainingSAEConfig  # type: ignore[assignment]
    selection_head: nn.Sequential

    def __init__(self, cfg: GumbelTopKTrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.selection = GumbelTopKSelection(k=cfg.k, tau=cfg.selection_tau)
        self.selection_head = nn.Sequential(
            nn.Linear(cfg.d_sae, cfg.selection_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.selection_hidden_dim, cfg.d_sae),
        ).to(device=self.device, dtype=self.dtype)

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate pre-activations, then keep the head-selected k features."""
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        selection_scores = self.selection_head(hidden_pre)
        feature_acts = self.hook_sae_acts_post(
            hidden_pre.relu() * self.selection(selection_scores)
        )
        return feature_acts, hidden_pre

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        output = super().training_forward_pass(step_input)
        output.metrics["mean_active_features"] = (
            (output.feature_acts != 0).sum(dim=-1).float().mean()
        )
        return output

    @override
    def save_inference_model(self, path: str | Path) -> tuple[Path, Path]:
        raise NotImplementedError(
            "GumbelTopKTrainingSAE cannot be exported to an inference SAE: the "
            "learned selection head picks a different feature subset per input, "
            "which no fixed-threshold inference architecture reproduces. Save "
            "the training checkpoint instead."
        )
