"""Top-AFA SAEs: per-input adaptive top-k selection via approximated quasi-orthogonality.

Adapted from "Evaluating and Designing Sparse Autoencoders by Approximating
Quasi-Orthogonality" (https://arxiv.org/abs/2503.24277). The paper shows that for an
epsilon-quasi-orthogonal dictionary the squared l2 norm of the sparse feature vector
approximates the squared l2 norm of the dense input. Top-AFA turns this into a
selection rule that replaces the fixed k of TopK SAEs: features are ranked by the
norm they contribute to a reconstruction, then kept until their cumulative squared
norm is closest to the squared norm of the input. The number of active features
(the l0) therefore adapts to each input instead of being a hyperparameter.
"""

from dataclasses import dataclass

import torch
from torch import nn
from typing_extensions import override

from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    TrainStepInput,
    TrainStepOutput,
)
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig


def topafa_activations(hidden_pre: torch.Tensor, sae_in: torch.Tensor) -> torch.Tensor:
    """
    Keep the features whose cumulative squared norm best matches the input norm.

    Features are ranked by squared activation (when hidden_pre was rescaled by decoder
    norms, this is the squared norm the feature contributes to a reconstruction with a
    unit-norm decoder), then kept until the cumulative squared norm is closest to the
    squared norm of sae_in. Under quasi-orthogonality these two norms coincide for the
    true sparse code, so the rule keeps just enough features to explain the input's
    norm. The last cumulative entry is replaced by an infinite sentinel so the full
    dictionary is never kept: with an exactly orthogonal decoder the full cumulative
    norm equals the input norm, which would degenerate to k = d_sae.

    Args:
        hidden_pre: Pre-activations of shape (..., d_sae), already rescaled by decoder
            norms if the SAE is configured that way.
        sae_in: The input consumed by the encoder, of shape (..., d_in).
    """
    acts = hidden_pre.relu()
    sorted_acts, sorted_indices = torch.sort(acts, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_acts.pow(2), dim=-1)
    cumulative[..., -1] = float("inf")
    input_norm_sq = sae_in.pow(2).sum(dim=-1, keepdim=True)
    k = (cumulative - input_norm_sq).abs().argmin(dim=-1) + 1
    keep_sorted = torch.arange(acts.shape[-1], device=acts.device) < k.unsqueeze(-1)
    keep = torch.zeros_like(acts, dtype=torch.bool).scatter(
        -1, sorted_indices, keep_sorted
    )
    return acts * keep


@dataclass
class TopAFASAEConfig(SAEConfig):
    """
    Configuration class for TopAFASAE inference.

    Top-AFA has no sparsity hyperparameters: the number of active features is
    determined per input by matching the cumulative feature norm to the input norm.

    Args:
        d_in (int): Input dimension (dimensionality of the activations being encoded).
            Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
        device (str): Device to place the SAE on. Inherited from SAEConfig.
        apply_b_dec_to_input (bool): Whether to apply decoder bias to the input
            before encoding. Inherited from SAEConfig.
        normalize_activations (Literal["none", "expected_average_only_in", "constant_norm_rescale", "layer_norm"]):
            Normalization strategy for input activations. Inherited from SAEConfig.
        reshape_activations (Literal["none", "hook_z"]): How to reshape activations
            (useful for attention head outputs). Inherited from SAEConfig.
        metadata (SAEMetadata): Metadata about the SAE (model name, hook name, etc.).
            Inherited from SAEConfig.
    """

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topafa"


class TopAFASAE(SAE[TopAFASAEConfig]):
    """
    An inference-only sparse autoencoder using the Top-AFA activation: the number of
    active features is chosen per input via approximated quasi-orthogonality instead
    of a fixed k.
    """

    b_enc: nn.Parameter

    def __init__(self, cfg: TopAFASAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

    @override
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts input x into feature activations, selecting how many features to
        keep per input by matching the cumulative feature norm to the input norm.
        """
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        return self.hook_sae_acts_post(topafa_activations(hidden_pre, sae_in))

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the input from the selected feature activations.
        """
        sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        raise NotImplementedError(
            "Folding W_dec_norm is not supported for TopAFASAE, as this would change "
            "the norm-matching feature selection"
        )


@dataclass
class TopAFATrainingSAEConfig(TopKTrainingSAEConfig):
    """
    Configuration class for training a TopAFATrainingSAE.

    Top-AFA SAEs replace the fixed k of TopK SAEs with a per-input selection rule:
    features are kept until their cumulative (decoder-scaled) squared norm best
    matches the squared norm of the input, an approximation of quasi-orthogonality.

    Args:
        afa_loss_coefficient (float): Coefficient for the norm-matching loss
            (||f||_2 - ||sae_in||_2)^2, which pulls the feature norm toward the input
            norm. The paper found 1/16 to be stable across layers and dictionary sizes.
        k (int): Unused; Top-AFA selects the number of active features per input.
            Inherited from TopKTrainingSAEConfig.
        aux_loss_coefficient (float): Coefficient for the auxiliary loss that
            encourages dead neurons to learn useful features. Inherited from
            TopKTrainingSAEConfig.
        rescale_acts_by_decoder_norm (bool): Treat the decoder as if it was already
            normalized. Inherited from TopKTrainingSAEConfig.
        use_sparse_activations (bool): Unused; Top-AFA always produces dense
            activations. Inherited from TopKTrainingSAEConfig.
        decoder_init_norm (float | None): Norm to initialize decoder weights to.
            Inherited from TrainingSAEConfig.
        d_in (int): Input dimension (dimensionality of the activations being encoded).
            Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
        device (str): Device to place the SAE on. Inherited from SAEConfig.
        apply_b_dec_to_input (bool): Whether to apply decoder bias to the input
            before encoding. Inherited from SAEConfig.
        normalize_activations (Literal["none", "expected_average_only_in", "constant_norm_rescale", "layer_norm"]):
            Normalization strategy for input activations. Inherited from SAEConfig.
        reshape_activations (Literal["none", "hook_z"]): How to reshape activations
            (useful for attention head outputs). Inherited from SAEConfig.
        metadata (SAEMetadata): Metadata about the SAE training (model name, hook name, etc.).
            Inherited from SAEConfig.
    """

    afa_loss_coefficient: float = 1 / 16

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topafa"


class TopAFATrainingSAE(TopKTrainingSAE):
    """
    Training SAE with per-input adaptive sparsity (Top-AFA).

    Instead of a fixed k, each input keeps the features whose cumulative
    (decoder-scaled) squared norm best matches the squared input norm, so the l0
    adapts to the input. Trained Top-AFA SAEs are saved as TopAFASAE, since the
    same selection rule is used at inference.
    """

    cfg: TopAFATrainingSAEConfig  # type: ignore[assignment]

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate pre-activations, then select activations with the Top-AFA
        norm-matching rule.
        """
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        feature_acts = self.hook_sae_acts_post(topafa_activations(hidden_pre, sae_in))
        return feature_acts, hidden_pre

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
        # With decoder-norm rescaling, feature_acts already carry the decoder norms,
        # i.e. they are the coefficients of the unit-norm dictionary.
        afa_acts = (
            feature_acts
            if self.cfg.rescale_acts_by_decoder_norm
            else feature_acts * self.W_dec.norm(dim=-1)
        )
        afa_loss = (
            (afa_acts.norm(dim=-1) - step_input.sae_in.norm(dim=-1)).pow(2).mean()
        )
        return {**aux_losses, "afa_loss": self.cfg.afa_loss_coefficient * afa_loss}

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        output = super().training_forward_pass(step_input)
        if step_input.is_logging_step:
            l0 = (output.feature_acts > 0).sum(-1).float()
            output.metrics["mean_l0"] = l0.mean()
            output.metrics["max_l0"] = l0.max()
            output.metrics["min_l0"] = l0.min()
        return output
