"""Top-AFA SAEs: per-input adaptive top-k selection via approximated quasi-orthogonality.

Adapted from "Evaluating and Designing Sparse Autoencoders by Approximating
Quasi-Orthogonality" (https://arxiv.org/abs/2503.24277). The paper shows that for an
epsilon-quasi-orthogonal dictionary the squared l2 norm of the sparse feature vector
approximates the squared l2 norm of the dense input. Top-AFA turns this into a
selection rule that replaces the fixed k of TopK SAEs: features are ranked by the
norm they contribute to a reconstruction, then kept until their cumulative norm is
closest to the norm of the input. The number of active features (the l0) therefore
adapts to each input instead of being a hyperparameter.
"""

from dataclasses import dataclass
from typing import Literal

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


def topafa_activations(
    hidden_pre: torch.Tensor,
    W_dec: torch.Tensor,
    x_cent: torch.Tensor,
) -> torch.Tensor:
    """
    Keep the features whose cumulative decoder-scaled norm best matches the input norm.

    Features are ranked by the squared norm they contribute to a reconstruction with
    the current decoder, (act * ||W_dec[feature]||)^2, and kept until their
    cumulative norm is closest to the norm of the centered input. As in the reference
    implementation, the match happens in norm space: the argmin runs over the square
    roots of the cumulative squared norms and of the input squared norm (matching in
    squared-norm space picks a different k on some inputs, because the tie-break
    midpoints differ). The last cumulative entry is replaced by a large sentinel so
    the full dictionary is effectively never kept: with an exactly orthogonal decoder
    the full cumulative norm equals the input norm, which would degenerate to
    k = d_sae.

    Args:
        hidden_pre: Pre-activations of shape (..., d_sae).
        W_dec: Decoder weights of shape (d_sae, d_in); only their row norms are used.
        x_cent: Encoder input of shape (..., d_in), already centered with b_dec.
    """
    acts = hidden_pre.relu()
    dec_scaled_acts = (acts * W_dec.norm(dim=-1)).pow(2)
    sorted_acts, sorted_indices = torch.sort(dec_scaled_acts, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_acts, dim=-1)
    # Ensure the last cumulative norm is large enough that the full dictionary is
    # effectively never selected.
    cumulative[..., -1] = 1e8
    k = (
        (cumulative.sqrt() - x_cent.norm(dim=-1, keepdim=True))
        .abs()
        .argmin(dim=-1)
        + 1
    )
    keep_sorted = torch.arange(acts.shape[-1], device=acts.device) < k.unsqueeze(-1)
    keep = torch.zeros_like(acts, dtype=torch.bool).scatter(
        -1, sorted_indices, keep_sorted
    )
    return acts * keep


def _center_sae_in(sae: SAE[SAEConfig], x: torch.Tensor) -> torch.Tensor:
    """
    The input as seen by the Top-AFA encoder: processed, then always centered with
    b_dec. The reference subtracts b_dec unconditionally before computing both the
    encoder input and the selection target, while process_sae_in only subtracts it
    when apply_b_dec_to_input is set.
    """
    sae_in = sae.process_sae_in(x)
    return sae_in if sae.cfg.apply_b_dec_to_input else sae_in - sae.b_dec


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
        apply_b_dec_to_input (bool): Unused; Top-AFA always centers the input with
            b_dec before encoding, as in the reference. Inherited from SAEConfig.
        normalize_activations (Literal["none", "expected_average_only_in", "constant_norm_rescale", "layer_norm"]):
            Normalization strategy for input activations. Defaults to "layer_norm",
            the per-input standardization the reference applies under
            input_unit_norm=True. Inherited from SAEConfig.
        reshape_activations (Literal["none", "hook_z"]): How to reshape activations
            (useful for attention head outputs). Inherited from SAEConfig.
        metadata (SAEMetadata): Metadata about the SAE (model name, hook name, etc.).
            Inherited from SAEConfig.
    """

    normalize_activations: Literal[
        "none",
        "expected_average_only_in",
        "constant_norm_rescale",
        "layer_norm",
    ] = "layer_norm"

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
        x_cent = _center_sae_in(self, x)
        # b_enc is initialized but not used, as in the reference Top-AFA forward.
        hidden_pre = self.hook_sae_acts_pre(x_cent @ self.W_enc)
        return self.hook_sae_acts_post(
            topafa_activations(hidden_pre, self.W_dec, x_cent)
        )

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
    features are kept until their cumulative decoder-scaled norm best matches the
    norm of the centered input, an approximation of quasi-orthogonality.

    Args:
        afa_loss_coefficient (float): Coefficient for the norm-matching loss
            (||f||_2 - ||sae_in||_2)^2, which pulls the feature norm toward the
            input norm. Defaults to 0.0 like the reference config; the reference's
            Top-AFA runs sweep {1/128, 1/64, 1/32, 1/24, 1/16}.
        top_k_aux (int): Number of dead features used by the dead-neuron auxiliary
            loss. Defaults to 512, as in the reference.
        aux_loss_coefficient (float): Coefficient of the dead-neuron auxiliary loss.
            Defaults to 1/32, as in the reference. Inherited from
            TopKTrainingSAEConfig.
        rescale_acts_by_decoder_norm (bool): Not supported; Top-AFA keeps the decoder
            unit-norm directly instead (see
            make_decoder_weights_and_grad_unit_norm). Inherited from
            TopKTrainingSAEConfig.
        k (int): Unused; Top-AFA selects the number of active features per input.
            Inherited from TopKTrainingSAEConfig.
        use_sparse_activations (bool): Unused; Top-AFA always produces dense
            activations. Inherited from TopKTrainingSAEConfig.
        decoder_init_norm (float | None): Norm to initialize decoder weights to.
            Defaults to 1.0 (a unit-norm decoder), as in the reference. Inherited
            from TrainingSAEConfig.
        normalize_activations (Literal["none", "expected_average_only_in", "constant_norm_rescale", "layer_norm"]):
            Normalization strategy for input activations. Defaults to "layer_norm",
            the per-input standardization the reference applies under
            input_unit_norm=True. Inherited from SAEConfig.
        d_in (int): Input dimension (dimensionality of the activations being encoded).
            Inherited from SAEConfig.
        d_sae (int): SAE latent dimension (number of features in the SAE).
            Inherited from SAEConfig.
        dtype (str): Data type for the SAE parameters. Inherited from SAEConfig.
        device (str): Device to place the SAE on. Inherited from SAEConfig.
        apply_b_dec_to_input (bool): Unused; Top-AFA always centers the input with
            b_dec before encoding, as in the reference. Inherited from SAEConfig.
        reshape_activations (Literal["none", "hook_z"]): How to reshape activations
            (useful for attention head outputs). Inherited from SAEConfig.
        metadata (SAEMetadata): Metadata about the SAE training (model name, hook name, etc.).
            Inherited from SAEConfig.
    """

    afa_loss_coefficient: float = 0.0
    top_k_aux: int = 512
    aux_loss_coefficient: float = 1 / 32
    rescale_acts_by_decoder_norm: bool = False
    decoder_init_norm: float | None = 1.0
    normalize_activations: Literal[
        "none",
        "expected_average_only_in",
        "constant_norm_rescale",
        "layer_norm",
    ] = "layer_norm"

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topafa"

    def __post_init__(self):
        super().__post_init__()
        if self.rescale_acts_by_decoder_norm:
            raise ValueError(
                "rescale_acts_by_decoder_norm is not supported for Top-AFA SAEs: the "
                "decoder is kept unit-norm directly instead "
                "(make_decoder_weights_and_grad_unit_norm)."
            )


class TopAFATrainingSAE(TopKTrainingSAE):
    """
    Training SAE with per-input adaptive sparsity (Top-AFA).

    Instead of a fixed k, each input keeps the features whose cumulative
    decoder-scaled norm best matches the norm of the centered input, so the l0
    adapts to the input. The decoder is kept unit-norm by re-normalizing its rows
    and projecting its gradients every optimizer step, as in the reference. Trained
    Top-AFA SAEs are saved as TopAFASAE, since the same selection rule is used at
    inference.
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
        x_cent = _center_sae_in(self, x)
        # b_enc is initialized but not used, as in the reference Top-AFA forward.
        hidden_pre = self.hook_sae_acts_pre(x_cent @ self.W_enc)

        feature_acts = self.hook_sae_acts_post(
            topafa_activations(hidden_pre, self.W_dec, x_cent)
        )
        return feature_acts, hidden_pre

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Norm-matching loss plus the reference's dead-neuron auxiliary loss: the top
        dead features reconstruct the residual of the live reconstruction. Both are
        computed on the preprocessed (but not b_dec-centered) input, as in the
        reference.
        """
        # The reference computes its losses on the preprocessed input; re-apply the
        # input normalization (decode has already consumed and cleared its state).
        x = self.run_time_activation_norm_fn_in(step_input.sae_in)
        afa_loss = (feature_acts.norm(dim=-1) - x.norm(dim=-1)).pow(2).mean()
        losses = {"afa_loss": self.cfg.afa_loss_coefficient * afa_loss}

        dead_neuron_mask = step_input.dead_neuron_mask
        if dead_neuron_mask is None or (num_dead := int(dead_neuron_mask.sum())) == 0:
            losses["auxiliary_reconstruction_loss"] = feature_acts.new_tensor(0.0)
            return losses

        # The residual is not detached, as in the reference.
        residual = x - (feature_acts @ self.W_dec + self.b_dec)
        dead_acts = hidden_pre.relu()[..., dead_neuron_mask]
        aux_topk = torch.topk(dead_acts, min(self.cfg.top_k_aux, num_dead), dim=-1)
        aux_acts = torch.zeros_like(dead_acts).scatter(
            -1, aux_topk.indices, aux_topk.values
        )
        recons = aux_acts @ self.W_dec[dead_neuron_mask]
        aux_squared_error = (recons.float() - residual.float()).pow(2).mean()
        losses["auxiliary_reconstruction_loss"] = (
            self.cfg.aux_loss_coefficient * aux_squared_error
        )
        return losses

    @override
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        """
        Re-normalize the decoder rows to unit norm and project their gradients onto
        the unit sphere, as the reference does every optimizer step. Called by the
        trainer after loss.backward().
        """
        assert self.W_dec.grad is not None
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        raise NotImplementedError(
            "Folding W_dec_norm is not supported for TopAFATrainingSAE, as this "
            "would change the norm-matching feature selection"
        )

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        output = super().training_forward_pass(step_input)
        # The reference logs the mean l0 every training step; compute it on every
        # step (the trainer emits it at its logging cadence).
        l0 = (output.feature_acts > 0).sum(-1).float()
        output.metrics["l0_norm"] = l0.mean()
        return output
