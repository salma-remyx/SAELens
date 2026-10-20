"""Sparse Coding SAE.

Adapted from "Beyond SAEs: Consistent Concept Discovery with Sparse Coding"
(arXiv:2610.23801). Instead of predicting coefficients with a single linear
encoder pass, each input's coefficients are solved per-sample as a nonnegative
Lasso problem against the (tied) decoder dictionary, and a co-active
similarity loss repels near-duplicate dictionary atoms that fire together on
the same input. Per-sample sparse coding plus this repulsion term is what the
paper (ConSDL) uses to improve dictionary recovery and cross-seed consistency.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Any

import torch
from typing_extensions import override

from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    TrainCoefficientConfig,
    TrainingSAE,
    TrainingSAEConfig,
    TrainStepInput,
)

# Number of power iterations used to estimate the squared largest singular
# value of the dictionary, following the paper.
_POWER_ITERATIONS = 32


def _largest_singular_value_squared(W_dec: torch.Tensor) -> torch.Tensor:
    """Estimate sigma_max(W_dec)^2 (the Lipschitz constant of the Lasso
    gradient) with power iteration on W_dec^T W_dec."""
    v = torch.randn(W_dec.shape[1], dtype=W_dec.dtype, device=W_dec.device)
    for _ in range(_POWER_ITERATIONS):
        v = W_dec.T @ (W_dec @ v)
        v = v / v.norm().clamp(min=1e-12)
    return (W_dec @ v).norm().pow(2).clamp(min=1e-12)


@torch.no_grad()
def sparse_code(
    sae_in_centered: torch.Tensor,
    W_dec: torch.Tensor,
    l1_coefficient: float,
    num_steps: int,
    refit_steps: int = 0,
) -> torch.Tensor:
    """
    Solve per-sample nonnegative Lasso coefficients with FISTA.

    Minimizes 0.5 * ||x - W_dec a||^2 + l1_coefficient * ||a||_1 subject to
    a >= 0 with FISTA (accelerated ISTA with nonnegative soft-thresholding),
    then optionally refits the magnitudes on the inferred support with
    projected gradient descent (removing the l1 shrinkage bias without
    activating new atoms). Coefficients are computed without gradients, as
    the dictionary is trained by alternating optimization.

    Args:
        sae_in_centered: Input activations, centered by b_dec. Shape (..., d_in).
        W_dec: Decoder weight matrix. Shape (d_sae, d_in).
        l1_coefficient: Coefficient of the l1 penalty.
        num_steps: Number of FISTA iterations.
        refit_steps: Number of support-constrained refit iterations (0 to skip).
    """
    original_shape = sae_in_centered.shape
    x = (
        sae_in_centered.reshape(-1, original_shape[-1])
        if sae_in_centered.ndim > 2
        else sae_in_centered
    )

    step_size = 1.0 / _largest_singular_value_squared(W_dec)
    threshold = step_size * l1_coefficient

    a = x.new_zeros(x.shape[0], W_dec.shape[0])
    y = a.clone()
    t = 1.0
    for _ in range(num_steps):
        residual = y @ W_dec - x
        a_new = (y - step_size * (residual @ W_dec.T) - threshold).clamp_min(0.0)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = a_new + ((t - 1.0) / t_next) * (a_new - a)
        a = a_new
        t = t_next

    if refit_steps > 0:
        support = a > 0
        for _ in range(refit_steps):
            residual = x - a @ W_dec
            a = ((a + step_size * (residual @ W_dec.T)) * support).clamp_min(0.0)

    if sae_in_centered.ndim > 2:
        a = a.reshape(*original_shape[:-1], a.shape[-1])
    return a


def coactive_similarity_loss(
    feature_acts: torch.Tensor,
    W_dec: torch.Tensor,
    margin: float,
    num_neighbors: int = 32,
) -> torch.Tensor:
    """
    Repulsion loss on co-active near-duplicate dictionary atoms.

    For every (sample, atom) pair where the atom is active and its most
    similar partner among the atoms co-active in the same sample has cosine
    similarity s above `margin`, penalize (s - margin)^2, averaged over such
    pairs. Candidate partners are restricted to the `num_neighbors` most
    similar atoms. Returns 0 when no pair exceeds the margin. Gradients flow
    through the atom directions (pushing duplicates apart), while the
    co-activity pattern is held fixed.

    Args:
        feature_acts: Sparse feature activations. Shape (..., d_sae).
        W_dec: Decoder weight matrix. Shape (d_sae, d_in).
        margin: Cosine similarity margin below which pairs are not penalized.
        num_neighbors: Number of most-similar candidate partner atoms per atom.
    """
    acts = (
        feature_acts.reshape(-1, feature_acts.shape[-1])
        if feature_acts.ndim > 2
        else feature_acts
    )
    active = acts > 0

    W_normed = W_dec / W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = W_normed @ W_normed.T
    diag = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(diag, -1.0)

    num_neighbors = min(num_neighbors, sim.shape[-1] - 1)
    neighbor_sim, neighbor_idx = sim.topk(num_neighbors, dim=-1)

    # Restrict partners to candidates that are active in the same sample.
    neighbor_active = active[:, neighbor_idx]  # (batch, d_sae, num_neighbors)
    partner_sim = neighbor_sim.unsqueeze(0).masked_fill(~neighbor_active, -1.0)
    most_similar = partner_sim.max(dim=-1).values  # (batch, d_sae)

    contributing = active & (most_similar > margin)
    count = contributing.sum()
    return ((most_similar - margin).pow(2) * contributing).sum() / count.clamp(min=1)


# --- inference ---


@dataclass
class SparseCodingSAEConfig(SAEConfig):
    """
    Configuration class for SparseCodingSAE inference.

    Args:
        fista_steps (int): Number of FISTA iterations used to encode each input.
        refit_steps (int): Number of support-constrained refit iterations used
            to remove the l1 shrinkage bias of the inferred coefficients.
        sparsity_coefficient (float): Coefficient of the l1 penalty in the
            nonnegative Lasso problem solved to encode each input.
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

    fista_steps: int = 30
    refit_steps: int = 5
    sparsity_coefficient: float = 0.1

    @override
    @classmethod
    def architecture(cls) -> str:
        return "sparse_coding"


class SparseCodingSAE(SAE[SparseCodingSAEConfig]):
    """
    An inference sparse autoencoder that encodes each input by per-sample
    sparse coding against the decoder dictionary instead of a linear encoder.
    """

    # Sparse coding uses the decoder as its dictionary, so we use W_enc as the
    # decoder transposed
    @property
    def W_enc(self) -> torch.Tensor:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.W_dec.T

    # hacky way to get around the base class having W_enc.
    # TODO: harmonize with the base class in next major release
    @override
    def __setattr__(self, name: str, value: Any):
        if name == "W_enc":
            return
        super().__setattr__(name, value)

    @override
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts input x into feature activations by solving the nonnegative
        Lasso problem against the decoder dictionary.
        """
        sae_in = self.process_sae_in(x)
        return sparse_code(
            sae_in,
            self.W_dec,
            l1_coefficient=self.cfg.sparsity_coefficient,
            num_steps=self.cfg.fista_steps,
            refit_steps=self.cfg.refit_steps,
        )

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """
        Decode the feature activations back to the input space.
        """
        sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        raise NotImplementedError(
            "Folding W_dec_norm is not safe for SparseCodingSAEs, as this may change the resulting activations"
        )

    @override
    @torch.no_grad()
    def fold_activation_norm_scaling_factor(self, scaling_factor: float):
        # the l1 penalty applies in the (rescaled) input space, so it must be
        # folded together with the weights for the encoding to be preserved
        self.cfg.sparsity_coefficient /= scaling_factor
        super().fold_activation_norm_scaling_factor(scaling_factor)


# --- training ---


@dataclass
class SparseCodingTrainingSAEConfig(TrainingSAEConfig):
    """
    Configuration class for training a SparseCodingTrainingSAE.

    Args:
        fista_steps (int): Number of FISTA iterations used to encode each input.
        refit_steps (int): Number of support-constrained refit iterations used
            to remove the l1 shrinkage bias of the inferred coefficients.
        sparsity_coefficient (float): Coefficient of the l1 penalty in the
            nonnegative Lasso problem solved to encode each input. Needs to be
            tuned to reach a target L0.
        similarity_loss_coefficient (float): Coefficient of the co-active
            similarity loss, which repels near-duplicate dictionary atoms that
            fire together on the same input.
        similarity_loss_warm_up_steps (int): Number of steps to linearly ramp up
            the co-active similarity loss coefficient over.
        similarity_margin (float): Cosine similarity margin below which
            co-active atom pairs are not penalized by the similarity loss.
        num_similarity_neighbors (int): Number of most-similar candidate
            partner atoms considered by the co-active similarity loss.
        decoder_init_norm (float | None): Norm to initialize decoder weights to.
            0.1 corresponds to the "heuristic" initialization from Anthropic's
            April update. Use None to disable. Inherited from TrainingSAEConfig.
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
        metadata (SAEMetadata): Metadata about the SAE training (model name, hook
            name, etc.). Inherited from SAEConfig.
    """

    fista_steps: int = 30
    refit_steps: int = 5
    sparsity_coefficient: float = 0.1
    similarity_loss_coefficient: float = 1.0
    similarity_loss_warm_up_steps: int = 0
    similarity_margin: float = 0.7
    num_similarity_neighbors: int = 32

    @override
    @classmethod
    def architecture(cls) -> str:
        return "sparse_coding"

    @override
    def __post_init__(self):
        if self.decoder_init_norm != 1.0:
            self.decoder_init_norm = 1.0
            warnings.warn(
                "decoder_init_norm must be set to 1.0 for SparseCodingTrainingSAE, setting to 1.0"
            )
        super().__post_init__()


class SparseCodingTrainingSAE(TrainingSAE[SparseCodingTrainingSAEConfig]):
    """
    Training variant of the sparse coding SAE. Coefficients are solved per
    sample with FISTA and held fixed, while the dictionary and b_dec are
    trained by gradient descent on the reconstruction loss plus the co-active
    similarity loss (alternating optimization).
    """

    # Sparse coding uses the decoder as its dictionary, so we use W_enc as the
    # decoder transposed
    @property
    def W_enc(self) -> torch.Tensor:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.W_dec.T

    # hacky way to get around the base class having W_enc.
    # TODO: harmonize with the base class in next major release
    @override
    def __setattr__(self, name: str, value: Any):
        if name == "W_enc":
            return
        super().__setattr__(name, value)

    @override
    @torch.no_grad()
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_pre doesn't make sense for sparse coding, since there is not a
        single pre-activation. We just return zeros for the hidden_pre.
        """
        sae_in = self.process_sae_in(x)
        acts = sparse_code(
            sae_in,
            self.W_dec,
            l1_coefficient=self.cfg.sparsity_coefficient,
            num_steps=self.cfg.fista_steps,
            refit_steps=self.cfg.refit_steps,
        )
        return acts, torch.zeros_like(acts)

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Repel near-duplicate dictionary atoms that fire together, which the
        # paper identifies as the key driver of cross-seed consistency.
        similarity_loss = coactive_similarity_loss(
            feature_acts,
            self.W_dec,
            margin=self.cfg.similarity_margin,
            num_neighbors=self.cfg.num_similarity_neighbors,
        )
        return {
            "coactive_similarity_loss": (
                step_input.coefficients["coactive_similarity"] * similarity_loss
            )
        }

    @override
    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]:
        return {
            "coactive_similarity": TrainCoefficientConfig(
                value=self.cfg.similarity_loss_coefficient,
                warm_up_steps=self.cfg.similarity_loss_warm_up_steps,
            ),
        }

    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        raise NotImplementedError(
            "Folding W_dec_norm is not safe for SparseCodingSAEs, as this may change the resulting activations"
        )

    @override
    @torch.no_grad()
    def fold_activation_norm_scaling_factor(self, scaling_factor: float):
        # the l1 penalty applies in the (rescaled) input space, so it must be
        # folded together with the weights for the encoding to be preserved
        self.cfg.sparsity_coefficient /= scaling_factor
        super().fold_activation_norm_scaling_factor(scaling_factor)
