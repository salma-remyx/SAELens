"""
Ablations comparing learned and frozen-random linear expansions in SAE encoders.

An expand-and-sparsify representation lifts an input into a much higher dimension
with a linear map and keeps only the top-k coordinates. The TopK SAE encoder is
exactly this layer with the map learned by gradient descent; freezing the encoder
at its random initialization recovers the random-map baseline that the classical
theory analyses. Sweeping the expansion width (d_sae / d_in) against both
variants measures what learning the expansion actually buys.

Adapted from "On Learning Expand-and-Sparsify Representations"
(https://arxiv.org/abs/2610.26470), which reports that learning the expansion
helps most at the narrowest expansions and that the advantage decays towards zero
as the expansion grows. The OpenML task suite and Wilcoxon protocol of that paper
are replaced by this repo's synthetic feature dictionaries, `train_toy_sae`, and
`eval_sae_on_synthetic_data`, which give ground-truth feature recovery alongside
reconstruction quality.

Use this module via `from sae_lens.synthetic.learned_expansion import ...`.
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any, Literal

import torch
from torch import nn

from sae_lens.saes.sae import TrainingSAE
from sae_lens.saes.standard_sae import StandardTrainingSAE, StandardTrainingSAEConfig
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig
from sae_lens.synthetic.activation_generator import ActivationGenerator
from sae_lens.synthetic.evals import eval_sae_on_synthetic_data
from sae_lens.synthetic.feature_dictionary import FeatureDictionary
from sae_lens.synthetic.training import train_toy_sae

EncoderMode = Literal["learned", "random"]


def freeze_encoder(
    sae: TrainingSAE[Any],
    freeze_bias: bool = True,
) -> TrainingSAE[Any]:
    """
    Freeze the encoder of a TrainingSAE so it stays at its random initialization.

    With the encoder frozen, training optimizes only the decoder (and any
    architecture-specific parameters), which turns the SAE into the
    random-expansion baseline: a fixed random linear map followed by top-k (or
    ReLU) selection, with a learned readout on top.

    Args:
        sae: The TrainingSAE whose encoder should be frozen.
        freeze_bias: Also freeze the encoder bias, so the expansion is purely
            the random linear map. Leave False to keep learning b_enc.

    Returns:
        The same SAE, with W_enc (and optionally b_enc) marked requires_grad=False.
    """
    sae.W_enc.requires_grad_(False)
    b_enc = getattr(sae, "b_enc", None)
    if freeze_bias and isinstance(b_enc, nn.Parameter):
        b_enc.requires_grad_(False)
    return sae


def _relative_drift(before: torch.Tensor, after: torch.Tensor) -> float:
    """Relative change in a parameter over training, as a fraction of its norm."""
    norm = before.norm()
    if norm == 0:
        return 0.0
    return ((after - before).norm() / norm).item()


def _build_training_sae(
    architecture: str,
    d_in: int,
    d_sae: int,
    k: int,
    decoder_init_norm: float | None,
    standard_l1_coefficient: float,
    device: str,
) -> TrainingSAE[Any]:
    if architecture == "topk":
        return TopKTrainingSAE(
            TopKTrainingSAEConfig(
                d_in=d_in,
                d_sae=d_sae,
                k=k,
                decoder_init_norm=decoder_init_norm,
                device=device,
            )
        )
    if architecture == "standard":
        return StandardTrainingSAE(
            StandardTrainingSAEConfig(
                d_in=d_in,
                d_sae=d_sae,
                l1_coefficient=standard_l1_coefficient,
                decoder_init_norm=decoder_init_norm,
                device=device,
            )
        )
    raise ValueError(
        f"Unknown architecture {architecture!r}; expected 'topk' or 'standard'."
    )


@dataclass
class ExpansionArmResult:
    """Metrics for one (expansion width, architecture, encoder variant) run."""

    architecture: str
    """SAE architecture used, 'topk' or 'standard' (ReLU)."""

    encoder: str
    """Whether the encoder was trained ('learned') or frozen ('random')."""

    d_sae: int
    """Expanded dimension of the encoder."""

    expansion_ratio: float
    """Expansion width as d_sae / d_in."""

    k: int | None
    """Number of coordinates kept by the topk activation (None for ReLU)."""

    encoder_drift: float
    """Relative change in W_enc over training. Exactly 0.0 for frozen encoders."""

    explained_variance: float
    """Fraction of input variance explained by the reconstruction (R^2)."""

    mcc: float
    """Mean correlation coefficient between SAE decoder and ground truth features."""

    sae_l0: float
    """Mean number of active SAE latents per sample."""

    dead_latents: int
    """Number of SAE latents that never fired during evaluation."""

    def to_log_dict(self, prefix: str = "") -> dict[str, Any]:
        """Convert to a flat dictionary of scalar values for logging (e.g. wandb)."""
        return {
            f"{prefix}{field.name}": getattr(self, field.name)
            for field in fields(self)
            if field.name != "architecture"
        }


@dataclass
class ExpansionStudyResult:
    """Collection of expansion-study arms over a shared synthetic dataset."""

    d_in: int
    """Input dimension (the hidden dimension of the feature dictionary)."""

    arms: list[ExpansionArmResult]
    """One entry per (expansion ratio, architecture, encoder) combination run."""

    def learned_vs_random_gain(self, architecture: str = "topk") -> dict[float, float]:
        """
        Map each expansion ratio to the explained-variance gain from learning.

        The gain is explained_variance(learned) - explained_variance(random) for
        the requested architecture. Positive values mean gradient descent on the
        expansion map beat the random map, which is the paper's headline effect.

        Args:
            architecture: Which architecture's arms to compare, 'topk' or 'standard'.

        Returns:
            Dictionary mapping expansion ratio to learned-minus-random gain.
        """
        gains: dict[float, float] = {}
        for arm in self.arms:
            if arm.architecture != architecture or arm.encoder != "learned":
                continue
            random_arms = [
                other
                for other in self.arms
                if other.architecture == architecture
                and other.encoder == "random"
                and other.expansion_ratio == arm.expansion_ratio
            ]
            if len(random_arms) != 1:
                raise ValueError(
                    f"Need exactly one 'random' arm at expansion ratio "
                    f"{arm.expansion_ratio} for architecture {architecture!r}."
                )
            gains[arm.expansion_ratio] = (
                arm.explained_variance - random_arms[0].explained_variance
            )
        return gains


def run_expansion_study(
    feature_dict: FeatureDictionary,
    activations_generator: ActivationGenerator,
    expansion_ratios: Sequence[float] = (2.0, 4.0, 8.0),
    architectures: Sequence[str] = ("topk", "standard"),
    encoders: Sequence[EncoderMode] = ("learned", "random"),
    k_fraction: float = 0.1,
    standard_l1_coefficient: float = 0.0,
    training_samples: int = 100_000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    eval_samples: int = 20_000,
    decoder_init_norm: float | None = 0.1,
    device: str | torch.device = "cpu",
) -> ExpansionStudyResult:
    """
    Train and evaluate SAEs across expansion widths and encoder variants.

    Every arm sees the same data distribution (the given feature dictionary and
    activation generator) and the same training budget, so arms differ only in
    expansion width, activation function, and whether the encoder is learned or
    frozen at its random initialization.

    Args:
        feature_dict: Feature dictionary mapping sparse feature activations to
            the hidden activations the SAEs are trained on.
        activations_generator: Generator for the ground-truth feature activations.
        expansion_ratios: Expansion widths to sweep, as d_sae / d_in.
        architectures: Architectures to compare, 'topk' and/or 'standard' (ReLU).
        encoders: Encoder variants to run, from ('learned', 'random').
        k_fraction: Fraction of the expanded dimension kept by the topk
            activation; the paper keeps at most a tenth.
        standard_l1_coefficient: L1 coefficient for the ReLU (standard) arms. 0.0
            gives the paper's dense ReLU baseline of the same expanded dimension;
            raise it to match the topk arms' L0 instead.
        training_samples: Number of training samples per arm.
        batch_size: Training batch size.
        lr: Learning rate.
        eval_samples: Number of samples used for evaluation.
        decoder_init_norm: Norm the decoder rows are initialized to.
        device: Device to train and evaluate on.

    Returns:
        ExpansionStudyResult with one arm per combination, plus gain helpers.
    """
    d_in = feature_dict.hidden_dim
    arms: list[ExpansionArmResult] = []

    for ratio in expansion_ratios:
        d_sae = int(round(ratio * d_in))
        k = max(1, int(round(k_fraction * d_sae)))
        for architecture in architectures:
            for encoder in encoders:
                sae = _build_training_sae(
                    architecture=architecture,
                    d_in=d_in,
                    d_sae=d_sae,
                    k=k,
                    decoder_init_norm=decoder_init_norm,
                    standard_l1_coefficient=standard_l1_coefficient,
                    device=str(device),
                )
                if encoder == "random":
                    freeze_encoder(sae)
                w_enc_before = sae.W_enc.detach().clone()

                train_toy_sae(
                    sae=sae,
                    feature_dict=feature_dict,
                    activations_generator=activations_generator,
                    training_samples=training_samples,
                    batch_size=batch_size,
                    lr=lr,
                    device=device,
                )

                eval_result = eval_sae_on_synthetic_data(
                    sae,
                    feature_dict,
                    activations_generator,
                    num_samples=eval_samples,
                )
                arms.append(
                    ExpansionArmResult(
                        architecture=architecture,
                        encoder=encoder,
                        d_sae=d_sae,
                        expansion_ratio=ratio,
                        k=k if architecture == "topk" else None,
                        encoder_drift=_relative_drift(w_enc_before, sae.W_enc.detach()),
                        explained_variance=eval_result.explained_variance,
                        mcc=eval_result.mcc,
                        sae_l0=eval_result.sae_l0,
                        dead_latents=eval_result.dead_latents,
                    )
                )

    return ExpansionStudyResult(d_in=d_in, arms=arms)
