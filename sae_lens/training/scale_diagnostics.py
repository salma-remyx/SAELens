"""Diagnostics for scale-symmetry breaking during SAE training.

Adapted from "Broken scale symmetries in undercomplete linear autoencoders"
(https://arxiv.org/abs/2610.03640). An autoencoder has a per-feature scale
symmetry: scaling a decoder weight vector up while scaling the matching
encoder weight vector down leaves the reconstruction unchanged. Gradient
flow preserves this symmetry, but finite-stepsize SGD breaks it with a
preferred sign: in undercomplete autoencoders the drift is directed toward
large decoder weights, it terminates near the finite-stepsize stability
boundary (learning rate approaching 2 / max eigenvalue of the loss
Hessian), and the drifted endpoints are sharper than a balanced baseline.

This module measures that drift on any TrainingSAE, giving visibility into
the decoder weight drift that `rescale_acts_by_decoder_norm` exists to
mitigate for TopK / BatchTopK / Matryoshka SAEs:

- `decoder_scale_stats` / `ScaleDriftTracker`: snapshot and track the
  decoder-vs-encoder scale imbalance over training, with a least-squares
  estimate of the directed drift rate.
- `loss_hessian_max_eig`: sharpness probe; the signed largest-magnitude
  eigenvalue of the training-loss Hessian, estimated by power iteration on
  Hessian-vector products.
- `critical_learning_rate` / `stability_margin`: distance to the
  finite-stepsize stability boundary.

The log dicts returned by `ScaleDriftTracker.update` and
`sharpness_log_dict` are meant to be merged into the trainer's log dict at
the logging step, next to `SAETrainer.build_train_step_log_dict`:

    tracker = ScaleDriftTracker()
    ...
    output = trainer.step(batch)
    trainer.n_training_steps += 1
    log_dict = trainer.build_train_step_log_dict(output, trainer.n_training_samples)
    log_dict.update(tracker.update(trainer.sae, trainer.n_training_steps))
    log_dict.update(
        sharpness_log_dict(
            trainer.sae,
            output.sae_in,
            trainer.optimizer.param_groups[0]["lr"],
        )
    )
"""

from __future__ import annotations

from typing import Any

import torch

from sae_lens.saes.sae import (
    TrainCoefficientConfig,
    TrainingSAE,
    TrainStepInput,
)


@torch.no_grad()
def decoder_scale_stats(sae: TrainingSAE[Any]) -> dict[str, float]:
    """
    Snapshot of the per-feature decoder/encoder scale imbalance of an SAE.

    `log(||W_dec[:, i]|| / ||W_enc[:, i]||)` is the coordinate of the scale
    symmetry that finite-stepsize SGD drifts along: it is exactly zero for a
    balanced parameterization, and a consistently signed drift during
    training is the broken scale symmetry of the loss geometry.
    """
    decoder_norm = sae.W_dec.norm(dim=-1)
    encoder_norm = sae.W_enc.norm(dim=0)
    log_imbalance = decoder_norm.log() - encoder_norm.log()
    return {
        "scale/mean_decoder_norm": decoder_norm.mean().item(),
        "scale/mean_encoder_norm": encoder_norm.mean().item(),
        "scale/mean_log_imbalance": log_imbalance.mean().item(),
        "scale/std_log_imbalance": log_imbalance.std().item(),
    }


class ScaleDriftTracker:
    """Accumulates scale snapshots over training and estimates drift rate.

    Call `update` at the trainer's logging cadence; `drift_slope` then
    returns the least-squares slope of the mean log imbalance against step,
    which is the directed drift rate of the broken scale symmetry (per
    training step). Its sign is the preferred drift direction and its
    magnitude the drift timescale.
    """

    steps: list[int]
    mean_log_imbalance: list[float]
    mean_decoder_norm: list[float]

    def __init__(self) -> None:
        self.steps = []
        self.mean_log_imbalance = []
        self.mean_decoder_norm = []

    @torch.no_grad()
    def update(self, sae: TrainingSAE[Any], step: int) -> dict[str, float]:
        stats = decoder_scale_stats(sae)
        self.steps.append(step)
        self.mean_log_imbalance.append(stats["scale/mean_log_imbalance"])
        self.mean_decoder_norm.append(stats["scale/mean_decoder_norm"])
        return stats

    def drift_slope(self) -> float:
        if len(self.steps) < 2:
            return 0.0
        n = len(self.steps)
        mean_step = sum(self.steps) / n
        mean_imbalance = sum(self.mean_log_imbalance) / n
        step_variance = sum((step - mean_step) ** 2 for step in self.steps)
        if step_variance == 0.0:
            return 0.0
        covariance = sum(
            (step - mean_step) * (imbalance - mean_imbalance)
            for step, imbalance in zip(self.steps, self.mean_log_imbalance)
        )
        return covariance / step_variance


def loss_hessian_max_eig(
    sae: TrainingSAE[Any],
    sae_in: torch.Tensor,
    coefficients: dict[str, float] | None = None,
    n_iters: int = 50,
    tol: float = 1e-4,
) -> float:
    """
    Sharpness probe: signed largest-magnitude eigenvalue of the loss Hessian.

    Runs power iteration on Hessian-vector products of the SAE's training
    loss (the same loss `training_forward_pass` returns) over `sae_in`. At
    a minimum of the loss this is the maximum eigenvalue of the Hessian,
    the sharpness measure that grows as scale drift pushes toward the
    finite-stepsize stability boundary; the Rayleigh quotient keeps the
    sign so a dominant negative curvature direction is reported as such.

    The probe uses `torch.autograd.grad` internally and never accumulates
    into `param.grad`, so it is safe to call mid-training.
    """
    params: list[torch.Tensor] = [p for p in sae.parameters() if p.requires_grad]
    if coefficients is None:
        coefficients = _float_coefficients(sae)
    vec = [torch.randn_like(p) for p in params]
    vec = _l2_normalize(vec)
    eigenvalue = 0.0
    for _ in range(n_iters):
        loss = _training_loss(sae, sae_in, coefficients)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        assert not isinstance(grads, dict)  # inputs is a list, not a dict
        grad_dot_vec = sum((g * v).sum() for g, v in zip(grads, vec))
        hessian_vec = torch.autograd.grad(grad_dot_vec, params)
        assert not isinstance(hessian_vec, dict)  # inputs is a list, not a dict
        rayleigh = sum((h * v).sum() for h, v in zip(hessian_vec, vec)).item()
        magnitude = _l2_norm(hessian_vec)
        if magnitude == 0.0:
            return 0.0
        vec = _l2_normalize(list(hessian_vec))
        if abs(magnitude - eigenvalue) <= tol * abs(magnitude):
            return rayleigh
        eigenvalue = magnitude
    return rayleigh


def sharpness_log_dict(
    sae: TrainingSAE[Any],
    sae_in: torch.Tensor,
    learning_rate: float,
    coefficients: dict[str, float] | None = None,
    n_iters: int = 50,
    tol: float = 1e-4,
) -> dict[str, float]:
    """
    Sharpness and stability-margin log dict, for the trainer's logging step.

    `sharpness/stability_margin` is the ratio of the largest stable learning
    rate to the current one: 1.0 sits exactly on the finite-stepsize
    stability boundary that directed scale drift terminates against, below
    1.0 the run is past it.
    """
    max_eig = loss_hessian_max_eig(
        sae, sae_in, coefficients=coefficients, n_iters=n_iters, tol=tol
    )
    return {
        "sharpness/loss_hessian_max_eig": max_eig,
        "sharpness/stability_margin": stability_margin(max_eig, learning_rate),
    }


def critical_learning_rate(max_eig: float) -> float:
    """
    Largest learning rate for which plain gradient descent is stable against
    curvature `max_eig` (the 2 / lambda_max boundary of the paper).
    """
    if max_eig <= 0.0:
        return float("inf")
    return 2.0 / max_eig


def stability_margin(max_eig: float, learning_rate: float) -> float:
    """Ratio of the critical learning rate to the current learning rate."""
    return critical_learning_rate(max_eig) / learning_rate


def _training_loss(
    sae: TrainingSAE[Any],
    sae_in: torch.Tensor,
    coefficients: dict[str, float],
) -> torch.Tensor:
    output = sae.training_forward_pass(
        step_input=TrainStepInput(
            sae_in=sae_in,
            coefficients=coefficients,
            dead_neuron_mask=None,
            n_training_steps=0,
            is_logging_step=False,
        ),
    )
    return output.loss


def _float_coefficients(sae: TrainingSAE[Any]) -> dict[str, float]:
    coefficients: dict[str, float] = {}
    for name, value in sae.get_coefficients().items():
        if isinstance(value, TrainCoefficientConfig):
            coefficients[name] = value.value
        else:
            coefficients[name] = value
    return coefficients


def _l2_norm(tensors: list[torch.Tensor]) -> float:
    return sum(t.pow(2).sum() for t in tensors).sqrt().item()


def _l2_normalize(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    norm = _l2_norm(tensors)
    return [t / norm for t in tensors]


__all__ = [
    "ScaleDriftTracker",
    "critical_learning_rate",
    "decoder_scale_stats",
    "loss_hessian_max_eig",
    "sharpness_log_dict",
    "stability_margin",
]
