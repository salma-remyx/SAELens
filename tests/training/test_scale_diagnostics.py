import math

import pytest
import torch

from sae_lens.saes.sae import TrainStepInput
from sae_lens.saes.topk_sae import TopKTrainingSAE
from sae_lens.training.scale_diagnostics import (
    ScaleDriftTracker,
    critical_learning_rate,
    decoder_scale_stats,
    loss_hessian_max_eig,
    stability_margin,
)
from tests.helpers import build_topk_sae_training_cfg, random_params


def build_linear_sae(d_in: int, d_sae: int) -> TopKTrainingSAE:
    # k = d_sae keeps every pre-activation, so the TopK SAE is (up to
    # TopK's relu) the linear autoencoder whose scale symmetry the
    # diagnostics track.
    cfg = build_topk_sae_training_cfg(
        d_in=d_in,
        d_sae=d_sae,
        k=d_sae,
        rescale_acts_by_decoder_norm=False,
        decoder_init_norm=None,
    )
    return TopKTrainingSAE(cfg)


def _training_loss(sae: TopKTrainingSAE, batch: torch.Tensor) -> torch.Tensor:
    return sae.training_forward_pass(
        step_input=TrainStepInput(
            sae_in=batch,
            coefficients={},
            dead_neuron_mask=None,
            n_training_steps=0,
            is_logging_step=False,
        )
    ).loss


def _flat_gradient(params: list[torch.Tensor], loss: torch.Tensor) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params)
    assert not isinstance(grads, dict)
    return torch.cat([g.reshape(-1) for g in grads])


def _finite_difference_hessian(
    sae: TopKTrainingSAE, batch: torch.Tensor, h: float = 1e-2
) -> torch.Tensor:
    params: list[torch.Tensor] = [p for p in sae.parameters() if p.requires_grad]
    views = [p.reshape(-1) for p in params]
    n = sum(v.numel() for v in views)
    hessian = torch.zeros(n, n)
    column = 0
    for view in views:
        for index in range(view.numel()):
            central_difference = torch.zeros(n)
            for sign in (1.0, -1.0):
                with torch.no_grad():
                    view[index] += sign * h
                central_difference += sign * _flat_gradient(
                    params, _training_loss(sae, batch)
                )
                with torch.no_grad():
                    view[index] -= sign * h
            hessian[:, column] = central_difference / (2 * h)
            column += 1
    return hessian


def test_decoder_scale_stats_matches_hand_computed_imbalance():
    sae = build_linear_sae(d_in=4, d_sae=3)
    decoder_scales = torch.tensor([2.0, 0.5, 4.0])
    encoder_scales = torch.tensor([1.0, 2.0, 0.25])
    with torch.no_grad():
        sae.W_dec.zero_()
        sae.W_enc.zero_()
        for i in range(3):
            sae.W_dec[i, i] = decoder_scales[i]
            sae.W_enc[i, i] = encoder_scales[i]

    stats = decoder_scale_stats(sae)

    assert stats["scale/mean_decoder_norm"] == pytest.approx(6.5 / 3)
    assert stats["scale/mean_encoder_norm"] == pytest.approx(3.25 / 3)
    # per-feature log imbalances are ln(2), ln(1/4), ln(16).
    assert stats["scale/mean_log_imbalance"] == pytest.approx(math.log(2))
    assert stats["scale/std_log_imbalance"] == pytest.approx(3 * math.log(2))


def test_scale_drift_tracker_slope_recovers_exact_linear_drift():
    sae = build_linear_sae(d_in=4, d_sae=3)
    with torch.no_grad():
        sae.W_dec.copy_(torch.eye(3, 4))
        sae.W_enc.copy_(torch.eye(4, 3))

    drift_rate = 0.01
    tracker = ScaleDriftTracker()
    for step in range(10):
        tracker.update(sae, step)
        with torch.no_grad():
            sae.W_dec *= math.exp(drift_rate)

    assert tracker.drift_slope() == pytest.approx(drift_rate, abs=1e-6)
    assert len(tracker.steps) == 10
    assert tracker.mean_log_imbalance[0] == pytest.approx(0.0, abs=1e-6)

    # The mirrored drift (encoder growing instead of the decoder) slopes the
    # other way, so the tracker reports the drift sign, not just its rate.
    mirrored = ScaleDriftTracker()
    for step in range(10):
        mirrored.update(sae, step)
        with torch.no_grad():
            sae.W_enc *= math.exp(drift_rate)
    assert mirrored.drift_slope() == pytest.approx(-drift_rate, abs=1e-6)

    # With no drift at all, and before there is enough history to fit one.
    stationary = ScaleDriftTracker()
    stationary.update(sae, 0)
    assert stationary.drift_slope() == 0.0
    stationary.update(sae, 1)
    assert stationary.drift_slope() == 0.0


def test_loss_hessian_max_eig_matches_finite_difference_hessian():
    sae = build_linear_sae(d_in=3, d_sae=2)
    random_params(sae)
    # Keep every pre-activation comfortably positive so the relu inside TopK
    # stays on its smooth side: finite differences then measure the same
    # curvature the Hessian-vector products see, rather than a kink.
    with torch.no_grad():
        sae.b_enc += 100.0
    batch = torch.randn(6, 3)

    hessian = _finite_difference_hessian(sae, batch)
    eigenvalues = torch.linalg.eigvalsh(hessian)
    expected = eigenvalues[int(eigenvalues.abs().argmax())].item()

    snapshot = {k: v.clone() for k, v in sae.state_dict().items()}
    max_eig = loss_hessian_max_eig(sae, batch, n_iters=500, tol=1e-8)

    assert max_eig == pytest.approx(expected, rel=2e-3)
    # The probe must not disturb the parameters it inspects.
    for name, value in sae.state_dict().items():
        assert torch.equal(value, snapshot[name])


def test_critical_learning_rate_and_stability_margin():
    assert critical_learning_rate(4.0) == pytest.approx(0.5)
    # Sitting exactly on the finite-stepsize stability boundary.
    assert stability_margin(4.0, critical_learning_rate(4.0)) == pytest.approx(1.0)
    assert stability_margin(4.0, 0.25) == pytest.approx(2.0)
    # Flat or negative curvature imposes no stability limit.
    assert critical_learning_rate(0.0) == float("inf")
    assert critical_learning_rate(-1.0) == float("inf")
