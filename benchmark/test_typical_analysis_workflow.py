"""Typical analysis workflows, exercised end to end on the public API.

Each test walks through a workflow a researcher would actually run, using
sae_lens components the way external code is meant to use them.
"""

from collections.abc import Iterator

import pytest
import torch

from sae_lens.config import SAETrainerConfig
from sae_lens.saes.topk_sae import TopKTrainingSAE
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.scale_diagnostics import (
    ScaleDriftTracker,
    decoder_scale_stats,
    sharpness_log_dict,
)
from tests.helpers import build_topk_sae_training_cfg


def test_typical_scale_drift_diagnostic_workflow():
    """
    Watch finite-stepsize training break the encoder/decoder scale symmetry.

    Adapted from "Broken scale symmetries in undercomplete linear
    autoencoders" (https://arxiv.org/abs/2610.03640): on the solution
    manifold, finite-stepsize training drifts toward large decoder weights
    (directed, not random), and the drifted endpoint is sharper. For plain
    SGD the drift terminates against the stability boundary where the
    learning rate reaches 2 / max eigenvalue of the loss Hessian; that
    boundary is what `sharpness/stability_margin` measures, so it doubles
    as the yardstick under the stock Adam trainer used here (which
    tolerates more curvature and can end at or just past it).

    A TopK SAE with k = d_sae keeps every pre-activation, so with
    d_sae < d_in it is (up to TopK's relu) the undercomplete autoencoder the
    paper studies. Training starts balanced on the PCA solution manifold:
    the decoder holds the top decoder-dimension eigenvectors of the data
    covariance, the encoder mirrors it, and the scale imbalance is exactly
    zero. The scale diagnostics merge into the trainer's log dict at the
    logging step, next to `build_train_step_log_dict`.
    """
    d_in, d_sae = 8, 4
    learning_rate = 3e-2
    n_steps = 1200

    cfg = build_topk_sae_training_cfg(
        d_in=d_in,
        d_sae=d_sae,
        k=d_sae,
        rescale_acts_by_decoder_norm=False,
        decoder_init_norm=None,
    )
    sae = TopKTrainingSAE(cfg)

    # Gaussian activations with a non-isotropic covariance spectrum.
    gram = torch.randn(d_in, d_in)
    spectrum, directions = torch.linalg.eigh(gram @ gram.T / d_in)
    data = torch.randn(2048, d_in) @ (directions * spectrum.sqrt().unsqueeze(0))
    probe_batch = data[:256]

    def batches() -> Iterator[torch.Tensor]:
        while True:
            yield data[torch.randint(0, data.shape[0], (32,))]

    top_directions = directions[:, -d_sae:]
    with torch.no_grad():
        sae.W_dec.copy_(top_directions.T)
        sae.W_enc.copy_(top_directions)
        sae.b_enc.zero_()
        sae.b_dec.zero_()

    trainer = SAETrainer(
        cfg=SAETrainerConfig(
            total_training_samples=n_steps * 32,
            train_batch_size_samples=32,
            lr=learning_rate,
            lr_end=learning_rate,
            dead_feature_window=10_000,
        ),
        sae=sae,
        data_provider=batches(),
    )

    tracker = ScaleDriftTracker()
    initial_stats = decoder_scale_stats(sae)
    initial_sharpness = sharpness_log_dict(sae, probe_batch, learning_rate)
    assert initial_stats["scale/mean_log_imbalance"] == pytest.approx(0.0, abs=1e-6)

    batch_source = trainer.data_provider
    first_log_dict: dict[str, float] = {}
    for step in range(n_steps):
        output = trainer.step(next(batch_source))
        trainer.n_training_steps += 1
        if step % 100 == 0:
            log_dict = trainer.build_train_step_log_dict(
                output, trainer.n_training_samples
            )
            log_dict.update(tracker.update(sae, trainer.n_training_steps))
            if step == 0:
                first_log_dict = log_dict

    final_stats = decoder_scale_stats(sae)
    final_sharpness = sharpness_log_dict(sae, probe_batch, learning_rate)

    print("directed drift slope (log imbalance per step):", tracker.drift_slope())
    print(
        "mean decoder norm:",
        initial_stats["scale/mean_decoder_norm"],
        "->",
        final_stats["scale/mean_decoder_norm"],
    )
    print(
        "loss Hessian max eigenvalue:",
        initial_sharpness["sharpness/loss_hessian_max_eig"],
        "->",
        final_sharpness["sharpness/loss_hessian_max_eig"],
    )
    print(
        "finite-stepsize stability margin:",
        initial_sharpness["sharpness/stability_margin"],
        "->",
        final_sharpness["sharpness/stability_margin"],
    )

    # The diagnostics wire into the trainer's logging step.
    assert "losses/mse_loss" in first_log_dict
    assert "scale/mean_log_imbalance" in first_log_dict

    # Directed scale drift toward large decoder weights.
    assert tracker.drift_slope() > 2e-4
    assert (
        final_stats["scale/mean_decoder_norm"]
        > initial_stats["scale/mean_decoder_norm"] + 0.5
    )
    # The drifted endpoint is sharper, and the drift carries the run from
    # an order of magnitude of learning-rate headroom down to the
    # finite-stepsize stability boundary.
    assert (
        final_sharpness["sharpness/loss_hessian_max_eig"]
        > 2.5 * initial_sharpness["sharpness/loss_hessian_max_eig"]
    )
    assert final_sharpness["sharpness/stability_margin"] < 2.5
    assert (
        final_sharpness["sharpness/stability_margin"]
        < initial_sharpness["sharpness/stability_margin"] / 3
    )
