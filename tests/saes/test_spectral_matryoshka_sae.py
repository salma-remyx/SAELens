from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
import torch

from sae_lens.registry import get_sae_training_class
from sae_lens.saes.jumprelu_sae import JumpReLUSAE
from sae_lens.saes.matryoshka_batchtopk_sae import (
    MatryoshkaBatchTopKTrainingSAE,
    MatryoshkaBatchTopKTrainingSAEConfig,
)
from sae_lens.saes.sae import SAE, TrainStepInput, TrainingSAE, TrainingSAEConfig
from sae_lens.saes.spectral_matryoshka_sae import (
    SpectralMatryoshkaBatchTopKTrainingSAE,
    SpectralMatryoshkaBatchTopKTrainingSAEConfig,
    load_spectral_basis,
    register_spectral_matryoshka_sae,
)
from tests.helpers import assert_close, random_params

register_spectral_matryoshka_sae()


def build_spectral_matryoshka_cfg(
    **kwargs: Any,
) -> SpectralMatryoshkaBatchTopKTrainingSAEConfig:
    defaults: dict[str, Any] = {
        "d_in": 8,
        "d_sae": 16,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
        "apply_b_dec_to_input": False,
        "decoder_init_norm": 0.1,
        "k": 4,
        "topk_threshold_lr": 0.02,
        "matryoshka_widths": [4, 8, 16],
        "use_matryoshka_aux_loss": False,
        "rescale_acts_by_decoder_norm": False,
        "spectral_proj_rank_multiplier": 1,
    }
    return SpectralMatryoshkaBatchTopKTrainingSAEConfig(**{**defaults, **kwargs})


def build_train_step_input(
    sae_in: torch.Tensor, dead_neuron_mask: torch.Tensor | None = None
) -> TrainStepInput:
    return TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=dead_neuron_mask,
        n_training_steps=0,
        is_logging_step=False,
    )


def build_matching_plain_matryoshka_sae(
    spectral_sae: SpectralMatryoshkaBatchTopKTrainingSAE,
) -> MatryoshkaBatchTopKTrainingSAE:
    cfg = spectral_sae.cfg
    plain_cfg = MatryoshkaBatchTopKTrainingSAEConfig(
        **{
            f.name: getattr(cfg, f.name)
            for f in fields(MatryoshkaBatchTopKTrainingSAEConfig)
        }
    )
    plain_sae = MatryoshkaBatchTopKTrainingSAE(plain_cfg)
    plain_sae.load_state_dict(
        {
            k: v
            for k, v in spectral_sae.state_dict().items()
            if k in plain_sae.state_dict()
        }
    )
    return plain_sae


def test_initialization_creates_orthonormal_spectral_basis():
    cfg = build_spectral_matryoshka_cfg(spectral_proj_rank_multiplier=4)
    sae = SpectralMatryoshkaBatchTopKTrainingSAE(cfg)

    assert sae.W_proj.shape == (16, 64)
    assert sae.spectral_basis.shape == (16, 16)
    assert sae.spectral_basis.dtype == torch.float32
    # the initial basis is orthonormal, up to float32 eigendecomposition error
    assert_close(sae.spectral_basis.T @ sae.spectral_basis, torch.eye(16), atol=1e-5)


def test_config_rejects_rank_multiplier_below_one():
    with pytest.raises(ValueError, match="spectral_proj_rank_multiplier"):
        SpectralMatryoshkaBatchTopKTrainingSAE(
            build_spectral_matryoshka_cfg(spectral_proj_rank_multiplier=0)
        )


def test_architecture_resolves_through_training_sae_registry():
    sae_cls, cfg_cls = get_sae_training_class("spectral_matryoshka_batchtopk")
    assert sae_cls is SpectralMatryoshkaBatchTopKTrainingSAE
    assert cfg_cls is SpectralMatryoshkaBatchTopKTrainingSAEConfig
    assert (
        TrainingSAE.get_sae_class_for_architecture("spectral_matryoshka_batchtopk")
        is SpectralMatryoshkaBatchTopKTrainingSAE
    )

    cfg = build_spectral_matryoshka_cfg()
    reloaded_cfg = TrainingSAEConfig.from_dict(cfg.to_dict())
    assert isinstance(reloaded_cfg, SpectralMatryoshkaBatchTopKTrainingSAEConfig)
    assert reloaded_cfg.spectral_proj_init_std == pytest.approx(0.02)
    assert reloaded_cfg.spectral_proj_rank_multiplier == 1
    assert cfg.get_inference_sae_cfg_dict()["architecture"] == "jumprelu"

    sae = sae_cls(cfg)
    random_params(sae)
    output = sae.training_forward_pass(build_train_step_input(torch.randn(10, 8)))

    assert "inner_mse_loss_4" in output.losses
    assert "inner_mse_loss_8" in output.losses
    assert "inner_mse_loss_16" not in output.losses


def test_spectral_level_decode_matches_manual_spectral_readout():
    sae = SpectralMatryoshkaBatchTopKTrainingSAE(build_spectral_matryoshka_cfg())
    random_params(sae)
    sae.spectral_basis = torch.linalg.qr(torch.randn(16, 16))[0]

    feature_acts = torch.randn(10, 16)
    levels = dict(sae._iterable_decode(feature_acts, include_outer_loss=True))

    U = sae.spectral_basis
    rotated_acts = feature_acts @ U
    rotated_W_dec = U.T @ sae.W_dec
    for width in [4, 8, 16]:
        expected = rotated_acts[:, :width] @ rotated_W_dec[:width] + sae.b_dec
        assert_close(levels[width], expected)


@pytest.mark.parametrize("rescale_acts_by_decoder_norm", [True, False])
def test_spectral_readout_full_width_matches_standard_decode(
    rescale_acts_by_decoder_norm: bool,
):
    cfg = build_spectral_matryoshka_cfg(
        matryoshka_widths=[4, 16],
        rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
    )
    sae = SpectralMatryoshkaBatchTopKTrainingSAE(cfg)
    random_params(sae)
    sae.spectral_basis = torch.linalg.qr(torch.randn(16, 16))[0]

    feature_acts = torch.randn(10, 16)
    output_spectral = list(sae._iterable_decode(feature_acts, True))[-1][1]
    output_base = sae.decode(feature_acts)
    # the rotation cancels at full width, up to float32 orthonormality of the basis
    assert_close(output_spectral, output_base, atol=1e-5)


@pytest.mark.parametrize("rescale_acts_by_decoder_norm", [True, False])
def test_identity_basis_reproduces_plain_matryoshka_readout(
    rescale_acts_by_decoder_norm: bool,
):
    sae = SpectralMatryoshkaBatchTopKTrainingSAE(
        build_spectral_matryoshka_cfg(
            rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm
        )
    )
    random_params(sae)
    sae.spectral_basis = torch.eye(16)
    plain_sae = build_matching_plain_matryoshka_sae(sae)

    feature_acts = torch.randn(10, 16)
    spectral_levels = list(sae._iterable_decode(feature_acts, True))
    plain_levels = list(plain_sae._iterable_decode(feature_acts, True))

    assert [width for width, _ in spectral_levels] == [4, 8, 16]
    for (_, spectral_out), (_, plain_out) in zip(spectral_levels, plain_levels):
        assert_close(spectral_out, plain_out)


def test_training_step_updates_basis_with_power_iteration():
    sae = SpectralMatryoshkaBatchTopKTrainingSAE(build_spectral_matryoshka_cfg())
    random_params(sae)
    U_prev = sae.spectral_basis.clone()
    W_proj = sae.W_proj.detach().clone()

    with torch.no_grad():
        output = sae.training_forward_pass(build_train_step_input(torch.randn(10, 8)))

    expected, _ = torch.linalg.qr(W_proj @ W_proj.T @ U_prev)
    assert_close(sae.spectral_basis, expected)
    assert_close(sae.spectral_basis.T @ sae.spectral_basis, torch.eye(16), atol=1e-5)
    assert not sae.spectral_basis.requires_grad
    assert "spectral_basis_drift" in output.metrics


def test_inner_losses_train_the_spectral_projection():
    sae = SpectralMatryoshkaBatchTopKTrainingSAE(build_spectral_matryoshka_cfg())
    random_params(sae)

    output = sae.training_forward_pass(build_train_step_input(torch.randn(10, 8)))
    output.loss.backward()

    assert sae.W_proj.grad is not None
    assert sae.W_proj.grad.abs().sum() > 0
    assert torch.isfinite(sae.W_proj.grad).all()


def test_power_iteration_orders_basis_by_feature_importance():
    d_sae = 6
    lambdas = torch.tensor([9.0, 4.0, 2.0, 1.0, 0.5, 0.25])
    # assign the spectrum to latent coordinates in a scrambled order, so the
    # basis must actually sort the coordinates rather than keep them
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    coord_lambdas = torch.zeros(d_sae)
    coord_lambdas[perm] = lambdas

    sae = SpectralMatryoshkaBatchTopKTrainingSAE(
        build_spectral_matryoshka_cfg(
            d_sae=d_sae, matryoshka_widths=[2, 4, 6], spectral_proj_rank_multiplier=1
        )
    )
    sae.spectral_basis = torch.linalg.qr(torch.randn(d_sae, d_sae))[0]
    sae.W_proj.data = torch.diag(torch.sqrt(coord_lambdas))

    with torch.no_grad():
        for _ in range(60):
            sae.spectral_basis = sae._next_spectral_basis()[0].detach()

    expected = torch.eye(d_sae)[:, perm]
    assert_close(sae.spectral_basis.abs(), expected.abs(), atol=1e-4)


def test_matryoshka_aux_loss_runs_with_spectral_readout():
    sae = SpectralMatryoshkaBatchTopKTrainingSAE(
        build_spectral_matryoshka_cfg(use_matryoshka_aux_loss=True)
    )
    random_params(sae)

    dead_neuron_mask = torch.zeros(16, dtype=torch.bool)
    dead_neuron_mask[0:4] = True
    dead_neuron_mask[12:16] = True

    output = sae.training_forward_pass(
        build_train_step_input(torch.randn(10, 8), dead_neuron_mask)
    )

    assert output.losses["auxiliary_reconstruction_loss"].item() > 0
    assert torch.isfinite(output.loss)


def test_save_inference_model_freezes_and_ships_the_spectral_basis(tmp_path: Path):
    sae = SpectralMatryoshkaBatchTopKTrainingSAE(build_spectral_matryoshka_cfg())
    random_params(sae)

    # run test data through repeatedly so the batchtopk threshold converges,
    # mirroring how the saved JumpReLU SAE is produced
    sae_in = torch.randn(30, 8)
    step_input = build_train_step_input(sae_in)
    with torch.no_grad():
        for _ in range(500):
            sae.training_forward_pass(step_input)

    basis_before_save = sae.spectral_basis.clone()
    W_proj = sae.W_proj.detach().clone()
    expected_frozen, _ = torch.linalg.qr(W_proj @ W_proj.T @ basis_before_save)

    sae.save_inference_model(str(tmp_path))

    assert_close(sae.spectral_basis, expected_frozen)
    frozen_basis = load_spectral_basis(tmp_path)
    assert frozen_basis.shape == (16, 16)
    assert_close(frozen_basis, expected_frozen)
    assert_close(frozen_basis.T @ frozen_basis, torch.eye(16), atol=1e-5)

    inference_sae = SAE.load_from_disk(str(tmp_path), device="cpu")
    assert isinstance(inference_sae, JumpReLUSAE)

    training_feature_acts, _ = sae.encode_with_hidden_pre(sae_in)
    training_sae_out = sae.decode(training_feature_acts)
    inference_feature_acts = inference_sae.encode(sae_in)
    inference_sae_out = inference_sae.decode(inference_feature_acts)
    assert_close(training_feature_acts, inference_feature_acts)
    assert_close(training_sae_out, inference_sae_out)
