from typing import Any
from unittest.mock import patch

import pytest
import torch

from sae_lens.registry import get_sae_training_class
from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.hopkins_batchtopk_sae import (
    HopkinsBatchTopKTrainingSAE,
    HopkinsBatchTopKTrainingSAEConfig,
    calculate_hopkins_statistic,
)
from sae_lens.saes.sae import TrainCoefficientConfig, TrainStepInput
from tests.helpers import assert_close, random_params


def _build_hopkins_batchtopk_cfg(**kwargs: Any) -> HopkinsBatchTopKTrainingSAEConfig:
    defaults: dict[str, Any] = {
        "d_in": 8,
        "d_sae": 16,
        "k": 8,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
        "decoder_init_norm": 0.1,
        "apply_b_dec_to_input": False,
    }
    return HopkinsBatchTopKTrainingSAEConfig(**{**defaults, **kwargs})


def _build_train_step_input(
    batch_size: int, coefficients: dict[str, float]
) -> TrainStepInput:
    return TrainStepInput(
        sae_in=torch.randn(batch_size, 8),
        coefficients=coefficients,
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=False,
    )


def test_hopkins_statistic_distinguishes_clustered_random_and_regular_topologies():
    dims = 4
    n = 8000
    sample_size = 400

    centers = 100.0 * torch.eye(dims)
    blob_noise = 0.05 * torch.randn(n, dims)
    clustered = centers.repeat_interleave(n // len(centers), dim=0) + blob_noise
    random = 100.0 * torch.rand(n, dims)
    grid = torch.cartesian_prod(*[10.0 * torch.arange(8)] * dims)

    h_clustered = calculate_hopkins_statistic(clustered, sample_size)
    h_random = calculate_hopkins_statistic(random, sample_size)
    h_regular = calculate_hopkins_statistic(grid, sample_size)

    assert h_clustered.item() > 0.95
    assert 0.45 < h_random.item() < 0.55
    assert h_regular.item() < 0.33
    assert h_regular < h_random < h_clustered


def test_hopkins_statistic_matches_manual_computation():
    x = torch.tensor([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0], [100.0, 100.0]])
    # patches make the random draws deterministic: x_tilde = first two rows and
    # y = [75, 75] for both rows ((min - max) * 0.25 + max of each dimension)
    with (
        patch("torch.rand", return_value=torch.full((2, 2), 0.25)),
        patch("torch.randperm", return_value=torch.arange(5)),
    ):
        h = calculate_hopkins_statistic(x, sample_size=2)

    # nearest neighbour of [75, 75] is [100, 100] (u = 25 for both rows), while
    # the nearest *other* neighbour of [0, 0] and [4, 0] is at distance 4
    assert h.item() == pytest.approx((25.0 + 25.0) / (25.0 + 25.0 + 4.0 + 4.0))


def test_hopkins_statistic_is_differentiable():
    x = torch.rand(64, 8)
    x.requires_grad_(True)
    h = calculate_hopkins_statistic(x, sample_size=8)
    h.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_minimizing_hopkins_loss_moves_codes_toward_clustered_topology():
    codes = torch.rand(400, 3)
    codes.requires_grad_(True)
    optimizer = torch.optim.Adam([codes], lr=0.05)
    h_initial = calculate_hopkins_statistic(codes.detach(), sample_size=20).item()

    for _ in range(300):
        optimizer.zero_grad()
        loss = (calculate_hopkins_statistic(codes, sample_size=20) - 0.97).abs()
        loss.backward()
        optimizer.step()

    h_final = calculate_hopkins_statistic(codes.detach(), sample_size=200).item()
    assert h_initial == pytest.approx(0.5, abs=0.15)
    assert h_final > 0.85


def test_config_rejects_hopkins_target_outside_unit_interval():
    with pytest.raises(ValueError, match="hopkins_target"):
        _build_hopkins_batchtopk_cfg(hopkins_target=1.5)


def test_hopkins_batchtopk_is_registered_in_training_registry():
    sae_class, cfg_class = get_sae_training_class("hopkins_batchtopk")
    assert sae_class is HopkinsBatchTopKTrainingSAE
    assert cfg_class is HopkinsBatchTopKTrainingSAEConfig


def test_get_coefficients_exposes_hopkins_train_coefficient():
    sae = HopkinsBatchTopKTrainingSAE(
        _build_hopkins_batchtopk_cfg(
            hopkins_loss_coefficient=0.25, hopkins_warm_up_steps=10
        )
    )
    assert sae.get_coefficients() == {
        "hopkins": TrainCoefficientConfig(value=0.25, warm_up_steps=10)
    }


def test_training_forward_pass_applies_hopkins_coefficient_and_adds_to_total_loss():
    sae = HopkinsBatchTopKTrainingSAE(_build_hopkins_batchtopk_cfg())
    random_params(sae)
    batch_size = 100
    step_input = _build_train_step_input(batch_size, {"hopkins": 2.0})

    # patches pin the random draws so the loss can be reproduced exactly
    with (
        patch("torch.rand", return_value=torch.full((5, 16), 0.25)),
        patch("torch.randperm", return_value=torch.arange(batch_size)),
    ):
        output = sae.training_forward_pass(step_input)
        expected_statistic = calculate_hopkins_statistic(
            output.feature_acts, sample_size=5
        )

    expected_hopkins_loss = 2.0 * (expected_statistic - 0.5).abs()
    assert_close(output.losses["hopkins_loss"], expected_hopkins_loss)
    assert_close(
        output.loss,
        output.losses["mse_loss"]
        + output.losses["auxiliary_reconstruction_loss"]
        + output.losses["hopkins_loss"],
    )


def test_zero_hopkins_coefficient_matches_plain_batchtopk():
    sae = HopkinsBatchTopKTrainingSAE(_build_hopkins_batchtopk_cfg())
    random_params(sae)
    plain_sae = BatchTopKTrainingSAE(
        BatchTopKTrainingSAEConfig(
            d_in=8, d_sae=16, k=8, device="cpu", apply_b_dec_to_input=False
        )
    )
    plain_sae.load_state_dict(sae.state_dict())

    step_input = _build_train_step_input(100, {"hopkins": 0.0})
    output = sae.training_forward_pass(step_input)
    plain_output = plain_sae.training_forward_pass(step_input)

    assert output.losses["hopkins_loss"].item() == pytest.approx(0.0)
    assert_close(output.loss, plain_output.loss)
    assert_close(output.sae_out, plain_output.sae_out)
    assert_close(output.feature_acts, plain_output.feature_acts)


def test_hopkins_loss_gradient_reaches_encoder_parameters():
    sae = HopkinsBatchTopKTrainingSAE(_build_hopkins_batchtopk_cfg())
    random_params(sae)
    step_input = _build_train_step_input(100, {"hopkins": 1.0})

    output = sae.training_forward_pass(step_input)
    output.loss.backward()

    assert sae.W_enc.grad is not None
    assert sae.W_enc.grad.abs().sum() > 0
