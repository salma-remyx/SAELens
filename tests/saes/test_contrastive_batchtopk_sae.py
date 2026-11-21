from typing import Any

import pytest
import torch

from sae_lens.registry import get_sae_training_class
from sae_lens.saes.contrastive_batchtopk_sae import (
    ContrastiveBatchTopKTrainingSAE,
    ContrastiveBatchTopKTrainingSAEConfig,
)
from sae_lens.saes.sae import TrainStepInput
from tests.helpers import (
    assert_close,
    build_contrastive_batchtopk_sae_training_cfg,
    random_params,
)


def _build_sae(**kwargs: Any) -> ContrastiveBatchTopKTrainingSAE:
    config = build_contrastive_batchtopk_sae_training_cfg(
        d_in=4,
        d_sae=8,
        k=kwargs.pop("k", 4),
        device="cpu",
        **kwargs,
    )
    return ContrastiveBatchTopKTrainingSAE(config)


def _train_step_input(sae_in: torch.Tensor) -> TrainStepInput:
    return TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=False,
    )


def test_contrastive_batchtopk_is_registered():
    sae_class, config_class = get_sae_training_class("contrastive_batchtopk")
    assert sae_class is ContrastiveBatchTopKTrainingSAE
    assert config_class is ContrastiveBatchTopKTrainingSAEConfig
    assert config_class.architecture() == "contrastive_batchtopk"


def test_contrastive_loss_matched_codes():
    sae = _build_sae(contrastive_temperature=0.5)
    # one-hot codes where query i matches document i: cosine similarity is 1 on
    # the diagonal and 0 elsewhere, so each row's logits are [1/tau, 0, 0]
    codes = torch.eye(3, 8)
    feature_acts = torch.cat([codes, codes])

    loss = sae.calculate_contrastive_loss(feature_acts)

    expected = torch.log1p(2 * torch.exp(torch.tensor(-1 / 0.5)))
    assert loss.item() == pytest.approx(expected.item())


def test_contrastive_loss_mismatched_pairing_is_higher_than_matched():
    sae = _build_sae(contrastive_temperature=0.5)
    queries = torch.eye(3, 8)
    # rolling the documents breaks the pairing: the diagonal similarity becomes 0
    # and each query's best match lands off-diagonal
    mismatched_docs = torch.roll(torch.eye(3, 8), shifts=1, dims=0)

    matched_loss = sae.calculate_contrastive_loss(torch.cat([queries, queries]))
    mismatched_loss = sae.calculate_contrastive_loss(
        torch.cat([queries, mismatched_docs])
    )

    expected_mismatched = torch.log1p(torch.exp(torch.tensor(1 / 0.5)) + 1)
    assert mismatched_loss.item() > matched_loss.item()
    assert mismatched_loss.item() == pytest.approx(expected_mismatched.item())


def test_training_forward_pass_adds_contrastive_loss_to_total_loss():
    coefficient = 2e-3
    sae = _build_sae(contrastive_loss_coefficient=coefficient)
    random_params(sae)
    sae_in = torch.randn(6, 4)

    output = sae.training_forward_pass(_train_step_input(sae_in))

    assert "contrastive_loss" in output.losses
    expected_total = (
        output.losses["mse_loss"]
        + output.losses["auxiliary_reconstruction_loss"]
        + output.losses["contrastive_loss"]
    )
    assert_close(output.loss, expected_total)
    manual_infonce = sae.calculate_contrastive_loss(output.feature_acts)
    assert output.losses["contrastive_loss"].item() == pytest.approx(
        coefficient * manual_infonce.item()
    )


def test_zero_contrastive_loss_coefficient_removes_the_term():
    sae = _build_sae(contrastive_loss_coefficient=0.0)
    random_params(sae)
    sae_in = torch.randn(6, 4)

    output = sae.training_forward_pass(_train_step_input(sae_in))

    assert output.losses["contrastive_loss"].item() == 0.0
    assert_close(
        output.loss,
        output.losses["mse_loss"] + output.losses["auxiliary_reconstruction_loss"],
    )


def test_contrastive_loss_requires_even_batch():
    sae = _build_sae()
    with pytest.raises(ValueError, match="even number of rows"):
        sae.calculate_contrastive_loss(torch.randn(5, 8))


def test_contrastive_loss_gradients_flow_to_encoder_only():
    sae = _build_sae(
        k=8,
        contrastive_loss_coefficient=1.0,
        contrastive_temperature=0.5,
        rescale_acts_by_decoder_norm=False,
    )
    random_params(sae)
    # correlated query/document pairs so the sparse codes overlap
    documents = torch.randn(4, 4)
    queries = documents + 0.05 * torch.randn(4, 4)

    output = sae.training_forward_pass(
        _train_step_input(torch.cat([queries, documents]))
    )
    output.losses["contrastive_loss"].backward()

    assert sae.W_enc.grad is not None
    assert sae.W_enc.grad.abs().sum().item() > 0
    # the contrastive loss never touches the decoder
    assert sae.W_dec.grad is None
