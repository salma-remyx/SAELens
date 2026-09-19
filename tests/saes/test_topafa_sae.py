import math
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from sae_lens.registry import get_sae_class, get_sae_training_class
from sae_lens.saes.sae import SAE, TrainStepInput
from sae_lens.saes.topafa_sae import (
    TopAFASAE,
    TopAFASAEConfig,
    TopAFATrainingSAE,
    TopAFATrainingSAEConfig,
    topafa_activations,
)
from tests.helpers import (
    assert_close,
    random_params,
    run_training_forward_pass_with_cache,
)


def build_topafa_sae_training_cfg(**kwargs: Any) -> TopAFATrainingSAEConfig:
    defaults: dict[str, Any] = {
        "d_in": 64,
        "d_sae": 256,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
        "decoder_init_norm": 0.1,
        "apply_b_dec_to_input": False,
    }
    return TopAFATrainingSAEConfig(**{**defaults, **kwargs})


def build_orthogonal_topafa_sae() -> TopAFATrainingSAE:
    """
    A (d_in=4, d_sae=4) SAE with W_enc = W_dec = identity and zero biases, so the
    pre-activations equal the input and the decoder norms are all 1.
    """
    cfg = build_topafa_sae_training_cfg(
        d_in=4,
        d_sae=4,
        apply_b_dec_to_input=False,
    )
    sae = TopAFATrainingSAE(cfg)
    with torch.no_grad():
        sae.W_enc.copy_(torch.eye(4))
        sae.W_dec.copy_(torch.eye(4))
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    return sae


def test_topafa_activations_keeps_features_matching_input_norm():
    hidden_pre = torch.tensor([[3.0, 2.0, 1.0, 0.5]])
    # cumulative squared norms are [9, 13, 14, 14.5]

    # input norm^2 = 13 is matched exactly by the top 2 features
    out = topafa_activations(hidden_pre, torch.tensor([[13.0**0.5, 0.0]]))
    assert_close(out, torch.tensor([[3.0, 2.0, 0.0, 0.0]]))

    # input norm^2 = 14 is matched exactly by the top 3 features
    out = topafa_activations(hidden_pre, torch.tensor([[14.0**0.5, 0.0]]))
    assert_close(out, torch.tensor([[3.0, 2.0, 1.0, 0.0]]))


def test_topafa_activations_never_keeps_the_full_dictionary():
    hidden_pre = torch.tensor([[3.0, 2.0, 1.0, 0.5]])
    # an input norm far beyond the total cumulative norm (14.5) must not select
    # all features: the last cumulative entry is an infinite sentinel, so the
    # best remaining match is the top 3
    out = topafa_activations(hidden_pre, torch.tensor([[10.0, 0.0]]))
    assert_close(out, torch.tensor([[3.0, 2.0, 1.0, 0.0]]))


def test_TopAFATrainingSAE_l0_adapts_to_the_input_not_its_norm():
    sae = build_orthogonal_topafa_sae()

    # both inputs have unit norm, but one concentrates its energy in a single
    # feature while the other spreads it evenly over all four
    concentrated = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    spread = torch.tensor([[0.5, 0.5, 0.5, 0.5]])

    concentrated_acts = sae.encode(concentrated)
    spread_acts = sae.encode(spread)

    assert_close(concentrated_acts, concentrated)
    # cumulative norms are [0.25, 0.5, 0.75] against a target of 1.0
    assert_close(spread_acts, torch.tensor([[0.5, 0.5, 0.5, 0.0]]))


def test_TopAFATrainingSAE_l0_adapts_per_input_for_random_parameters():
    cfg = build_topafa_sae_training_cfg(d_in=32, d_sae=128)
    sae = TopAFATrainingSAE(cfg)
    random_params(sae)

    sae_in = torch.randn(1000, cfg.d_in)
    feature_acts, _ = sae.encode_with_hidden_pre(sae_in)

    l0 = (feature_acts > 0).sum(-1)
    # the sentinel guarantees the full dictionary is never selected
    assert int(l0.max()) <= cfg.d_sae - 1
    # sparsity adapts per input rather than being a constant k
    assert l0.unique().numel() > 1


def test_TopAFATrainingSAE_calculates_the_norm_matching_loss():
    sae = build_orthogonal_topafa_sae()
    sae_in = torch.tensor([[0.5, 0.5, 0.5, 0.5]])

    step_input = TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=False,
    )
    with torch.no_grad():
        output = sae.training_forward_pass(step_input)

    # selected activations are [0.5, 0.5, 0.5, 0], whose norm is sqrt(3/4),
    # against an input norm of 1.0
    expected_afa_loss = sae.cfg.afa_loss_coefficient * (math.sqrt(0.75) - 1.0) ** 2
    assert output.losses["afa_loss"].item() == pytest.approx(expected_afa_loss)
    assert "mse_loss" in output.losses
    assert "auxiliary_reconstruction_loss" in output.losses


def test_TopAFATrainingSAE_norm_matching_loss_is_zero_when_norms_match():
    sae = build_orthogonal_topafa_sae()
    # a single unit feature has ||f||_2 = ||sae_in||_2 = 1.0
    sae_in = torch.tensor([[2.0, 0.0, 0.0, 0.0]])

    step_input = TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=False,
    )
    with torch.no_grad():
        output = sae.training_forward_pass(step_input)

    assert output.losses["afa_loss"].item() == pytest.approx(0.0)


def test_TopAFATrainingSAE_logs_l0_metrics_on_logging_steps():
    sae = build_orthogonal_topafa_sae()
    # rows have an l0 of 1 and an l0 of 3 respectively
    sae_in = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])

    step_input = TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=True,
    )
    with torch.no_grad():
        output = sae.training_forward_pass(step_input)

    assert output.metrics["mean_l0"].item() == pytest.approx(2.0)
    assert output.metrics["min_l0"].item() == pytest.approx(1.0)
    assert output.metrics["max_l0"].item() == pytest.approx(3.0)


def test_TopAFATrainingSAE_training_forward_pass_hooks():
    sae = TopAFATrainingSAE(build_topafa_sae_training_cfg(d_in=8, d_sae=16))
    x = torch.randn(32, sae.cfg.d_in)
    step_input = TrainStepInput(
        sae_in=x,
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=False,
    )
    # topafa rescales hidden_pre by the decoder norm after hook_sae_acts_pre fires,
    # so the hook captures the raw pre-activation, not train_step_output.hidden_pre
    raw_hidden_pre = sae.process_sae_in(x) @ sae.W_enc + sae.b_enc

    train_step_output, train_cache = run_training_forward_pass_with_cache(
        sae, step_input
    )
    assert train_cache["hook_sae_input"].equal(x)
    assert train_cache["hook_sae_acts_pre"].equal(raw_hidden_pre)
    assert train_cache["hook_sae_acts_post"].equal(train_step_output.feature_acts)
    assert train_cache["hook_sae_recons"].equal(train_step_output.sae_out)

    # Verify training output matches a regular run_with_cache forward pass
    _, cache = sae.run_with_cache(x)
    assert train_cache["hook_sae_acts_post"].equal(cache["hook_sae_acts_post"])
    assert train_cache["hook_sae_recons"].equal(cache["hook_sae_recons"])


def test_TopAFATrainingSAE_get_inference_sae_cfg_dict():
    cfg = build_topafa_sae_training_cfg()
    sae = TopAFATrainingSAE(cfg)

    inference_config = sae.cfg.get_inference_sae_cfg_dict()

    assert inference_config["architecture"] == "topafa"
    assert inference_config["d_in"] == cfg.d_in
    assert inference_config["d_sae"] == cfg.d_sae
    assert inference_config["dtype"] == cfg.dtype
    assert inference_config["device"] == cfg.device

    # training-only fields should not leak into the inference config
    assert "afa_loss_coefficient" not in inference_config
    assert "k" not in inference_config


def test_TopAFATrainingSAE_initialization():
    cfg = build_topafa_sae_training_cfg()
    sae = TopAFATrainingSAE(cfg)
    assert isinstance(sae.W_enc, nn.Parameter)
    assert isinstance(sae.W_dec, nn.Parameter)
    assert isinstance(sae.b_enc, nn.Parameter)
    assert isinstance(sae.b_dec, nn.Parameter)

    assert sae.W_enc.shape == (cfg.d_in, cfg.d_sae)
    assert sae.W_dec.shape == (cfg.d_sae, cfg.d_in)
    assert sae.b_enc.shape == (cfg.d_sae,)
    assert sae.b_dec.shape == (cfg.d_in,)


@pytest.mark.parametrize("rescale_acts_by_decoder_norm", [True, False])
def test_TopAFATrainingSAE_save_and_load_inference_sae(
    tmp_path: Path, rescale_acts_by_decoder_norm: bool
) -> None:
    cfg = build_topafa_sae_training_cfg(
        d_in=8,
        d_sae=32,
        rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
    )
    training_sae = TopAFATrainingSAE(cfg)
    random_params(training_sae)

    sae_in = torch.randn(30, training_sae.cfg.d_in)
    original_W_dec = training_sae.W_dec.data.clone()

    model_path = str(tmp_path)
    training_sae.save_inference_model(model_path)
    assert os.path.exists(model_path)

    inference_sae = SAE.load_from_disk(model_path, device="cpu")
    assert isinstance(inference_sae, TopAFASAE)

    if rescale_acts_by_decoder_norm:
        # decoder norms are folded into the weights when saving
        assert_close(
            inference_sae.W_dec.norm(dim=-1), torch.ones(training_sae.cfg.d_sae)
        )
        assert not torch.allclose(inference_sae.W_dec, original_W_dec)
    else:
        assert_close(inference_sae.W_dec, original_W_dec)

    # the inference SAE must reproduce the training-time selection exactly
    training_feature_acts, _ = training_sae.encode_with_hidden_pre(sae_in)
    inference_feature_acts = inference_sae.encode(sae_in)
    assert_close(training_feature_acts, inference_feature_acts)
    assert_close(
        training_sae.decode(training_feature_acts),
        inference_sae.decode(inference_feature_acts),
    )
    assert_close(training_sae(sae_in), inference_sae(sae_in))


def test_topafa_is_registered_in_the_sae_registries():
    assert get_sae_class("topafa") == (TopAFASAE, TopAFASAEConfig)
    assert get_sae_training_class("topafa") == (
        TopAFATrainingSAE,
        TopAFATrainingSAEConfig,
    )
