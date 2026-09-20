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
from sae_lens.training.sae_trainer import SAETrainer
from tests.helpers import (
    assert_close,
    build_topafa_runner_cfg,
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


def build_orthogonal_topafa_sae(**kwargs: Any) -> TopAFATrainingSAE:
    """
    A (d_in=4, d_sae=4) SAE with W_enc = W_dec = identity and zero biases, so the
    pre-activations equal the centered input and the decoder norms are all 1.
    """
    cfg = build_topafa_sae_training_cfg(
        d_in=4,
        d_sae=4,
        **kwargs,
    )
    sae = TopAFATrainingSAE(cfg)
    with torch.no_grad():
        sae.W_enc.copy_(torch.eye(4))
        sae.W_dec.copy_(torch.eye(4))
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    return sae


def build_step_input(
    sae_in: torch.Tensor,
    dead_neuron_mask: torch.Tensor | None = None,
    is_logging_step: bool = False,
) -> TrainStepInput:
    return TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=dead_neuron_mask,
        n_training_steps=0,
        is_logging_step=is_logging_step,
    )


def test_topafa_activations_keeps_features_matching_input_norm():
    hidden_pre = torch.tensor([[3.0, 2.0, 1.0, 0.5]])
    # cumulative norms are [3, sqrt(13), sqrt(14)] (plus the sentinel)

    # an input norm of sqrt(13) is matched exactly by the top 2 features
    out = topafa_activations(hidden_pre, torch.eye(4), torch.tensor([[13.0**0.5, 0.0]]))
    assert_close(out, torch.tensor([[3.0, 2.0, 0.0, 0.0]]))

    # an input norm of sqrt(14) is matched exactly by the top 3 features
    out = topafa_activations(hidden_pre, torch.eye(4), torch.tensor([[14.0**0.5, 0.0]]))
    assert_close(out, torch.tensor([[3.0, 2.0, 1.0, 0.0]]))


def test_topafa_activations_matches_in_norm_space_not_squared_norm_space():
    hidden_pre = torch.tensor([[3.0, 2.0, 1.0, 0.5]])
    # an input squared norm of 11 is a tie in squared-norm space (|11 - 9| == |11 - 13|,
    # argmin picks the first), but in norm space sqrt(13) is closer to sqrt(11) than 3
    out = topafa_activations(
        hidden_pre, torch.eye(4), torch.tensor([[11.0**0.5, 0.0]])
    )
    assert_close(out, torch.tensor([[3.0, 2.0, 0.0, 0.0]]))


def test_topafa_activations_never_keeps_the_full_dictionary():
    hidden_pre = torch.tensor([[3.0, 2.0, 1.0, 0.5]])
    # an input norm far beyond the total cumulative norm (sqrt(14.5)) must not select
    # all features: the last cumulative entry is a large sentinel, so the best
    # remaining match is the top 3
    out = topafa_activations(hidden_pre, torch.eye(4), torch.tensor([[10.0, 0.0]]))
    assert_close(out, torch.tensor([[3.0, 2.0, 1.0, 0.0]]))


def test_topafa_activations_sentinel_is_large_rather_than_infinite():
    hidden_pre = torch.tensor([[3.0, 2.0, 1.0, 0.5]])
    # the reference sentinel is 1e8 (in squared space), so an input norm beyond its
    # square root selects the full dictionary
    out = topafa_activations(hidden_pre, torch.eye(4), torch.tensor([[1e5, 0.0]]))
    assert_close(out, hidden_pre)


def test_TopAFATrainingSAE_l0_adapts_to_the_input_not_its_norm():
    sae = build_orthogonal_topafa_sae()

    # both inputs have unit norm, but one concentrates its energy in a single
    # feature while the other spreads it evenly over all four
    concentrated = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    spread = torch.tensor([[0.5, 0.5, 0.5, 0.5]])

    concentrated_acts = sae.encode(concentrated)
    spread_acts = sae.encode(spread)

    assert_close(concentrated_acts, concentrated)
    # cumulative norms are [0.5, sqrt(0.5), sqrt(0.75)] against a target of 1.0
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


def test_TopAFA_encoders_ignore_b_enc():
    training_sae = build_orthogonal_topafa_sae()
    inference_cfg = TopAFASAEConfig(d_in=4, d_sae=4, normalize_activations="none")
    inference_sae = TopAFASAE(inference_cfg)
    with torch.no_grad():
        inference_sae.W_enc.copy_(torch.eye(4))
        inference_sae.W_dec.copy_(torch.eye(4))
        inference_sae.b_dec.zero_()

    x = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
    for sae in (training_sae, inference_sae):
        with torch.no_grad():
            expected = sae.encode(x)
            sae.b_enc.fill_(10.0)
        assert_close(sae.encode(x), expected)


@pytest.mark.parametrize("apply_b_dec_to_input", [True, False])
def test_TopAFATrainingSAE_always_centers_the_input_with_b_dec(
    apply_b_dec_to_input: bool,
):
    sae = build_orthogonal_topafa_sae(apply_b_dec_to_input=apply_b_dec_to_input)
    with torch.no_grad():
        sae.b_dec.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    feature_acts = sae.encode(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    # the encoder sees [0, 0, 0, 0], so nothing is activated
    assert_close(feature_acts, torch.zeros(1, 4))
    assert_close(sae.decode(feature_acts), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


def test_TopAFATrainingSAE_standardizes_each_input_by_default():
    cfg = build_topafa_sae_training_cfg(normalize_activations="layer_norm")
    sae = TopAFATrainingSAE(cfg)

    x = torch.arange(1.0, cfg.d_in + 1.0).unsqueeze(0)
    # the reference's input_unit_norm standardizes each input, making the encoding
    # invariant to per-input affine maps
    assert_close(sae.encode(x), sae.encode(3.0 * x - 100.0))


def test_TopAFATrainingSAE_calculates_the_norm_matching_loss():
    sae = build_orthogonal_topafa_sae(afa_loss_coefficient=1.0)
    sae_in = torch.tensor([[0.5, 0.5, 0.5, 0.5]])

    with torch.no_grad():
        output = sae.training_forward_pass(build_step_input(sae_in))

    # selected activations are [0.5, 0.5, 0.5, 0], whose norm is sqrt(3/4),
    # against an input norm of 1.0
    expected_afa_loss = (math.sqrt(0.75) - 1.0) ** 2
    assert output.losses["afa_loss"].item() == pytest.approx(expected_afa_loss)
    assert "mse_loss" in output.losses
    assert "auxiliary_reconstruction_loss" in output.losses


def test_TopAFATrainingSAE_norm_matching_loss_targets_the_uncentered_input():
    sae = build_orthogonal_topafa_sae(afa_loss_coefficient=1 / 16)
    with torch.no_grad():
        sae.b_dec.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    sae_in = torch.tensor([[2.0, 0.0, 0.0, 0.0]])

    with torch.no_grad():
        output = sae.training_forward_pass(build_step_input(sae_in))

    # the selection targets ||sae_in - b_dec|| = 1.0, keeping a single unit feature,
    # while the norm-matching loss targets ||sae_in|| = 2.0
    assert_close(output.feature_acts, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert (
        output.losses["afa_loss"].item() == pytest.approx((1 / 16) * (1.0 - 2.0) ** 2)
    )
    assert output.losses["mse_loss"].item() == pytest.approx(0.0)


def test_TopAFATrainingSAE_norm_matching_loss_is_zero_when_norms_match():
    sae = build_orthogonal_topafa_sae(afa_loss_coefficient=1.0)
    # a single unit feature has ||f||_2 = ||sae_in||_2 = 1.0
    sae_in = torch.tensor([[2.0, 0.0, 0.0, 0.0]])

    with torch.no_grad():
        output = sae.training_forward_pass(build_step_input(sae_in))

    assert output.losses["afa_loss"].item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("top_k_aux", "expected_squared_error"),
    [(512, 1.3125), (1, 1.0625)],
)
def test_TopAFATrainingSAE_dead_neuron_aux_loss_matches_reference_flavor(
    top_k_aux: int, expected_squared_error: float
):
    sae = build_orthogonal_topafa_sae(top_k_aux=top_k_aux)
    sae_in = torch.tensor([[3.0, 2.0, 1.0, 0.5]])

    # features 1 and 2 are dead; the selection keeps [3, 2, 1, 0], so the residual
    # is [0, 0, 0, 0.5] and the dead features reconstruct it with their own
    # (relu'd) pre-activations
    dead_neuron_mask = torch.tensor([False, True, True, False])
    with torch.no_grad():
        output = sae.training_forward_pass(
            build_step_input(sae_in, dead_neuron_mask=dead_neuron_mask)
        )

    assert output.losses["auxiliary_reconstruction_loss"].item() == pytest.approx(
        expected_squared_error / 32
    )
    # afa_loss_coefficient defaults to 0.0, as in the reference config
    assert output.losses["afa_loss"].item() == pytest.approx(0.0)


def test_TopAFATrainingSAE_make_decoder_weights_and_grad_unit_norm():
    sae = build_orthogonal_topafa_sae()
    with torch.no_grad():
        sae.W_dec.data = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    sae.W_dec.grad = torch.ones(4, 4)

    sae.make_decoder_weights_and_grad_unit_norm()

    # rows are re-normalized to unit norm
    assert_close(sae.W_dec, torch.eye(4))
    # the radial gradient component ((grad . row) * row) is removed: each row of the
    # ones gradient has a radial component of exactly its own axis
    assert_close(sae.W_dec.grad, torch.ones(4, 4) - torch.eye(4))


def test_sae_trainer_keeps_topafa_decoder_unit_norm():
    # lr=0 so the optimizer step after the re-normalization changes nothing
    runner_cfg = build_topafa_runner_cfg(d_in=8, d_sae=16, lr=0.0)
    sae = TopAFATrainingSAE(runner_cfg.sae)
    trainer = SAETrainer(
        cfg=runner_cfg.to_sae_trainer_config(),
        sae=sae,
        data_provider=iter([]),
    )
    with torch.no_grad():
        sae.W_dec.data *= torch.arange(
            1, sae.cfg.d_sae + 1, dtype=sae.dtype
        ).unsqueeze(1)
    assert not torch.allclose(sae.W_dec.norm(dim=-1), torch.ones(sae.cfg.d_sae))

    trainer.step(torch.randn(32, sae.cfg.d_in))

    assert_close(sae.W_dec.norm(dim=-1), torch.ones(sae.cfg.d_sae))


def test_TopAFATrainingSAE_logs_mean_l0_every_step():
    sae = build_orthogonal_topafa_sae()
    # rows have an l0 of 1 and an l0 of 3 respectively
    sae_in = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])

    with torch.no_grad():
        output = sae.training_forward_pass(build_step_input(sae_in))

    # the mean l0 is computed on every step, not only on logging steps
    assert output.metrics["l0_norm"].item() == pytest.approx(2.0)


def test_TopAFATrainingSAE_training_forward_pass_hooks():
    sae = TopAFATrainingSAE(build_topafa_sae_training_cfg(d_in=8, d_sae=16))
    x = torch.randn(32, sae.cfg.d_in)
    step_input = build_step_input(x)
    # Top-AFA always centers the input with b_dec and does not use b_enc, so the
    # hook captures the centered pre-activation
    raw_hidden_pre = (sae.process_sae_in(x) - sae.b_dec) @ sae.W_enc

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
    assert "top_k_aux" not in inference_config
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


def test_TopAFA_configs_default_to_the_reference_settings():
    training_cfg = TopAFATrainingSAEConfig(d_in=8, d_sae=16)
    assert training_cfg.afa_loss_coefficient == 0.0
    assert training_cfg.top_k_aux == 512
    assert training_cfg.aux_loss_coefficient == pytest.approx(1 / 32)
    assert training_cfg.decoder_init_norm == 1.0
    assert training_cfg.normalize_activations == "layer_norm"
    assert training_cfg.rescale_acts_by_decoder_norm is False
    assert TopAFASAEConfig(d_in=8, d_sae=16).normalize_activations == "layer_norm"

    # a unit-norm decoder, as initialized in the reference
    sae = TopAFATrainingSAE(training_cfg)
    assert_close(sae.W_dec.norm(dim=-1), torch.ones(training_cfg.d_sae))
    assert_close(sae.W_enc, sae.W_dec.T)


def test_TopAFATrainingSAE_rejects_rescale_acts_by_decoder_norm():
    with pytest.raises(ValueError, match="rescale_acts_by_decoder_norm"):
        TopAFATrainingSAEConfig(
            d_in=8,
            d_sae=16,
            rescale_acts_by_decoder_norm=True,
        )


def test_TopAFATrainingSAE_fold_w_dec_norm_is_not_supported():
    sae = build_orthogonal_topafa_sae()
    with pytest.raises(NotImplementedError):
        sae.fold_W_dec_norm()


def test_TopAFATrainingSAE_save_and_load_inference_sae(tmp_path: Path) -> None:
    cfg = build_topafa_sae_training_cfg(d_in=8, d_sae=32)
    training_sae = TopAFATrainingSAE(cfg)
    random_params(training_sae)

    sae_in = torch.randn(30, training_sae.cfg.d_in)
    original_W_dec = training_sae.W_dec.data.clone()

    model_path = str(tmp_path)
    training_sae.save_inference_model(model_path)
    assert os.path.exists(model_path)

    inference_sae = SAE.load_from_disk(model_path, device="cpu")
    assert isinstance(inference_sae, TopAFASAE)

    # decoder norms are no longer folded into the weights when saving
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
