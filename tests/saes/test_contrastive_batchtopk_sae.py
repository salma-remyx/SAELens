import math
from pathlib import Path
from typing import cast

import pytest
import torch

from sae_lens.registry import (
    SAE_TRAINING_CLASS_REGISTRY,
    get_sae_training_class,
    register_sae_training_class,
)
from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.contrastive_batchtopk_sae import (
    ContrastiveBatchTopKTrainingSAE,
    ContrastiveBatchTopKTrainingSAEConfig,
    calculate_infonce_loss,
)
from sae_lens.saes.jumprelu_sae import JumpReLUSAE
from sae_lens.saes.sae import SAE, TrainCoefficientConfig, TrainStepInput
from tests.helpers import assert_close, random_params


def _build_step_input(
    sae_in: torch.Tensor, coefficients: dict[str, float]
) -> TrainStepInput:
    return TrainStepInput(
        sae_in=sae_in,
        coefficients=coefficients,
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=False,
    )


def test_calculate_infonce_loss_matches_hand_computed_value():
    # input neighbours: 0 <-> 1 (cos 0.8), 1 <-> 2 (cos 0.6), 0 <-/-> 2 (cos 0)
    sae_in = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    # codes preserve that structure: 0 and 1 aligned, 2 orthogonal
    feature_acts = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # with temperature 1 the per-sample losses are:
    # anchor 0 (positive 1): log(1 + e^-1)
    # anchor 1 (positive 0): log(1 + e^-1)
    # anchor 2 (positive 1): log(2)
    expected = (2 * math.log(1 + math.exp(-1.0)) + math.log(2.0)) / 3

    loss = calculate_infonce_loss(sae_in, feature_acts, temperature=1.0)

    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_calculate_infonce_loss_penalizes_structure_breaking_codes():
    sae_in = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    # codes pair 0 with 2 instead of with their input neighbours
    broken_acts = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    # anchor 0 (positive 1, but code 0 aligns with 2): log(1 + e)
    # anchor 1 (positive 0, no candidate aligns): log(2)
    # anchor 2 (positive 1, but code 2 aligns with 0): log(1 + e)
    expected = (2 * math.log(1 + math.e) + math.log(2.0)) / 3

    loss = calculate_infonce_loss(sae_in, broken_acts, temperature=1.0)

    assert loss.item() == pytest.approx(expected, abs=1e-6)
    assert loss.item() > (2 * math.log(1 + math.exp(-1.0)) + math.log(2.0)) / 3


def test_calculate_infonce_loss_excludes_self_similarity():
    # duplicate inputs make each sample the other's nearest neighbour
    sae_in = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    # orthogonal codes: with only one in-batch candidate, the positive wins
    # with probability 1 -> zero loss. If self-similarity were included the
    # diagonal logit would dominate and the loss would be log(1 + e^(1/tau)).
    feature_acts = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    loss = calculate_infonce_loss(sae_in, feature_acts, temperature=0.1)

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_calculate_infonce_loss_prefers_codes_that_preserve_input_neighbourhoods():
    # n_pairs close pairs in input space, each pair given its own code
    # direction. Codes matching the pairing should be near-free to identify,
    # shuffled codes should be close to the random-guess entropy.
    n_pairs = 50
    d_in = 128
    centers = torch.randn(n_pairs, d_in)
    centers = centers / centers.norm(dim=-1, keepdim=True)
    sae_in = torch.cat([centers, centers]) + 0.05 * torch.randn(2 * n_pairs, d_in)
    pair_ids = torch.arange(n_pairs).repeat(2)
    codes = torch.zeros(2 * n_pairs, n_pairs)
    codes[torch.arange(2 * n_pairs), pair_ids] = 1.0
    shuffled_codes = codes[torch.randperm(2 * n_pairs)]

    preserved_loss = calculate_infonce_loss(sae_in, codes, temperature=0.1)
    shuffled_loss = calculate_infonce_loss(sae_in, shuffled_codes, temperature=0.1)

    assert preserved_loss.item() < 0.05
    assert shuffled_loss.item() > 5.0


def test_calculate_infonce_loss_handles_degenerate_batches_and_temperature():
    sae_in = torch.randn(1, 8)
    feature_acts = torch.randn(1, 16)

    assert calculate_infonce_loss(sae_in, feature_acts, temperature=0.1).item() == (
        pytest.approx(0.0, abs=1e-8)
    )

    with pytest.raises(ValueError):
        calculate_infonce_loss(torch.randn(4, 8), torch.randn(4, 16), temperature=0.0)
    with pytest.raises(ValueError):
        calculate_infonce_loss(torch.randn(4, 8), torch.randn(4, 16), temperature=-1.0)


def test_training_forward_pass_sums_infonce_into_total_loss():
    cfg = ContrastiveBatchTopKTrainingSAEConfig(
        d_in=8,
        d_sae=32,
        k=4,
        infonce_loss_coefficient=1.0,
        infonce_temperature=0.1,
    )
    sae = ContrastiveBatchTopKTrainingSAE(cfg)
    random_params(sae)
    sae_in = torch.randn(64, 8)

    output = sae.training_forward_pass(_build_step_input(sae_in, {"infonce": 1.0}))

    assert {"mse_loss", "auxiliary_reconstruction_loss", "infonce_loss"} <= set(
        output.losses
    )
    assert_close(
        output.loss, torch.stack(list(output.losses.values())).sum(), rtol=1e-5
    )
    expected_infonce = calculate_infonce_loss(
        sae_in, output.feature_acts, temperature=0.1
    )
    assert_close(output.losses["infonce_loss"], expected_infonce, rtol=1e-5)

    # without trainer-provided coefficients the config value is used
    fallback_output = sae.training_forward_pass(_build_step_input(sae_in, {}))
    assert_close(fallback_output.losses["infonce_loss"], expected_infonce, rtol=1e-5)


def test_infonce_loss_gradient_reaches_encoder_parameters():
    cfg = ContrastiveBatchTopKTrainingSAEConfig(
        d_in=8, d_sae=32, k=4, infonce_loss_coefficient=1.0
    )
    sae = ContrastiveBatchTopKTrainingSAE(cfg)
    random_params(sae)
    sae_in = torch.randn(64, 8)

    feature_acts, hidden_pre = sae.encode_with_hidden_pre(sae_in)
    losses = sae.calculate_aux_loss(
        step_input=_build_step_input(sae_in, {"infonce": 1.0}),
        feature_acts=feature_acts,
        hidden_pre=hidden_pre,
        sae_out=sae.decode(feature_acts),
    )
    losses["infonce_loss"].backward()

    assert sae.W_enc.grad is not None
    assert sae.W_enc.grad.abs().sum().item() > 0


def test_get_coefficients_declares_infonce_warmup():
    cfg = ContrastiveBatchTopKTrainingSAEConfig(
        d_in=8,
        d_sae=32,
        k=4,
        infonce_loss_coefficient=3.0,
        infonce_warm_up_steps=100,
    )
    sae = ContrastiveBatchTopKTrainingSAE(cfg)

    assert sae.get_coefficients() == {
        "infonce": TrainCoefficientConfig(value=3.0, warm_up_steps=100)
    }


def test_zero_infonce_coefficient_matches_plain_batchtopk():
    contrastive_cfg = ContrastiveBatchTopKTrainingSAEConfig(d_in=8, d_sae=32, k=4)
    batchtopk_cfg = BatchTopKTrainingSAEConfig(d_in=8, d_sae=32, k=4)
    contrastive_sae = ContrastiveBatchTopKTrainingSAE(contrastive_cfg)
    random_params(contrastive_sae)
    batchtopk_sae = BatchTopKTrainingSAE(batchtopk_cfg)
    batchtopk_sae.load_state_dict(contrastive_sae.state_dict())
    sae_in = torch.randn(64, 8)

    contrastive_output = contrastive_sae.training_forward_pass(
        _build_step_input(sae_in, {})
    )
    batchtopk_output = batchtopk_sae.training_forward_pass(
        _build_step_input(sae_in, {})
    )

    assert contrastive_output.losses["infonce_loss"].item() == pytest.approx(
        0.0, abs=1e-8
    )
    assert_close(
        contrastive_output.feature_acts, batchtopk_output.feature_acts, rtol=1e-5
    )
    assert_close(contrastive_output.loss, batchtopk_output.loss, rtol=1e-5)


def test_contrastive_batchtopk_registered_architecture_dispatch():
    if "contrastive_batchtopk" not in SAE_TRAINING_CLASS_REGISTRY:
        register_sae_training_class(
            "contrastive_batchtopk",
            ContrastiveBatchTopKTrainingSAE,
            ContrastiveBatchTopKTrainingSAEConfig,
        )

    training_cls, training_cfg_cls = get_sae_training_class("contrastive_batchtopk")

    assert training_cls is ContrastiveBatchTopKTrainingSAE
    assert training_cfg_cls is ContrastiveBatchTopKTrainingSAEConfig
    cfg_cls = cast("type[ContrastiveBatchTopKTrainingSAEConfig]", training_cfg_cls)
    sae = training_cls(cfg_cls(d_in=8, d_sae=16, k=2))
    assert sae.cfg.architecture() == "contrastive_batchtopk"
    assert sae.cfg.get_inference_sae_cfg_dict()["architecture"] == "jumprelu"


def test_contrastive_batchtopk_saves_as_jumprelu_inference_sae(tmp_path: Path):
    cfg = ContrastiveBatchTopKTrainingSAEConfig(
        d_in=8, d_sae=32, k=4, infonce_loss_coefficient=1.0
    )
    sae = ContrastiveBatchTopKTrainingSAE(cfg)
    random_params(sae)
    sae_in = torch.randn(32, 8)
    # run a real training step first so the full loss path is exercised
    sae.training_forward_pass(_build_step_input(sae_in, {"infonce": 1.0}))
    # place the threshold just below the smallest selected activation so the
    # saved JumpReLU SAE fires exactly the batch-topk features on this input
    feature_acts = sae.encode(sae_in)
    min_selected = feature_acts[feature_acts > 0].min()
    sae.topk_threshold = (min_selected * (1 - 1e-5)).to(torch.double)

    sae.save_inference_model(str(tmp_path))

    loaded = SAE.load_from_disk(str(tmp_path), device="cpu")
    assert isinstance(loaded, JumpReLUSAE)
    assert_close(
        loaded.threshold,
        sae.topk_threshold.item() * torch.ones_like(loaded.b_enc),
    )
    assert_close(loaded.encode(sae_in), feature_acts)
    assert_close(loaded(sae_in), sae(sae_in))
