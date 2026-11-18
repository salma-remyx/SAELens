from pathlib import Path

import pytest
import torch

from sae_lens.registry import get_sae_class, get_sae_training_class
from sae_lens.saes.abstopk_sae import (
    AbsTopK,
    AbsTopKSAE,
    AbsTopKSAEConfig,
    AbsTopKTrainingSAE,
    AbsTopKTrainingSAEConfig,
    calculate_abstopk_aux_acts,
)
from sae_lens.saes.sae import SAE, SAEConfig, TrainStepInput
from sae_lens.saes.topk_sae import (
    TopK,
    TopKSAE,
    TopKTrainingSAE,
    calculate_topk_aux_acts,
)
from tests.helpers import (
    assert_close,
    assert_not_close,
    build_abstopk_sae_cfg,
    build_abstopk_sae_training_cfg,
    build_topk_sae_cfg,
    build_topk_sae_training_cfg,
    random_params,
)


def test_AbsTopK_keeps_largest_magnitude_activations_with_their_signs():
    # magnitudes: 3, 0.5, 5, 0.2, 4 -> top 2 are -5 and 4, signs preserved
    x = torch.tensor([[3.0, -0.5, -5.0, 0.2, 4.0]])
    assert_close(AbsTopK(k=2)(x), torch.tensor([[0.0, 0.0, -5.0, 0.0, 4.0]]))


def test_AbsTopK_agrees_with_TopK_for_nonnegative_inputs():
    x = torch.rand(64, 128)
    assert_close(AbsTopK(k=8)(x), TopK(k=8)(x))


def test_AbsTopK_keeps_exactly_k_original_values_per_row():
    x = torch.randn(200, 128)
    out = AbsTopK(k=16)(x)
    assert torch.equal((out != 0).sum(dim=-1), torch.full((200,), 16))
    assert_close(out[out != 0], x[out != 0])


def test_AbsTopKSAE_reconstructs_bidirectional_inputs_where_TopKSAE_cannot():
    # a single feature aligned with e_0; inputs pointing along -e_0 fire it
    # negatively, which only AbsTopK can represent
    abstopk_sae = AbsTopKSAE(build_abstopk_sae_cfg(k=1, d_in=2, d_sae=1))
    topk_sae = TopKSAE(build_topk_sae_cfg(k=1, d_in=2, d_sae=1))
    for sae in (abstopk_sae, topk_sae):
        random_params(sae)
        with torch.no_grad():
            sae.W_enc.data = torch.tensor([[1.0], [0.0]])
            sae.W_dec.data = torch.tensor([[1.0, 0.0]])
            sae.b_enc.data = torch.zeros_like(sae.b_enc.data)
            sae.b_dec.data = torch.zeros_like(sae.b_dec.data)

    x = torch.tensor([[-3.0, 0.0]])
    assert_close(abstopk_sae(x), x)
    assert_close(topk_sae(x), torch.zeros_like(x))


def test_abstopk_architecture_is_registered():
    assert get_sae_class("abstopk") == (AbsTopKSAE, AbsTopKSAEConfig)
    assert get_sae_training_class("abstopk") == (
        AbsTopKTrainingSAE,
        AbsTopKTrainingSAEConfig,
    )
    cfg = SAEConfig.from_dict(
        {"architecture": "abstopk", "d_in": 8, "d_sae": 16, "k": 3, "device": "cpu"}
    )
    assert isinstance(cfg, AbsTopKSAEConfig)
    assert cfg.k == 3


def test_AbsTopKSAE_save_and_load_from_pretrained(tmp_path: Path) -> None:
    cfg = build_abstopk_sae_cfg(k=4, d_in=8, d_sae=16)
    sae = AbsTopKSAE(cfg)
    random_params(sae)

    sae.save_model(str(tmp_path))
    sae_loaded = SAE.load_from_pretrained(str(tmp_path), device="cpu")
    assert isinstance(sae_loaded, AbsTopKSAE)

    sae_in = torch.randn(10, cfg.d_in)
    assert_close(sae_loaded(sae_in), sae(sae_in))


@pytest.mark.parametrize("rescale_acts_by_decoder_norm", [True, False])
def test_AbsTopKTrainingSAE_save_and_load_inference_sae(
    rescale_acts_by_decoder_norm: bool,
    tmp_path: Path,
) -> None:
    cfg = build_abstopk_sae_training_cfg(
        d_in=8,
        d_sae=16,
        k=4,
        rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
    )
    training_sae = AbsTopKTrainingSAE(cfg)
    random_params(training_sae)

    training_sae.save_inference_model(str(tmp_path))
    inference_sae = SAE.load_from_disk(str(tmp_path), device="cpu")
    assert isinstance(inference_sae, AbsTopKSAE)
    assert inference_sae.cfg.k == cfg.k
    if rescale_acts_by_decoder_norm:
        assert_close(
            inference_sae.W_dec.norm(dim=-1),
            torch.ones_like(inference_sae.b_enc),
        )

    sae_in = torch.randn(10, cfg.d_in)
    training_acts, _ = training_sae.encode_with_hidden_pre(sae_in)
    assert_close(training_acts, inference_sae.encode(sae_in), rtol=1e-4, atol=1e-4)
    assert_close(
        training_sae.decode(training_acts),
        inference_sae.decode(inference_sae.encode(sae_in)),
        rtol=1e-4,
        atol=1e-4,
    )


def test_calculate_abstopk_aux_acts_selects_dead_neurons_by_magnitude():
    hidden_pre = torch.tensor([[5.0, -6.0, 0.5, -0.1]])
    dead_neuron_mask = torch.ones(4, dtype=torch.bool)
    assert_close(
        calculate_abstopk_aux_acts(
            k_aux=2, hidden_pre=hidden_pre, dead_neuron_mask=dead_neuron_mask
        ),
        torch.tensor([[5.0, -6.0, 0.0, 0.0]]),
    )
    # TopK's aux selection instead keeps the largest values (5 and 0.5)
    assert_close(
        calculate_topk_aux_acts(
            k_aux=2, hidden_pre=hidden_pre, dead_neuron_mask=dead_neuron_mask
        ),
        torch.tensor([[5.0, 0.0, 0.5, 0.0]]),
    )


def test_AbsTopKTrainingSAE_aux_loss_selects_dead_latents_by_magnitude():
    d_in = 4
    d_sae = 8
    sae = AbsTopKTrainingSAE(
        build_abstopk_sae_training_cfg(
            d_in=d_in, d_sae=d_sae, k=4, rescale_acts_by_decoder_norm=False
        )
    )
    random_params(sae)
    topk_sae = TopKTrainingSAE(
        build_topk_sae_training_cfg(
            d_in=d_in, d_sae=d_sae, k=4, rescale_acts_by_decoder_norm=False
        )
    )
    topk_sae.load_state_dict(sae.state_dict())

    sae_in = torch.randn(32, d_in)
    # all-negative pre-activations: magnitude selection picks large negative
    # latents while TopK selection picks values near zero, so the losses differ
    hidden_pre = -torch.rand(32, d_sae) * 10.0
    dead_neuron_mask = torch.ones(d_sae, dtype=torch.bool)

    abstopk_aux_loss = sae.calculate_topk_aux_loss(
        sae_in, torch.zeros_like(sae_in), hidden_pre, dead_neuron_mask
    )
    topk_aux_loss = topk_sae.calculate_topk_aux_loss(
        sae_in, torch.zeros_like(sae_in), hidden_pre, dead_neuron_mask
    )
    assert_not_close(abstopk_aux_loss, topk_aux_loss)


def test_AbsTopKTrainingSAE_training_forward_pass_keeps_signs_and_reports_losses():
    d_in = 8
    d_sae = 64
    sae = AbsTopKTrainingSAE(
        build_abstopk_sae_training_cfg(
            d_in=d_in, d_sae=d_sae, k=16, rescale_acts_by_decoder_norm=False
        )
    )
    random_params(sae)
    with torch.no_grad():
        sae.b_enc.data = torch.zeros_like(sae.b_enc.data)

    sae_in = torch.randn(128, d_in)
    output = sae.training_forward_pass(
        TrainStepInput(
            sae_in=sae_in,
            coefficients={},
            dead_neuron_mask=torch.ones(d_sae, dtype=torch.bool),
            n_training_steps=0,
            is_logging_step=False,
        )
    )

    acts = output.feature_acts
    assert torch.equal((acts != 0).sum(dim=-1), torch.full((128,), 16))
    # symmetric pre-activations make all-positive selections vanishingly unlikely
    assert (acts < 0).any()
    # kept activations are the raw pre-activations, not rectified values
    assert_close(acts[acts != 0], output.hidden_pre[acts != 0])
    assert "mse_loss" in output.losses
    assert "auxiliary_reconstruction_loss" in output.losses
    assert output.losses["auxiliary_reconstruction_loss"].item() > 0

    # gradients reach the encoder through the magnitude-selected activations
    output.loss.backward()
    assert sae.W_enc.grad is not None
    assert sae.W_enc.grad.abs().sum().item() > 0
