from typing import Any

import pytest
import torch

from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
from sae_lens.saes.chunk_batchtopk_sae import (
    ChunkBatchTopKTrainingSAE,
    ChunkBatchTopKTrainingSAEConfig,
    mean_pool_chunks,
)
from sae_lens.saes.sae import TrainStepInput
from tests.helpers import assert_close, random_params


def build_chunk_sae_training_cfg(
    chunk_variant: str = "mean", chunk_size: int = 2, **kwargs: Any
) -> ChunkBatchTopKTrainingSAEConfig:
    defaults: dict[str, Any] = {
        "d_in": 6,
        "d_sae": 12,
        "k": 4,
        "chunk_size": chunk_size,
        "chunk_variant": chunk_variant,
        "decoder_init_norm": 0.1,
    }
    return ChunkBatchTopKTrainingSAEConfig(**{**defaults, **kwargs})


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


def test_mean_pool_chunks_pools_contiguous_token_spans():
    sae_in = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
        ]
    )
    pooled = mean_pool_chunks(sae_in, chunk_size=2)
    assert_close(pooled, torch.tensor([[1.5, 15.0], [3.5, 35.0], [5.5, 55.0]]))


def test_mean_pool_chunks_drops_trailing_tokens():
    sae_in = torch.arange(9, dtype=torch.float32).reshape(9, 1)
    pooled = mean_pool_chunks(sae_in, chunk_size=4)
    # 9 tokens -> 2 full chunks, the 9th token is dropped
    assert_close(pooled, mean_pool_chunks(sae_in[:8], chunk_size=4))
    assert pooled.shape == (2, 1)


def test_mean_variant_matches_batchtopk_trained_on_pooled_chunks():
    cfg = build_chunk_sae_training_cfg(chunk_variant="mean", chunk_size=3)
    chunk_sae = ChunkBatchTopKTrainingSAE(cfg)
    random_params(chunk_sae)

    batchtopk_cfg = BatchTopKTrainingSAEConfig(
        d_in=cfg.d_in,
        d_sae=cfg.d_sae,
        k=cfg.k,
        decoder_init_norm=cfg.decoder_init_norm,
    )
    batchtopk_sae = BatchTopKTrainingSAE(batchtopk_cfg)
    batchtopk_sae.load_state_dict(chunk_sae.state_dict())

    sae_in = torch.randn(36, cfg.d_in)
    with torch.no_grad():
        chunk_output = chunk_sae.training_forward_pass(build_train_step_input(sae_in))
        pooled = mean_pool_chunks(sae_in, chunk_size=3)
        batchtopk_output = batchtopk_sae.training_forward_pass(
            build_train_step_input(pooled)
        )

    assert_close(chunk_output.feature_acts, batchtopk_output.feature_acts)
    assert_close(chunk_output.sae_out, batchtopk_output.sae_out)
    assert_close(chunk_output.loss, batchtopk_output.loss)
    assert_close(chunk_output.sae_in, pooled)


def test_cross_variant_reconstructs_neighbor_chunk():
    cfg = build_chunk_sae_training_cfg(chunk_variant="cross", chunk_size=2)
    sae = ChunkBatchTopKTrainingSAE(cfg)
    random_params(sae)

    sae_in = torch.randn(16, cfg.d_in)
    with torch.no_grad():
        output = sae.training_forward_pass(build_train_step_input(sae_in))

    pooled = mean_pool_chunks(sae_in, chunk_size=2)
    source, target = pooled[:-1], pooled[1:]
    feature_acts, _ = sae.encode_with_hidden_pre(source)
    sae_out = sae.decode(feature_acts)

    expected_cross_loss = ((sae_out - target) ** 2).sum(dim=-1).mean()
    assert output.losses["cross_chunk_mse_loss"] == pytest.approx(
        expected_cross_loss.item()
    )
    assert "mean_chunk_mse_loss" not in output.losses
    assert_close(output.sae_in, source)
    assert_close(output.sae_out, sae_out)
    # aux loss is zero without dead neurons, so the total is just the cross loss
    assert output.loss.item() == pytest.approx(expected_cross_loss.item())


def test_joint_variant_combines_both_chunk_targets():
    cfg = build_chunk_sae_training_cfg(chunk_variant="joint", chunk_size=2)
    sae = ChunkBatchTopKTrainingSAE(cfg)
    random_params(sae)

    sae_in = torch.randn(16, cfg.d_in)
    with torch.no_grad():
        output = sae.training_forward_pass(build_train_step_input(sae_in))

    pooled = mean_pool_chunks(sae_in, chunk_size=2)
    source, target = pooled[:-1], pooled[1:]
    feature_acts, _ = sae.encode_with_hidden_pre(source)
    sae_out = sae.decode(feature_acts)

    expected_cross_loss = ((sae_out - target) ** 2).sum(dim=-1).mean()
    expected_mean_loss = ((sae_out - source) ** 2).sum(dim=-1).mean()
    assert output.losses["cross_chunk_mse_loss"] == pytest.approx(
        expected_cross_loss.item()
    )
    assert output.losses["mean_chunk_mse_loss"] == pytest.approx(
        expected_mean_loss.item()
    )
    assert output.loss.item() == pytest.approx(
        (expected_cross_loss + expected_mean_loss).item()
    )


def test_cross_variant_aux_loss_uses_neighbor_residual():
    cfg = build_chunk_sae_training_cfg(chunk_variant="cross", chunk_size=2)
    sae = ChunkBatchTopKTrainingSAE(cfg)
    random_params(sae)

    aux_step_inputs: list[torch.Tensor] = []
    original_calculate_aux_loss = sae.calculate_aux_loss

    def spy(step_input: TrainStepInput, **kwargs: Any) -> dict[str, torch.Tensor]:
        aux_step_inputs.append(step_input.sae_in)
        return original_calculate_aux_loss(step_input, **kwargs)

    sae.calculate_aux_loss = spy  # type: ignore[method-assign]

    sae_in = torch.randn(12, cfg.d_in)
    dead_neuron_mask = torch.zeros(cfg.d_sae, dtype=torch.bool)
    dead_neuron_mask[0] = True
    with torch.no_grad():
        sae.training_forward_pass(
            build_train_step_input(sae_in, dead_neuron_mask=dead_neuron_mask)
        )

    pooled = mean_pool_chunks(sae_in, chunk_size=2)
    assert len(aux_step_inputs) == 1
    assert_close(aux_step_inputs[0], pooled[1:])


def test_training_step_updates_threshold_and_metrics():
    cfg = build_chunk_sae_training_cfg(chunk_variant="mean", chunk_size=4)
    sae = ChunkBatchTopKTrainingSAE(cfg)
    sae.b_enc.data = torch.randn(cfg.d_sae) + 10

    with torch.no_grad():
        output = sae.training_forward_pass(
            build_train_step_input(torch.randn(16, cfg.d_in))
        )

    assert sae.topk_threshold.item() > 0.0
    assert output.metrics["topk_threshold"] == sae.topk_threshold


def test_chunk_sae_exports_jumprelu_inference_config():
    cfg = build_chunk_sae_training_cfg(chunk_variant="joint", chunk_size=2)
    sae = ChunkBatchTopKTrainingSAE(cfg)

    inference_config = sae.cfg.get_inference_sae_cfg_dict()

    assert inference_config["architecture"] == "jumprelu"
    assert "chunk_size" not in inference_config
    assert "chunk_variant" not in inference_config


def test_chunk_config_validates_chunk_size_and_variant():
    with pytest.raises(ValueError, match="chunk_size"):
        ChunkBatchTopKTrainingSAE(build_chunk_sae_training_cfg(chunk_size=0))
    with pytest.raises(ValueError, match="chunk_variant"):
        ChunkBatchTopKTrainingSAE(build_chunk_sae_training_cfg(chunk_variant="bogus"))


def test_cross_variant_requires_two_chunks_per_batch():
    cfg = build_chunk_sae_training_cfg(chunk_variant="cross", chunk_size=8)
    sae = ChunkBatchTopKTrainingSAE(cfg)
    # 8 tokens with chunk_size=8 pools to a single chunk, too few for a neighbor
    with pytest.raises(ValueError, match="at least 2 pooled chunks"):
        sae.training_forward_pass(build_train_step_input(torch.randn(8, cfg.d_in)))
