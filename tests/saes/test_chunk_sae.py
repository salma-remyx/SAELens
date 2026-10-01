from pathlib import Path
from typing import Any

import pytest
import torch

from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.chunk_sae import (
    ChunkTrainingSAE,
    ChunkTrainingSAEConfig,
    chunk_means,
)
from sae_lens.saes.jumprelu_sae import JumpReLUSAE
from sae_lens.saes.sae import SAE, TrainStepInput
from tests.helpers import assert_close, random_params


def build_chunk_sae_training_cfg(**kwargs: Any) -> ChunkTrainingSAEConfig:
    defaults: dict[str, Any] = {
        "d_in": 4,
        "d_sae": 16,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
        "decoder_init_norm": 0.1,
        "apply_b_dec_to_input": False,
        "k": 4,
        "chunk_size": 2,
    }
    return ChunkTrainingSAEConfig(**{**defaults, **kwargs})


def build_batchtopk_sae_training_cfg(**kwargs: Any) -> BatchTopKTrainingSAEConfig:
    defaults: dict[str, Any] = {
        "d_in": 4,
        "d_sae": 16,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
        "decoder_init_norm": 0.1,
        "apply_b_dec_to_input": False,
        "k": 4,
    }
    return BatchTopKTrainingSAEConfig(**{**defaults, **kwargs})


def train_step_input(sae_in: torch.Tensor) -> TrainStepInput:
    return TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=False,
    )


def centered_per_chunk(noise: torch.Tensor, per_chunk: int) -> torch.Tensor:
    grouped = noise.reshape(noise.shape[0], -1, per_chunk, noise.shape[-1])
    return (grouped - grouped.mean(dim=2, keepdim=True)).reshape(noise.shape)


def test_chunk_means_pools_contiguous_spans() -> None:
    acts = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
    # (1 span, 4 tokens, 2 dims) with chunk_size 2 -> means of token pairs
    assert_close(chunk_means(acts, 2), torch.tensor([[[2.0, 3.0], [6.0, 7.0]]]))

    # a chunk spanning all tokens is the sequence mean, per leading batch row
    batched = torch.arange(3 * 8 * 2, dtype=torch.float32).reshape(3, 8, 2)
    assert_close(chunk_means(batched, 8), batched.mean(dim=1, keepdim=True))


def test_chunk_means_rejects_inputs_without_complete_chunks() -> None:
    acts = torch.randn(2, 4, 4)
    with pytest.raises(ValueError, match="at least 1"):
        chunk_means(acts, 0)
    with pytest.raises(ValueError, match="token dimension"):
        chunk_means(torch.randn(4, 4), 2)
    with pytest.raises(ValueError, match="divisible"):
        chunk_means(acts, 3)


def test_chunk_sae_loss_depends_only_on_chunk_means() -> None:
    # two span sets with identical per-chunk means but different token content
    target_means = torch.randn(6, 2, 4)
    spans_a = target_means.repeat_interleave(2, dim=1) + centered_per_chunk(
        torch.randn(6, 4, 4), 2
    )
    spans_b = target_means.repeat_interleave(2, dim=1) + centered_per_chunk(
        torch.randn(6, 4, 4), 2
    )
    assert_close(chunk_means(spans_a, 2), chunk_means(spans_b, 2), atol=1e-6)

    sae_a = ChunkTrainingSAE(build_chunk_sae_training_cfg())
    sae_b = ChunkTrainingSAE(build_chunk_sae_training_cfg())
    random_params(sae_a)
    sae_b.load_state_dict(sae_a.state_dict())

    with torch.no_grad():
        out_a = sae_a.training_forward_pass(train_step_input(spans_a))
        out_b = sae_b.training_forward_pass(train_step_input(spans_b))
        control = sae_a.training_forward_pass(train_step_input(torch.randn(6, 4, 4)))

    assert_close(out_a.sae_in, chunk_means(spans_a, 2), atol=1e-6)
    assert_close(out_a.feature_acts, out_b.feature_acts, atol=1e-4)
    assert_close(out_a.loss, out_b.loss, atol=1e-4)
    # the control input has different chunk means, so its loss must differ
    # by far more than the gap between the two same-mean inputs
    same_means_gap = (out_a.loss - out_b.loss).abs().item()
    different_means_gap = (out_a.loss - control.loss).abs().item()
    assert same_means_gap * 100 < different_means_gap


def test_chunk_sae_with_chunk_size_1_matches_token_level_batchtopk_sae() -> None:
    chunk_sae = ChunkTrainingSAE(build_chunk_sae_training_cfg(chunk_size=1))
    token_sae = BatchTopKTrainingSAE(build_batchtopk_sae_training_cfg())
    random_params(token_sae)
    chunk_sae.load_state_dict(token_sae.state_dict())

    tokens = torch.randn(32, 4)
    spans = tokens.unsqueeze(1)  # one token per chunk

    with torch.no_grad():
        chunk_out = chunk_sae.training_forward_pass(train_step_input(spans))
        token_out = token_sae.training_forward_pass(train_step_input(tokens))

    assert_close(chunk_out.loss, token_out.loss)
    assert_close(chunk_out.feature_acts.flatten(0, -2), token_out.feature_acts)
    assert_close(chunk_out.sae_out.flatten(0, -2), token_out.sae_out)


def test_chunk_sae_encode_and_forward_operate_on_chunk_means() -> None:
    sae = ChunkTrainingSAE(build_chunk_sae_training_cfg())
    random_params(sae)
    token_sae = BatchTopKTrainingSAE(build_batchtopk_sae_training_cfg())
    token_sae.load_state_dict(sae.state_dict())

    spans = torch.randn(5, 6, 4)  # 5 sequences of 6 tokens, chunk_size 2
    means = chunk_means(spans, 2)

    with torch.no_grad():
        assert sae.encode(spans).shape == (5, 3, 16)
        assert sae(spans).shape == (5, 3, 4)
        assert_close(
            sae.encode(spans).flatten(0, -2), token_sae.encode(means.flatten(0, -2))
        )
        assert_close(sae(spans).flatten(0, -2), token_sae(means.flatten(0, -2)))


def test_chunk_sae_saves_as_jumprelu_over_chunk_means(tmp_path: Path) -> None:
    sae = ChunkTrainingSAE(build_chunk_sae_training_cfg())
    random_params(sae)

    spans = torch.randn(16, 2, 4)
    with torch.no_grad():
        # let the batchtopk threshold EMA converge before saving as jumprelu
        for _ in range(500):
            sae.training_forward_pass(train_step_input(spans))

    model_path = str(tmp_path)
    sae.save_inference_model(model_path)
    inference_sae = SAE.load_from_disk(model_path, device="cpu")

    assert isinstance(inference_sae, JumpReLUSAE)
    means = chunk_means(spans, 2).flatten(0, -2)
    with torch.no_grad():
        assert_close(inference_sae.encode(means), sae.encode(spans).flatten(0, -2))
        assert_close(inference_sae(means), sae(spans).flatten(0, -2))
