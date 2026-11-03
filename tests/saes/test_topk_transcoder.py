from pathlib import Path
from typing import Any

import pytest
import torch

from sae_lens import SAE
from sae_lens.registry import get_sae_class
from sae_lens.saes.topk_transcoder import (
    TopKSkipTranscoder,
    TopKSkipTranscoderConfig,
    TopKTranscoder,
    TopKTranscoderConfig,
)
from sae_lens.saes.transcoder import SkipTranscoder, SkipTranscoderConfig
from tests.helpers import assert_close, random_params

DEFAULT_CFG: dict[str, Any] = {
    "d_in": 32,
    "d_sae": 64,
    "d_out": 48,
    "k": 8,
    "dtype": "float32",
    "device": "cpu",
    "normalize_activations": "none",
    "apply_b_dec_to_input": False,
}


def build_topk_transcoder_cfg(**kwargs: Any) -> TopKTranscoderConfig:
    return TopKTranscoderConfig(**{**DEFAULT_CFG, **kwargs})


def build_topk_skip_transcoder_cfg(**kwargs: Any) -> TopKSkipTranscoderConfig:
    return TopKSkipTranscoderConfig(**{**DEFAULT_CFG, **kwargs})


def build_transcoder_with_identity_encoder(
    cfg: TopKTranscoderConfig,
) -> TopKTranscoder:
    """Encoder maps x into the first d_in latents, leaving the rest at zero."""
    transcoder = TopKTranscoder(cfg)
    transcoder.W_enc.data = torch.zeros_like(transcoder.W_enc)
    transcoder.W_enc.data[: cfg.d_in, : cfg.d_in] = torch.eye(cfg.d_in)
    transcoder.b_enc.data = torch.zeros_like(transcoder.b_enc)
    return transcoder


def test_topk_transcoder_keeps_only_k_largest_latents() -> None:
    cfg = build_topk_transcoder_cfg(d_in=8, d_sae=16, d_out=12, k=3)
    transcoder = build_transcoder_with_identity_encoder(cfg)

    x = torch.tensor([[0.9, 0.5, 0.7, 0.1, 0.3, 0.2, 0.8, 0.4]])
    _, feature_acts = transcoder.forward_with_activations(x)

    expected = torch.zeros(1, cfg.d_sae)
    expected[0, 0] = 0.9
    expected[0, 2] = 0.7
    expected[0, 6] = 0.8
    assert_close(feature_acts, expected)


def test_topk_transcoder_output_matches_manual_topk_computation() -> None:
    cfg = build_topk_transcoder_cfg()
    transcoder = TopKTranscoder(cfg)
    random_params(transcoder)

    x = torch.randn(20, cfg.d_in)
    sae_out, feature_acts = transcoder.forward_with_activations(x)

    hidden_pre = x @ transcoder.W_enc + transcoder.b_enc
    topk = hidden_pre.topk(cfg.k, dim=-1)
    expected_acts = torch.zeros_like(hidden_pre)
    expected_acts.scatter_(-1, topk.indices, topk.values.relu())
    assert_close(feature_acts, expected_acts)
    assert_close(sae_out, expected_acts @ transcoder.W_dec + transcoder.b_dec)


def test_topk_transcoder_never_activates_more_than_k_latents() -> None:
    cfg = build_topk_transcoder_cfg()
    transcoder = TopKTranscoder(cfg)
    random_params(transcoder)

    feature_acts = transcoder.encode(torch.randn(512, cfg.d_in))

    assert feature_acts.count_nonzero(dim=-1).max().item() <= cfg.k


def test_topk_transcoder_gradients_reach_only_active_latents() -> None:
    cfg = build_topk_transcoder_cfg()
    transcoder = TopKTranscoder(cfg)
    random_params(transcoder)

    _, feature_acts = transcoder.forward_with_activations(torch.randn(8, cfg.d_in))
    feature_acts.sum().backward()

    latent_was_active = (feature_acts != 0).any(dim=0)
    w_enc_grad = transcoder.W_enc.grad
    assert w_enc_grad is not None

    assert torch.all(w_enc_grad[:, ~latent_was_active] == 0)
    assert torch.any(w_enc_grad[:, latent_was_active] != 0)


def test_topk_skip_transcoder_adds_skip_path_on_top_of_topk_latents() -> None:
    cfg = build_topk_skip_transcoder_cfg()
    transcoder = TopKSkipTranscoder(cfg)
    random_params(transcoder)

    x = torch.randn(16, cfg.d_in)
    output, feature_acts = transcoder.forward_with_activations(x)
    skip_contribution = x @ transcoder.W_skip.T

    transcoder.W_skip.data.zero_()
    out_no_skip, acts_no_skip = transcoder.forward_with_activations(x)

    assert_close(feature_acts, acts_no_skip)
    assert_close(output, out_no_skip + skip_contribution)
    assert feature_acts.count_nonzero(dim=-1).max().item() <= cfg.k


def test_topk_skip_transcoder_with_k_eq_d_sae_matches_relu_skip_transcoder() -> None:
    topk_cfg = build_topk_skip_transcoder_cfg(k=64)
    topk_transcoder = TopKSkipTranscoder(topk_cfg)
    relu_transcoder = SkipTranscoder(
        SkipTranscoderConfig(
            d_in=topk_cfg.d_in,
            d_sae=topk_cfg.d_sae,
            d_out=topk_cfg.d_out,
            apply_b_dec_to_input=False,
        )
    )
    relu_transcoder.load_state_dict(topk_transcoder.state_dict())

    x = torch.randn(32, topk_cfg.d_in)

    topk_out, topk_acts = topk_transcoder.forward_with_activations(x)
    relu_out, relu_acts = relu_transcoder.forward_with_activations(x)

    assert_close(topk_acts, relu_acts)
    assert_close(topk_out, relu_out)


def test_topk_transcoder_architectures_are_registered() -> None:
    sae_class, cfg_class = get_sae_class("topk_transcoder")
    assert sae_class is TopKTranscoder
    assert cfg_class is TopKTranscoderConfig

    sae_class, cfg_class = get_sae_class("topk_skip_transcoder")
    assert sae_class is TopKSkipTranscoder
    assert cfg_class is TopKSkipTranscoderConfig


def test_topk_skip_transcoder_round_trips_through_disk_loader(tmp_path: Path) -> None:
    cfg = build_topk_skip_transcoder_cfg()
    transcoder = TopKSkipTranscoder(cfg)
    random_params(transcoder)
    transcoder.save_model(tmp_path)

    loaded = SAE.load_from_disk(str(tmp_path))

    assert isinstance(loaded, TopKSkipTranscoder)
    assert loaded.cfg.k == cfg.k
    assert loaded.cfg.d_out == cfg.d_out
    x = torch.randn(6, cfg.d_in)
    assert_close(loaded.forward_with_activations(x)[0], transcoder(x))


def test_topk_transcoder_config_from_dict_round_trips() -> None:
    cfg = build_topk_skip_transcoder_cfg(k=11)
    cfg_dict = cfg.to_dict()

    assert cfg_dict["k"] == 11

    restored = TopKSkipTranscoderConfig.from_dict(cfg_dict)
    assert restored.k == 11
    assert restored.d_out == cfg.d_out
    assert restored.architecture() == "topk_skip_transcoder"


def test_topk_transcoder_config_rejects_k_out_of_range() -> None:
    with pytest.raises(ValueError, match="0 < k <= d_sae"):
        build_topk_transcoder_cfg(k=0)
    with pytest.raises(ValueError, match="0 < k <= d_sae"):
        build_topk_transcoder_cfg(k=65)


def test_topk_transcoder_fold_w_dec_norm_is_unsafe_and_raises() -> None:
    transcoder = TopKTranscoder(build_topk_transcoder_cfg())
    random_params(transcoder)

    with pytest.raises(NotImplementedError, match="not safe for TopK transcoders"):
        transcoder.fold_W_dec_norm()
