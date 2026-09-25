"""Tests for sae_lens.analysis.feature_effect_geometry."""

import math
from collections.abc import Generator

import pytest
import torch

from sae_lens.analysis.feature_effect_geometry import (
    analyze_feature_effect,
    effect_geometry,
    feature_effect_vectors,
)
from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from sae_lens.saes.sae import SAEMetadata
from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig
from tests.helpers import TINYSTORIES_MODEL, assert_close, random_params

MODEL = TINYSTORIES_MODEL
PROMPTS = [
    "Once upon a time there was a little girl",
    "One day, a big dragon flew over the village",
    "The little boy found a shiny coin in the garden",
    "Sarah and Tom went to the park to play",
    "The cat sat on the warm mat by the fire",
    "In a small house lived a friendly mouse",
    "Every morning, the sun rose over the hills",
    "The children laughed and ran across the field",
    "A kind wizard lived at the edge of the forest",
    "The puppy chased its tail around the yard",
]
ACTIVE_FEATURE = 0


def make_sae(act_name: str, d_in: int) -> StandardSAE:
    sae = StandardSAE(
        StandardSAEConfig(
            d_in=d_in,
            d_sae=d_in * 2,
            dtype="float32",
            device="cpu",
            metadata=SAEMetadata(
                model_name=MODEL,
                hook_name=act_name,
                hook_head_index=None,
                prepend_bos=True,
            ),
        )
    )
    random_params(sae)
    # Give one feature a constant unit activation so it fires in every
    # context, making the tests below independent of the random weights.
    with torch.no_grad():
        sae.W_enc[:, ACTIVE_FEATURE] = 0.0
        sae.b_enc[ACTIVE_FEATURE] = 1.0
    return sae


def last_position_activations(
    model: HookedSAETransformer, sae: StandardSAE, act_name: str
) -> torch.Tensor:
    activations = []
    with torch.no_grad():
        for prompt in PROMPTS:
            _, cache = model.run_with_cache(model.to_tokens(prompt))
            activations.append(sae.encode(cache[act_name])[0, -1])
    return torch.stack(activations)


@pytest.fixture(scope="module")
def model() -> Generator[HookedSAETransformer, None, None]:
    model = HookedSAETransformer.from_pretrained(MODEL, device="cpu")
    yield model
    model.reset_saes()
    model.remove_all_hook_fns(including_permanent=True)


def test_effect_geometry_identical_directions_form_a_perfect_ray() -> None:
    scales = torch.tensor([3.0, 2.0, 1.0, 5.0, 0.5, 4.0, 2.0, 1.0])
    effects = scales.unsqueeze(-1) * torch.tensor([[1.0, 0.0, 0.0]])

    geometry = effect_geometry(effects)

    assert geometry.num_contexts == 8
    assert geometry.sufficient_contexts is True
    assert geometry.ray_consistency == pytest.approx(1.0)
    assert geometry.axis_eigenvalue_fraction == pytest.approx(1.0)
    assert geometry.top2_span_variance == pytest.approx(1.0)
    assert geometry.participation_ratio == pytest.approx(1.0)
    assert geometry.entropy_rank == pytest.approx(1.0)
    assert geometry.centered_residual_energy == pytest.approx(0.0)
    assert geometry.mean_effect_norm == pytest.approx(scales.mean().item())
    assert geometry.axis_split_fraction == pytest.approx(0.0)


def test_effect_geometry_antipodal_directions_form_an_axis_not_a_ray() -> None:
    up = torch.tensor([0.6, 0.8, 0.0])
    scales = torch.tensor([3.0, 2.0, 1.0, 5.0, -1.0, -2.0, -4.0, -0.5])
    effects = scales.unsqueeze(-1) * up

    geometry = effect_geometry(effects)

    assert geometry.num_contexts == 8
    assert geometry.ray_consistency == pytest.approx(-1.0 / 7.0)
    assert geometry.axis_eigenvalue_fraction == pytest.approx(1.0)
    assert geometry.axis_split_fraction == pytest.approx(0.5)
    assert geometry.participation_ratio == pytest.approx(1.0)
    assert geometry.centered_residual_energy == pytest.approx(1.0)


def test_effect_geometry_orthogonal_directions_fill_the_span() -> None:
    effects = torch.eye(16) * torch.arange(1.0, 17.0).unsqueeze(-1)

    geometry = effect_geometry(effects)

    assert geometry.num_contexts == 16
    assert geometry.ray_consistency == pytest.approx(0.0)
    assert geometry.axis_eigenvalue_fraction == pytest.approx(1.0 / 16.0)
    assert geometry.top2_span_variance == pytest.approx(2.0 / 16.0)
    assert geometry.participation_ratio == pytest.approx(16.0)
    assert geometry.entropy_rank == pytest.approx(16.0)
    assert geometry.centered_residual_energy == pytest.approx(1.0 - 1.0 / 16.0)


def test_effect_geometry_evenly_spread_plane_is_two_dimensional() -> None:
    # float64 construction: float32 trig error would leak into the spectrum.
    angles = torch.arange(64, dtype=torch.float64) * (2 * math.pi / 64)
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    effects = torch.arange(1.0, 65.0, dtype=torch.float64).unsqueeze(-1) * directions

    geometry = effect_geometry(effects)

    assert geometry.num_contexts == 64
    assert geometry.ray_consistency == pytest.approx(-1.0 / 63.0)
    assert geometry.top2_span_variance == pytest.approx(1.0)
    assert geometry.participation_ratio == pytest.approx(2.0)
    assert geometry.entropy_rank == pytest.approx(2.0)
    assert geometry.centered_residual_energy == pytest.approx(1.0)


def test_effect_geometry_flags_insufficient_and_empty_contexts() -> None:
    insufficient = effect_geometry(torch.eye(4))
    assert insufficient.num_contexts == 4
    assert insufficient.sufficient_contexts is False

    empty = effect_geometry(torch.empty((0, 10)))
    assert empty.num_contexts == 0
    assert empty.sufficient_contexts is False
    assert empty.ray_consistency == pytest.approx(0.0)
    assert empty.participation_ratio == pytest.approx(0.0)


def test_feature_effect_vectors_match_analytic_logit_lens_effects(
    model: HookedSAETransformer,
) -> None:
    act_name = "ln_final.hook_normalized"
    sae = make_sae(act_name, model.cfg.d_model)
    prompts = PROMPTS[:5]

    effects = feature_effect_vectors(model, sae, prompts, ACTIVE_FEATURE)

    activations = last_position_activations(model, sae, act_name)[: len(prompts)]
    assert_close(
        activations[:, ACTIVE_FEATURE], torch.ones(len(prompts)), msg="fixture setup"
    )
    assert effects.shape == (len(prompts), model.cfg.d_vocab)
    decoder_direction = sae.W_dec[ACTIVE_FEATURE] @ model.W_U
    assert_close(
        effects,
        -activations[:, ACTIVE_FEATURE].unsqueeze(-1) * decoder_direction,
        atol=1e-4,
        rtol=1e-4,
        msg="two float32 forward passes carry ~1e-5 logit noise",
    )


def test_feature_effect_vectors_are_zero_for_a_dead_decoder_feature(
    model: HookedSAETransformer,
) -> None:
    sae = make_sae("blocks.0.hook_mlp_out", model.cfg.d_model)
    with torch.no_grad():
        sae.W_dec[ACTIVE_FEATURE] = 0.0

    effects = feature_effect_vectors(model, sae, PROMPTS, ACTIVE_FEATURE)

    assert effects.shape == (0, model.cfg.d_vocab)


def test_feature_effect_vectors_drop_contexts_where_feature_is_inactive(
    model: HookedSAETransformer,
) -> None:
    sae = make_sae("blocks.0.hook_mlp_out", model.cfg.d_model)
    with torch.no_grad():
        sae.b_enc[ACTIVE_FEATURE] = -1e6

    effects = feature_effect_vectors(model, sae, PROMPTS, ACTIVE_FEATURE)

    assert effects.shape == (0, model.cfg.d_vocab)


def test_analyze_feature_effect_scores_an_active_feature(
    model: HookedSAETransformer,
) -> None:
    sae = make_sae("blocks.0.hook_mlp_out", model.cfg.d_model)

    geometry = analyze_feature_effect(model, sae, PROMPTS, ACTIVE_FEATURE)

    assert geometry.num_contexts == len(PROMPTS)
    assert geometry.sufficient_contexts is True
    assert -1.0 <= geometry.ray_consistency <= 1.0
    assert 1.0 <= geometry.participation_ratio <= len(PROMPTS)
    assert geometry.mean_effect_norm > 0.0


def test_analyze_feature_effect_respects_max_contexts(
    model: HookedSAETransformer,
) -> None:
    sae = make_sae("blocks.0.hook_mlp_out", model.cfg.d_model)

    geometry = analyze_feature_effect(
        model, sae, PROMPTS, ACTIVE_FEATURE, max_contexts=3
    )

    assert geometry.num_contexts == 3
