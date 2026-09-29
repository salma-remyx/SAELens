"""Typical SAE analysis workflow: contrastive feature selection and steering.

Runs the end-to-end workflow of collecting SAE feature activations over groups
of parallel texts, selecting the features that distinguish a target group, and
steering the model with the decoded feature directions, following
"Strengthening Target-Language Features: SAE-Based Steering for Multilingual
Inference" (arXiv:2608.04904).
"""

import pytest
import torch
from transformer_lens.hook_points import HookPoint

from sae_lens.analysis.feature_steering import (
    ContrastiveFeatures,
    select_contrastive_features,
    sentence_mean_activations,
    steering_hook,
    steering_vector_from_features,
)
from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from sae_lens.saes.sae import SAEMetadata
from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig
from tests.helpers import (
    TINYSTORIES_MODEL,
    assert_close,
    assert_not_close,
)

MODEL = TINYSTORIES_MODEL
HOOK_NAME = "blocks.0.hook_resid_pre"

# Parallel sentence groups. tiny-stories is an English-only model, so the
# non-English groups are just token distributions the SAE responds differently
# to; the workflow under test is the selection and steering mechanism, not
# real language transfer.
PARALLEL_TEXTS = {
    "english": [
        "Once upon a time there was a little girl.",
        "She liked to play with her friends in the park.",
        "One sunny day the little boy found a big red ball.",
    ],
    "french": [
        "Il était une fois une petite fille.",
        "Elle aimait jouer avec ses amis dans le parc.",
        "Un beau jour le petit garçon a trouvé un grand ballon rouge.",
    ],
    "german": [
        "Es war einmal ein kleines Mädchen.",
        "Sie spielte gern mit ihren Freunden im Park.",
        "An einem sonnigen Tag fand der kleine Junge einen großen roten Ball.",
    ],
}


@pytest.fixture(scope="module")
def model():
    model = HookedSAETransformer.from_pretrained(MODEL, device="cpu")
    yield model
    model.reset_saes()  # type: ignore


@pytest.fixture(scope="module")
def sae(model: HookedSAETransformer) -> StandardSAE:
    # keep the constructor's init: its zero b_dec is what apply_b_dec_to_input
    # expects, and random_params would fill b_dec with positive values that
    # swamp the input and zero out every feature activation.
    return StandardSAE(
        StandardSAEConfig(
            d_in=model.cfg.d_model,
            d_sae=model.cfg.d_model * 2,
            dtype="float32",
            device="cpu",
            metadata=SAEMetadata(
                model_name=MODEL,
                hook_name=HOOK_NAME,
                hook_head_index=None,
                prepend_bos=True,
            ),
        )
    )


@pytest.fixture(scope="module")
def activations(model: HookedSAETransformer, sae: StandardSAE):
    return sentence_mean_activations(model, sae, PARALLEL_TEXTS)


def test_select_contrastive_features_picks_largest_absolute_contrasts():
    activations_by_group = {
        "target": torch.tensor([[4.0, 0.1, 2.0, 3.0], [4.2, 0.0, 2.1, 2.8]]),
        "left": torch.tensor([[0.2, 5.0, 2.0, 1.0], [0.1, 4.8, 1.9, 1.2]]),
        "right": torch.tensor([[0.3, 5.2, 2.0, 0.9], [0.0, 5.1, 2.2, 1.1]]),
    }
    # group means: target (4.1, 0.05, 2.05, 2.9), left (0.15, 4.9, 2.025, 1.05),
    # right (0.15, 5.15, 2.025, 1.05); reference mean (0.15, 5.025, 2.025, 1.05)
    # contrast (3.95, -4.975, 0.025, 1.85), so the two largest absolute
    # contrasts are feature 1 (suppressed for the target) and feature 0.
    features = select_contrastive_features(
        activations_by_group, "target", num_features=2
    )
    assert torch.equal(features.feature_indices, torch.tensor([1, 0]))
    assert_close(features.contrasts, torch.tensor([-4.975, 3.95]))


def test_sentence_mean_activations_match_directly_encoded_features(
    model: HookedSAETransformer, sae: StandardSAE
):
    texts = {
        "english": PARALLEL_TEXTS["english"][:2],
        "french": PARALLEL_TEXTS["french"][:2],
    }
    sentence_means = sentence_mean_activations(model, sae, texts)
    for group, group_texts in texts.items():
        assert sentence_means[group].shape == (len(group_texts), sae.cfg.d_sae)
        for i, text in enumerate(group_texts):
            _, cache = model.run_with_cache(text)
            expected = sae.encode(cache[HOOK_NAME]).mean(dim=1)[0]
            assert_close(sentence_means[group][i], expected)


def test_selected_features_rank_contrasts(activations: dict[str, torch.Tensor]):
    features = select_contrastive_features(activations, "french", num_features=3)
    contrast = activations["french"].mean(dim=0) - torch.stack(
        [activations["english"].mean(dim=0), activations["german"].mean(dim=0)]
    ).mean(dim=0)
    selected_contrasts = contrast.abs()[features.feature_indices]
    unselected = contrast.abs()
    unselected[features.feature_indices] = 0.0
    assert selected_contrasts.min() >= unselected.max()
    assert_close(features.contrasts, contrast[features.feature_indices])


def test_steering_vector_decodes_selected_contrasts(sae: StandardSAE):
    features = ContrastiveFeatures(
        feature_indices=torch.tensor([1, 0]),
        contrasts=torch.tensor([-4.0, 3.0]),
    )
    steering_vector = steering_vector_from_features(sae, features)
    assert steering_vector.hook_name == HOOK_NAME
    assert_close(
        steering_vector.vector,
        torch.tensor([-4.0, 3.0]) @ sae.W_dec[torch.tensor([1, 0])],
    )


def test_steering_hook_adds_vector_at_final_position_only(
    model: HookedSAETransformer,
    sae: StandardSAE,
    activations: dict[str, torch.Tensor],
):
    steering_vector = steering_vector_from_features(
        sae, select_contrastive_features(activations, "french")
    )
    captured: dict[str, torch.Tensor] = {}

    def capture(hidden_states: torch.Tensor, hook: HookPoint) -> None:  # noqa: ARG001
        captured["resid"] = hidden_states.detach().clone()

    prompt = "Once upon a time"
    steered_logits = model.run_with_hooks(
        prompt,
        fwd_hooks=[
            (HOOK_NAME, steering_hook(steering_vector, 5.0)),
            (HOOK_NAME, capture),
        ],
    )
    _, plain_cache = model.run_with_cache(prompt)
    expected_resid = plain_cache[HOOK_NAME].clone()
    expected_resid[:, -1, :] += 5.0 * steering_vector.vector
    assert_close(captured["resid"], expected_resid)
    assert_not_close(steered_logits, model(prompt))


def test_run_with_hooks_with_saes_applies_steering(
    model: HookedSAETransformer,
    sae: StandardSAE,
    activations: dict[str, torch.Tensor],
):
    steering_vector = steering_vector_from_features(
        sae, select_contrastive_features(activations, "french")
    )
    sae_output_hook = HOOK_NAME + ".hook_sae_output"
    prompt = "Once upon a time"
    baseline = model.run_with_saes(prompt, saes=[sae])
    unchanged = model.run_with_hooks_with_saes(
        prompt,
        saes=[sae],
        fwd_hooks=[(sae_output_hook, steering_hook(steering_vector, 0.0))],
    )
    assert_close(unchanged, baseline)
    steered = model.run_with_hooks_with_saes(
        prompt,
        saes=[sae],
        fwd_hooks=[(sae_output_hook, steering_hook(steering_vector, 12.0))],
    )
    assert_not_close(steered, baseline)
