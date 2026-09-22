import pytest
import torch

from sae_lens.analysis.contrastive_projection import (
    contrastive_feature_attribution,
    contrastive_logit_lens,
    run_contrastive_projection,
)
from sae_lens.analysis.hooked_sae_transformer import HookedSAETransformer
from sae_lens.saes.sae import SAEMetadata
from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig
from tests.helpers import TINYSTORIES_MODEL, assert_close, random_params

MODEL = TINYSTORIES_MODEL
PROMPT = "The capital of France is Paris"
BASELINES = ["The capital of Germany is Berlin", "The capital of Italy is Rome"]
SHORT_PROMPT = "one two three four"
SHORT_BASELINE = "one two three five"


@pytest.fixture(scope="module")
def model():
    model = HookedSAETransformer.from_pretrained(MODEL, device="cpu")
    yield model
    model.reset_saes()  # type: ignore


@pytest.fixture(scope="module")
def sae(model: HookedSAETransformer):
    cfg = StandardSAEConfig(
        d_in=model.cfg.d_model,
        d_sae=model.cfg.d_model * 2,
        dtype="float32",
        device="cpu",
        metadata=SAEMetadata(
            model_name=MODEL,
            hook_name=f"blocks.{model.cfg.n_layers - 1}.hook_resid_post",
            prepend_bos=True,
        ),
    )
    sae = StandardSAE(cfg)
    random_params(sae)
    return sae


def test_contrastive_logit_lens_equals_differenced_raw_lenses():
    hidden_a = torch.randn(100, 32)
    hidden_b = torch.randn(100, 32)
    w_unembed = torch.randn(32, 50)
    lens_a = contrastive_logit_lens(hidden_a, w_unembed)
    lens_b = contrastive_logit_lens(hidden_b, w_unembed)
    contrast = contrastive_logit_lens(hidden_a - hidden_b, w_unembed)
    assert_close(contrast, lens_a - lens_b)
    assert_close(contrast, -contrastive_logit_lens(hidden_b - hidden_a, w_unembed))


def test_contrastive_feature_attribution_projects_contrast_on_decoder_directions():
    w_dec = torch.randn(16, 32)
    contrast = torch.randn(64, 32)
    scores = contrastive_feature_attribution(contrast, w_dec)
    assert scores.shape == (64, 16)
    assert_close(scores, contrast @ w_dec.T)

    # a contrast orthogonal to a feature's decoder direction does not load on it
    contrast = torch.randn(1, 32)
    projection = (contrast @ w_dec[0]) / (w_dec[0] @ w_dec[0])
    contrast = contrast - projection * w_dec[0]
    scores = contrastive_feature_attribution(contrast, w_dec)
    assert scores[0, 0].item() == pytest.approx(0, abs=1e-4)
    assert scores[0, 1:].abs().sum() > 0


def test_run_contrastive_projection_of_identical_prompts_is_exactly_zero(
    model: HookedSAETransformer,
):
    result = run_contrastive_projection(model, SHORT_PROMPT, baselines=SHORT_PROMPT)
    assert set(result.contrast) == {
        f"blocks.{layer}.hook_resid_post" for layer in range(model.cfg.n_layers)
    }
    for contrast in result.contrast.values():
        assert torch.all(contrast == 0)
    for token_scores in result.token_scores.values():
        assert torch.all(token_scores == 0)
    assert result.feature_scores is None
    assert result.feature_hook is None
    with pytest.raises(ValueError, match="requires an SAE"):
        result.top_features()


def test_run_contrastive_projection_matches_manual_baseline_average(
    model: HookedSAETransformer,
):
    hook = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
    _, cache_prompt = model.run_with_cache(PROMPT, remove_batch_dim=True)
    baseline_caches = [
        model.run_with_cache(baseline, remove_batch_dim=True)[1]
        for baseline in BASELINES
    ]
    expected_contrast = cache_prompt[hook] - torch.stack(
        [cache[hook] for cache in baseline_caches]
    ).mean(dim=0)

    result = run_contrastive_projection(model, PROMPT, baselines=BASELINES)

    assert_close(result.contrast[hook], expected_contrast)
    assert_close(result.token_scores[hook], expected_contrast @ model.W_U)


def test_run_contrastive_projection_with_sae_attributes_contrast_to_features(
    model: HookedSAETransformer, sae: StandardSAE
):
    hook = sae.cfg.metadata.hook_name
    assert hook is not None

    result = run_contrastive_projection(
        model, SHORT_PROMPT, baselines=SHORT_BASELINE, sae=sae
    )

    assert result.feature_hook == hook
    assert result.feature_scores is not None
    assert result.feature_scores.shape == (
        result.contrast[hook].shape[0],
        sae.cfg.d_sae,
    )
    # the SAE runs with an error term, so the traced contrast is the clean one
    _, cache_prompt = model.run_with_cache(SHORT_PROMPT, remove_batch_dim=True)
    _, cache_baseline = model.run_with_cache(SHORT_BASELINE, remove_batch_dim=True)
    clean_contrast = cache_prompt[hook] - cache_baseline[hook]
    assert_close(result.contrast[hook], clean_contrast, atol=1e-4)
    # feature scores are the clean contrast projected onto the decoder directions
    assert_close(
        result.feature_scores,
        contrastive_feature_attribution(clean_contrast, sae.W_dec),
        atol=1e-4,
    )
    # the run must not leave the SAE attached
    assert len(model._acts_to_saes) == 0  # type: ignore

    top_features = result.top_features(k=5)
    assert [feature for feature, _ in top_features] == torch.topk(
        result.feature_scores[-1], 5
    ).indices.tolist()
    assert [value for _, value in top_features] == pytest.approx(
        result.feature_scores[-1].topk(5).values.tolist()
    )


def test_run_contrastive_projection_top_tokens_match_token_scores(
    model: HookedSAETransformer,
):
    result = run_contrastive_projection(model, SHORT_PROMPT, baselines=SHORT_BASELINE)
    hook = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
    scores = result.token_scores[hook][-1]

    boosted = result.top_tokens(hook, k=5)
    suppressed = result.top_tokens(hook, k=5, largest=False)

    expected_boosted = scores.topk(5)
    assert [token for token, _ in boosted] == [
        model.to_single_str_token(int(index)) for index in expected_boosted.indices
    ]
    assert [value for _, value in boosted] == pytest.approx(
        expected_boosted.values.tolist()
    )
    assert [value for _, value in suppressed] == pytest.approx(
        scores.topk(5, largest=False).values.tolist()
    )
    assert suppressed[-1][1] <= boosted[-1][1]


def test_run_contrastive_projection_rejects_mismatched_and_empty_baselines(
    model: HookedSAETransformer,
):
    with pytest.raises(ValueError, match="same number of positions"):
        run_contrastive_projection(model, SHORT_PROMPT, baselines=BASELINES[0])
    with pytest.raises(ValueError, match="At least one baseline"):
        run_contrastive_projection(model, SHORT_PROMPT, baselines=[])
