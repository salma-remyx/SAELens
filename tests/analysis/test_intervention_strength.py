import math
from collections.abc import Callable

import pytest
import torch

from sae_lens.analysis import (
    calibrate_intervention_strength,
    output_kl,
    positionwise_norm_coefficients,
    run_with_feature_patch,
    scaled_patch,
    total_norm_coefficients,
)
from sae_lens.saes.sae import SAEMetadata
from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig
from tests.helpers import assert_close, random_params

ACT_NAME = "blocks.0.hook_resid_pre"
D_SAE = 8
D_VOCAB = 6
SEQ_LEN = 5
BATCH_SIZE = 4


class LinearReadoutModel:
    """Stand-in for a HookedSAETransformer with a linear feature readout.

    Feature activations start at zero, forward hooks (the same hook functions
    sae_lens builds for feature interventions) patch them, and a fixed readout
    maps them to logits, so logits are an exact linear function of the applied
    patch.
    """

    def __init__(self, d_sae: int, d_vocab: int, silent_features: tuple[int, ...] = ()):
        readout = torch.randn(d_sae, d_vocab)
        for feature in silent_features:
            readout[feature] = 0.0
        self.readout = readout

    def run_with_hooks_with_saes(
        self,
        tokens: torch.Tensor,
        saes: list[object] | None = None,
        fwd_hooks: list[tuple[str, Callable[..., torch.Tensor]]] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        acts = torch.zeros(tokens.shape[0], tokens.shape[1], self.readout.shape[0])
        for name, hook in fwd_hooks or []:
            acts = hook(acts, name)
        return acts @ self.readout


@pytest.fixture(scope="module")
def sae() -> StandardSAE:
    toy_sae = StandardSAE(
        StandardSAEConfig(
            d_in=D_SAE,
            d_sae=D_SAE,
            dtype="float32",
            device="cpu",
            metadata=SAEMetadata(
                model_name="toy-model",
                hook_name=ACT_NAME,
                hook_head_index=None,
                prepend_bos=True,
            ),
        )
    )
    random_params(toy_sae)
    return toy_sae


@pytest.fixture(scope="module")
def tokens() -> torch.Tensor:
    return torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long)


def test_total_norm_coefficients_match_reference_norms():
    patch = torch.randn(BATCH_SIZE, SEQ_LEN, D_SAE)
    reference_patch = torch.randn(BATCH_SIZE, SEQ_LEN, D_SAE)

    coefficients = total_norm_coefficients(patch, reference_patch)

    assert_close(
        torch.linalg.vector_norm(scaled_patch(patch, coefficients).flatten(1), dim=-1),
        torch.linalg.vector_norm(reference_patch.flatten(1), dim=-1),
    )
    assert (coefficients >= 0).all()


def test_total_norm_coefficients_are_zero_for_empty_patches():
    patch = torch.randn(BATCH_SIZE, SEQ_LEN, D_SAE)
    patch[0] = 0.0
    reference_patch = torch.randn(BATCH_SIZE, SEQ_LEN, D_SAE)

    coefficients = total_norm_coefficients(patch, reference_patch)

    assert coefficients[0] == 0.0
    assert_close(
        coefficients[1:],
        torch.linalg.vector_norm(reference_patch.flatten(1), dim=-1)[1:]
        / torch.linalg.vector_norm(patch.flatten(1), dim=-1)[1:],
    )


def test_positionwise_norm_coefficients_match_reference_norms():
    patch = torch.randn(BATCH_SIZE, SEQ_LEN, D_SAE)
    reference_patch = torch.randn(BATCH_SIZE, SEQ_LEN, D_SAE)

    coefficients = positionwise_norm_coefficients(patch, reference_patch)

    assert_close(
        torch.linalg.vector_norm(scaled_patch(patch, coefficients), dim=-1),
        torch.linalg.vector_norm(reference_patch, dim=-1),
    )


def test_output_kl_matches_manual_computation():
    log_three = math.log(3.0)
    # reference puts 3/4 of its mass on token 1, candidate puts it on token 0
    logits = torch.tensor([[[log_three, 0.0]]])
    reference_logits = torch.tensor([[[0.0, log_three]]])

    kl = output_kl(logits, reference_logits)
    two_position_kl = output_kl(
        logits.repeat(1, 2, 1), reference_logits.repeat(1, 2, 1)
    )

    assert kl[0].item() == pytest.approx(log_three / 2)
    assert two_position_kl[0].item() == pytest.approx(log_three)


def test_output_kl_drops_excluded_positions():
    logits = torch.randn(BATCH_SIZE, SEQ_LEN, D_VOCAB)
    reference_logits = torch.randn(BATCH_SIZE, SEQ_LEN, D_VOCAB)
    keep = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
    keep[:, 1] = True

    excluded = output_kl(logits, reference_logits, exclude=~keep)
    kept_only = output_kl(logits[:, 1:2], reference_logits[:, 1:2])

    assert_close(excluded, kept_only)


def test_run_with_feature_patch_scales_logits_by_coefficients(
    sae: StandardSAE, tokens: torch.Tensor
):
    model = LinearReadoutModel(D_SAE, D_VOCAB)
    patch = torch.randn(BATCH_SIZE, SEQ_LEN, D_SAE)
    coefficients = torch.rand(BATCH_SIZE) + 0.5

    logits = run_with_feature_patch(model, sae, tokens, patch, coefficients)

    assert_close(logits, (coefficients[:, None, None] * patch) @ model.readout)


def test_calibrate_intervention_strength_recovers_reference_strengths(
    sae: StandardSAE, tokens: torch.Tensor
):
    model = LinearReadoutModel(D_SAE, D_VOCAB)
    patch = torch.randn(BATCH_SIZE, SEQ_LEN, D_SAE)
    strengths = torch.tensor([0.25, 0.5, 2.0, 4.0])
    reference_logits = run_with_feature_patch(model, sae, tokens, patch, strengths)

    calibration = calibrate_intervention_strength(
        model, sae, tokens, patch, reference_logits
    )

    assert_close(calibration.coefficients, strengths)
    assert_close(calibration.gaps, torch.zeros(BATCH_SIZE), atol=1e-10)
    assert_close(calibration.gaps, calibration.candidate_gaps.min(dim=0).values)
    assert calibration.candidates.shape == (9,)


def test_calibrate_intervention_strength_uses_the_supplied_grid(
    sae: StandardSAE, tokens: torch.Tensor
):
    model = LinearReadoutModel(D_SAE, D_VOCAB)
    patch = torch.randn(BATCH_SIZE, SEQ_LEN, D_SAE)
    strengths = torch.tensor([0.5, 1.0, 2.0, 3.0])
    reference_logits = run_with_feature_patch(model, sae, tokens, patch, strengths)

    calibration = calibrate_intervention_strength(
        model,
        sae,
        tokens,
        patch,
        reference_logits,
        candidates=torch.tensor([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]),
    )

    assert_close(calibration.coefficients, strengths)
    assert_close(calibration.gaps, torch.zeros(BATCH_SIZE), atol=1e-10)


def test_calibration_leaves_smaller_gap_than_total_norm_matching(
    sae: StandardSAE, tokens: torch.Tensor
):
    model = LinearReadoutModel(D_SAE, D_VOCAB, silent_features=(0,))
    reference_patch = torch.randn(BATCH_SIZE, SEQ_LEN, D_SAE)
    # the candidate patch carries a lot of mass on a feature the readout ignores
    candidate_patch = reference_patch.clone()
    candidate_patch[..., 0] = 4.0 * torch.linalg.vector_norm(reference_patch, dim=-1)
    reference_logits = run_with_feature_patch(
        model, sae, tokens, reference_patch, torch.ones(BATCH_SIZE)
    )

    norm_coefficients = total_norm_coefficients(candidate_patch, reference_patch)
    norm_matched_logits = run_with_feature_patch(
        model, sae, tokens, candidate_patch, norm_coefficients
    )
    norm_matched_gap = output_kl(norm_matched_logits, reference_logits).mean()

    calibration = calibrate_intervention_strength(
        model,
        sae,
        tokens,
        candidate_patch,
        reference_logits,
        candidates=torch.tensor([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
    )

    assert_close(calibration.coefficients, torch.ones(BATCH_SIZE))
    assert calibration.mean_gap.item() == pytest.approx(0.0, abs=1e-8)
    assert calibration.mean_gap.item() < norm_matched_gap.item()
