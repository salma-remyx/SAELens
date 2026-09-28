import pytest
import torch

from sae_lens.analysis.latent_set_stability import (
    latent_set_jaccard,
    latent_set_jaccard_matrix,
    latent_set_stability,
)
from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE
from tests.helpers import build_batchtopk_sae_training_cfg


def test_latent_set_jaccard_matches_hand_computed_overlaps():
    codes_a = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    codes_b = torch.tensor(
        [
            [1.0, 5.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [3.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    overlaps = latent_set_jaccard(codes_a, codes_b)

    # {0, 2} vs {0, 1} -> 1/3; {} vs {1} -> 0; identical -> 1; {} vs {} -> 1
    expected = torch.tensor([1 / 3, 0.0, 1.0, 1.0])
    torch.testing.assert_close(overlaps, expected, rtol=0, atol=1e-6)


def test_latent_set_jaccard_threshold_shrinks_active_sets():
    codes_a = torch.tensor([[1.0, 0.0, 2.0, 3.0]])
    codes_b = torch.tensor([[1.0, 4.0, 0.0, 3.0]])

    overlaps = latent_set_jaccard(codes_a, codes_b, threshold=1.5)

    # active sets become {2, 3} vs {1, 3} -> 1/3
    assert overlaps[0].item() == pytest.approx(1 / 3)


def test_latent_set_jaccard_matrix_returns_all_pairs_overlaps():
    codes_a = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    codes_b = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )

    matrix = latent_set_jaccard_matrix(codes_a, codes_b)

    assert matrix.shape == (2, 3)
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.0],
        ]
    )
    torch.testing.assert_close(matrix, expected, rtol=0, atol=1e-6)

    self_matrix = latent_set_jaccard_matrix(codes_a)
    torch.testing.assert_close(
        self_matrix, torch.eye(2), rtol=0, atol=1e-6, check_dtype=False
    )


def test_latent_set_jaccard_raises_on_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        latent_set_jaccard(torch.zeros(2, 4), torch.zeros(3, 4))


def test_latent_set_stability_reports_unchanged_sets_for_identical_inputs():
    sae = BatchTopKTrainingSAE(build_batchtopk_sae_training_cfg())
    activations = torch.randn(64, sae.cfg.d_in)

    metrics = latent_set_stability(sae, activations, activations)

    assert metrics["n_examples"] == 64
    assert metrics["mean_jaccard_overlap"] == pytest.approx(1.0)
    assert metrics["identical_set_fraction"] == pytest.approx(1.0)
    assert metrics["mean_retained_fraction"] == pytest.approx(1.0)
    # BatchTopK keeps k features active per row on average across the batch
    assert metrics["mean_active_set_size_base"] == pytest.approx(sae.cfg.k)
    assert metrics["mean_active_set_size_modified"] == pytest.approx(sae.cfg.k)
    assert metrics["per_example_jaccard"] == pytest.approx([1.0] * 64)
    assert metrics["mean_jaccard_overlap"] == pytest.approx(
        torch.tensor(metrics["per_example_jaccard"]).mean().item()
    )


def test_latent_set_stability_overlap_shrinks_as_modification_grows():
    sae = BatchTopKTrainingSAE(build_batchtopk_sae_training_cfg())
    base = torch.randn(1024, sae.cfg.d_in)
    slightly_modified = base + 0.02 * torch.randn_like(base)
    unrelated = base[torch.randperm(base.shape[0])]

    identical_overlap = latent_set_stability(sae, base, base)["mean_jaccard_overlap"]
    small_overlap = latent_set_stability(sae, base, slightly_modified)[
        "mean_jaccard_overlap"
    ]
    unrelated_overlap = latent_set_stability(sae, base, unrelated)[
        "mean_jaccard_overlap"
    ]

    assert identical_overlap == pytest.approx(1.0)
    assert small_overlap > 0.9
    # unrelated inputs share only chance-level set overlap (k^2 / d_sae of
    # about 0.39 shared latents out of ~20 in the union)
    assert unrelated_overlap < 0.3
    assert small_overlap > unrelated_overlap
