import pytest
import torch

from sae_lens.analysis.feature_correspondence import (
    best_match_jaccard,
    feature_correspondence,
)
from sae_lens.saes.topk_sae import TopKSAE
from tests.helpers import build_topk_sae_cfg, random_params


def test_best_match_jaccard_matches_hand_computed_set_overlap():
    acts_a = torch.zeros(6, 2)
    acts_a[:3, 0] = 1.0
    acts_a[3:, 1] = 2.0
    acts_b = torch.zeros(6, 2)
    acts_b[1:3, 0] = 1.0
    acts_b[2:5, 1] = 1.0

    report = best_match_jaccard(
        acts_a,
        acts_b,
        divergence_threshold=0.6,
        token_batch_size=2,
        feature_batch_size=1,
    )

    # feature a0 fires on {0, 1, 2}: 2/3 against b0 ({1, 2}), 1/4 against b1 ({2, 3, 4})
    assert report.best_match_similarity[0] == pytest.approx(2 / 3)
    assert report.best_match_indices[0] == 0
    # feature a1 fires on {3, 4, 5}: 0 against b0, 2/4 against b1
    assert report.best_match_similarity[1] == pytest.approx(0.5)
    assert report.best_match_indices[1] == 1
    assert report.n_active_features == 2
    assert torch.equal(report.divergent_features, torch.tensor([1]))
    assert report.divergent_fraction == pytest.approx(0.5)


def test_best_match_jaccard_recovers_permuted_feature_order():
    n_tokens = 96
    n_features = 12
    acts_a = torch.zeros(n_tokens, n_features)
    active_feature = torch.arange(n_tokens) % n_features
    acts_a[torch.arange(n_tokens), active_feature] = 1.0
    permutation = torch.tensor([4, 0, 7, 2, 10, 1, 9, 3, 11, 5, 8, 6])
    acts_b = acts_a[:, permutation]

    report = best_match_jaccard(acts_a, acts_b)

    assert report.n_active_features == n_features
    assert torch.all(report.best_match_similarity == 1.0)
    # feature i of acts_a lands at column permutation.index(i) of acts_b, so the
    # best matches recover the inverse permutation
    assert torch.equal(report.best_match_indices, torch.argsort(permutation))
    assert report.divergent_fraction == 0.0


def test_dead_reference_features_are_not_reported_as_divergent():
    acts_a = torch.zeros(4, 2)
    acts_a[:, 0] = 1.0
    acts_b = torch.zeros(4, 1)
    acts_b[0, 0] = 1.0

    report = best_match_jaccard(acts_a, acts_b, divergence_threshold=0.9)

    # a0 ({0, 1, 2, 3}) vs b0 ({0}) is 1/4: active but far below the threshold
    assert report.best_match_similarity[0] == pytest.approx(0.25)
    # a1 never fires, so it carries no correspondence evidence and must be
    # excluded from the divergent subset despite a similarity of zero
    assert report.best_match_similarity[1] == 0.0
    assert report.n_active_features == 1
    assert torch.equal(report.divergent_features, torch.tensor([0]))
    assert report.divergent_fraction == pytest.approx(1.0)


def test_feature_correspondence_identical_saes_have_no_divergent_features():
    sae = TopKSAE(build_topk_sae_cfg(d_in=64, d_sae=128, k=16))
    random_params(sae)
    activations = torch.rand(2, 1024, 64)

    report = feature_correspondence(sae, sae, activations, divergence_threshold=0.9)

    assert report.best_match_similarity.shape == (128,)
    # encoding the same activations with the same weights gives identical active
    # token position sets, so every feature that fires is its own perfect match
    assert torch.all(report.best_match_similarity[report.active_mask] == 1.0)
    assert torch.equal(
        report.best_match_indices[report.active_mask],
        torch.nonzero(report.active_mask).flatten(),
    )
    assert report.divergent_fraction == 0.0


def test_feature_correspondence_supports_saes_with_different_input_dims():
    sae_a = TopKSAE(build_topk_sae_cfg(d_in=64, d_sae=128, k=16))
    sae_b = TopKSAE(build_topk_sae_cfg(d_in=32, d_sae=96, k=12))
    random_params(sae_a)
    random_params(sae_b)

    report = feature_correspondence(
        sae_a,
        sae_b,
        torch.rand(512, 64),
        torch.rand(512, 32),
    )

    assert report.best_match_similarity.shape == (128,)
    assert torch.all(report.best_match_indices < 96)
    # independently initialized features fire on unrelated token positions, so
    # best-match similarities stay far below the perfect-match value of 1.0
    active_similarity = report.best_match_similarity[report.active_mask]
    assert float(active_similarity.median()) < 0.3

    with pytest.raises(ValueError):
        feature_correspondence(
            sae_a,
            sae_b,
            torch.rand(512, 64),
            torch.rand(256, 32),
        )
