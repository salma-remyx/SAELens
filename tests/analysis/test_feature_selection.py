import math

import pytest
import torch

from sae_lens.analysis.feature_selection import (
    SupervisedFeatureScorer,
    pool_over_context,
    select_supervised_features,
)


def test_pool_over_context_sums_tokens_and_honors_mask():
    feature_acts = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])

    unmasked = pool_over_context(feature_acts)
    masked = pool_over_context(feature_acts, mask)

    assert torch.equal(unmasked, torch.tensor([[6.0, 60.0], [15.0, 150.0]]))
    assert torch.equal(masked, torch.tensor([[3.0, 30.0], [4.0, 40.0]]))
    assert torch.equal(pool_over_context(feature_acts[:, 0]), feature_acts[:, 0])


def test_pool_over_context_rejects_mask_on_pooled_acts():
    with pytest.raises(ValueError):
        pool_over_context(torch.zeros(2, 3), mask=torch.ones(2, dtype=torch.bool))


def test_f_statistics_matches_closed_form_for_two_classes():
    pooled_acts = torch.tensor(
        [
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0],
            [7.0, 5.0],
            [8.0, 5.0],
            [9.0, 5.0],
        ]
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])

    scorer = SupervisedFeatureScorer()
    scorer.add_batch(pooled_acts, labels)

    # Feature 0 has class means 2 and 8 around an overall mean of 5, so
    # SSB = 3*9 + 3*9 = 54 and SSW = 4, giving F = 54 / (4 / 4) = 54.
    # Feature 1 never varies, so it carries no class signal and scores 0.
    assert scorer.f_statistics().tolist() == pytest.approx([54.0, 0.0])


def test_f_statistics_matches_closed_form_for_three_classes():
    pooled_acts = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    scorer = SupervisedFeatureScorer()
    scorer.add_batch(pooled_acts, labels)

    # Class means 1.5, 3.5 and 5.5 around an overall mean of 3.5 give
    # SSB = 16 on 2 df and SSW = 1.5 on 3 df, so F = 8 / 0.5 = 16.
    assert scorer.f_statistics().tolist() == pytest.approx([16.0])


def test_f_statistics_is_infinite_for_perfectly_separated_feature():
    pooled_acts = torch.tensor([[0.0], [0.0], [0.0], [4.0], [4.0], [4.0]])
    labels = torch.tensor([0, 0, 0, 1, 1, 1])

    scorer = SupervisedFeatureScorer()
    scorer.add_batch(pooled_acts, labels)

    assert scorer.f_statistics()[0].item() == math.inf


def test_f_statistics_is_invariant_to_per_feature_constant_offsets():
    # float64 throughout, so the only difference between the two runs is the
    # offset itself rather than rounding of the shifted inputs.
    pooled_acts = torch.randn(200, 6, dtype=torch.float64)
    labels = torch.randint(0, 2, (200,))
    offsets = torch.tensor([-3.0, 0.0, 12.5, -0.25, 7.0, 1.5], dtype=torch.float64)

    base = SupervisedFeatureScorer()
    base.add_batch(pooled_acts, labels)
    shifted = SupervisedFeatureScorer()
    shifted.add_batch(pooled_acts + offsets, labels)

    assert torch.allclose(
        base.f_statistics(), shifted.f_statistics(), rtol=1e-9, atol=1e-9
    )


def test_streaming_batches_match_a_single_batch():
    column_scales = torch.where(torch.randint(0, 2, (16,)) == 0, 0.1, 10.0)
    pooled_acts = torch.randn(120, 16) * column_scales
    labels = torch.randint(0, 3, (120,))

    single = SupervisedFeatureScorer()
    single.add_batch(pooled_acts, labels)
    streamed = SupervisedFeatureScorer()
    for start in range(0, 120, 25):
        streamed.add_batch(pooled_acts[start : start + 25], labels[start : start + 25])

    assert torch.allclose(
        single.f_statistics(), streamed.f_statistics(), rtol=1e-9, atol=1e-9
    )
    assert torch.equal(single.top_features(5), streamed.top_features(5))


def test_top_features_breaks_ties_by_index():
    pooled_acts = torch.zeros(6, 8)
    pooled_acts[:3, [2, 5]] = 1.0
    pooled_acts[3:, [2, 5]] = 3.0
    labels = torch.tensor([0, 0, 0, 1, 1, 1])

    scorer = SupervisedFeatureScorer()
    scorer.add_batch(pooled_acts, labels)

    assert torch.equal(scorer.top_features(2), torch.tensor([2, 5]))
    assert torch.equal(scorer.top_features(64), torch.tensor([2, 5, 0, 1, 3, 4, 6, 7]))


def test_select_supervised_features_pools_then_selects():
    token_acts = torch.zeros(6, 4, 3)
    token_acts[:3, :, 1] = 2.0
    token_acts[3:, :, 1] = 8.0
    token_acts[:, -1, :] = 1e6
    mask = torch.ones(6, 4, dtype=torch.bool)
    mask[:, -1] = False
    labels = torch.tensor([0, 0, 0, 1, 1, 1])

    indices, f_stats = select_supervised_features(token_acts, labels, k=1, mask=mask)

    assert torch.equal(indices, torch.tensor([1]))
    assert f_stats[1].item() == math.inf
    assert f_stats[[0, 2]].tolist() == pytest.approx([0.0, 0.0])


def test_f_statistics_rejects_degenerate_label_sets():
    one_class = SupervisedFeatureScorer()
    one_class.add_batch(torch.ones(4, 3), torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError):
        one_class.f_statistics()

    one_example_per_class = SupervisedFeatureScorer()
    one_example_per_class.add_batch(torch.ones(2, 3), torch.tensor([0, 1]))
    with pytest.raises(ValueError):
        one_example_per_class.f_statistics()


def test_top_features_rejects_invalid_k_and_empty_scorer():
    scorer = SupervisedFeatureScorer()
    scorer.add_batch(torch.ones(4, 3), torch.tensor([0, 0, 1, 1]))

    with pytest.raises(ValueError):
        scorer.top_features(0)
    with pytest.raises(ValueError):
        SupervisedFeatureScorer().top_features(3)
    with pytest.raises(ValueError):
        select_supervised_features(torch.ones(4, 2, 3), torch.tensor([0, 0, 1, 1]), k=0)
