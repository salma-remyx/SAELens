import math

import pytest
import torch

from sae_lens.analysis import (
    BilingualAlignment,
    bilingual_alignment,
    collect_token_attributions,
    compute_monosemanticity,
    feature_mask,
    language_selectivity,
    make_feature_injection_hook,
    peak_alignment_layer,
    transfer_decomposed_features,
)
from sae_lens.saes.standard_sae import StandardSAE
from tests.helpers import assert_close, assert_not_close, build_sae_cfg, random_params


def test_language_selectivity_classifies_language_specific_and_mixed_features():
    acts_a = torch.zeros(4, 5)
    acts_b = torch.zeros(4, 5)
    acts_a[:, 0] = torch.tensor([1.0, 0.9, 0.0, 0.8])
    acts_b[:, 1] = torch.tensor([0.5, 0.0, 0.7, 0.6])
    acts_a[:, 2] = torch.tensor([1.0, 0.0, 0.8, 0.0])
    acts_b[:, 2] = torch.tensor([0.0, 0.9, 0.0, 0.7])

    selectivity = language_selectivity(acts_a, acts_b)

    assert selectivity.labels == ["language_a", "language_b", "mixed", "dead", "dead"]
    assert selectivity.p_language_a.tolist() == pytest.approx([1.0, 0.0, 0.5, 0.5, 0.5])
    assert selectivity.counts_a.tolist() == [3, 0, 2, 0, 0]
    assert selectivity.counts_b.tolist() == [0, 3, 2, 0, 0]
    assert selectivity.activation_counts.tolist() == [3, 3, 4, 0, 0]
    assert selectivity.live_mask.tolist() == [True, True, True, False, False]
    assert selectivity.mixed_proportion == pytest.approx(1 / 3)


def test_language_selectivity_ignores_positions_below_fraction_of_max():
    acts_a = torch.tensor([[1.0, 0.0], [0.71, 0.0], [0.69, 0.0], [0.0, 0.0]])
    acts_b = torch.zeros(4, 2)

    selectivity = language_selectivity(acts_a, acts_b, activation_fraction=0.7)
    assert selectivity.counts_a.tolist() == [2, 0]
    assert selectivity.labels == ["language_a", "dead"]

    stricter = language_selectivity(acts_a, acts_b, activation_fraction=0.8)
    assert stricter.counts_a.tolist() == [1, 0]


def test_feature_mask_ranks_features_by_language_balance():
    n = 20
    acts_a = torch.zeros(n, 5)
    acts_b = torch.zeros(n, 5)
    for feature, (n_a, n_b) in enumerate([(5, 5), (6, 4), (15, 5), (10, 0), (0, 10)]):
        acts_a[:n_a, feature] = 1.0
        acts_b[:n_b, feature] = 1.0
    selectivity = language_selectivity(acts_a, acts_b)
    assert selectivity.p_language_a.tolist() == pytest.approx(
        [0.5, 0.6, 0.75, 1.0, 0.0]
    )

    all_mixed = feature_mask(selectivity, "mixed")
    assert all_mixed.tolist() == [True, True, True, False, False]
    assert feature_mask(selectivity, "mixed", m=2).tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert feature_mask(selectivity, "language_a", m=1).tolist() == [
        False,
        False,
        False,
        True,
        False,
    ]
    assert feature_mask(selectivity, "language_b", m=1).tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_collect_token_attributions_counts_only_activating_tokens():
    feature_acts = torch.tensor([[1.0, 0.0], [0.75, 0.0], [0.0, 0.9], [0.6, 0.8]])
    token_ids = torch.tensor([10, 11, 10, 12])

    attributions = collect_token_attributions(feature_acts, token_ids)

    assert attributions == [{10: 1, 11: 1}, {10: 1, 12: 1}]


def test_compute_monosemanticity_groups_tokens_into_semantic_clusters():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    attributions = [
        {0: 1, 1: 1, 2: 1, 3: 1},
        {0: 1, 1: 1},
        {0: 1, 2: 1},
        {4: 3},
        {},
        {0: 1, 4: 1},
    ]

    result = compute_monosemanticity(attributions, embeddings)

    assert result.token_entropy.tolist() == pytest.approx(
        [math.log(4), math.log(2), math.log(2), 0.0, 0.0, math.log(2)]
    )
    assert result.semantic_entropy.tolist() == pytest.approx(
        [math.log(2), 0.0, math.log(2), 0.0, 0.0, 0.0]
    )
    assert result.score.tolist() == pytest.approx([0.5, 1.0, 0.0, 1.0, 1.0, 1.0])

    strict = compute_monosemanticity(attributions, embeddings, cos_threshold=0.9)
    assert strict.score.tolist() == pytest.approx([0.5, 1.0, 0.0, 1.0, 1.0, 0.0])


def test_bilingual_alignment_excludes_dead_features_from_monosemanticity():
    acts_a = torch.zeros(4, 4)
    acts_b = torch.zeros(4, 4)
    acts_a[:, 0] = 1.0
    acts_b[:, 0] = 1.0
    acts_a[:, 1] = 1.0
    acts_b[:, 1] = 1.0
    acts_b[:, 2] = 1.0
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # feature 0: mixed, fires on two tokens with distinct meanings -> score 0
    # feature 1: mixed, single token -> score 1
    # feature 2: language_b, single token -> score 1
    # feature 3: dead, scored 1 by convention and excluded below
    attributions = [{0: 1, 1: 1}, {0: 1}, {1: 1}, {}]

    profile = bilingual_alignment(
        language_selectivity(acts_a, acts_b),
        compute_monosemanticity(attributions, embeddings),
    )

    assert profile.mixed_proportion == pytest.approx(2 / 3)
    assert profile.mean_monosemanticity == pytest.approx(2 / 3)
    assert profile.score == pytest.approx(4 / 9)


def test_peak_alignment_layer_finds_mid_layer():
    layers = [
        BilingualAlignment(mixed_proportion=0.9, mean_monosemanticity=0.2),
        BilingualAlignment(mixed_proportion=0.8, mean_monosemanticity=0.9),
        BilingualAlignment(mixed_proportion=0.4, mean_monosemanticity=0.95),
    ]
    assert peak_alignment_layer(layers) == 1


def test_transfer_decomposed_features_masks_and_scales_donor_features():
    sae = StandardSAE(build_sae_cfg(d_in=16, d_sae=32))
    random_params(sae)
    donor = torch.rand(3, 4, 16) * 2.0
    mask = torch.zeros(32, dtype=torch.bool)
    mask[5] = True

    update = transfer_decomposed_features(sae, donor, mask, alpha=0.1)

    feature_acts = sae.encode(donor)
    masked_acts = torch.zeros_like(feature_acts)
    masked_acts[..., 5] = feature_acts[..., 5]
    assert_close(update, 0.1 * (masked_acts @ sae.W_dec + sae.b_dec))

    mask[11] = True
    wider = transfer_decomposed_features(sae, donor, mask, alpha=0.1)
    assert wider.shape == update.shape
    assert_not_close(wider, update)


def test_make_feature_injection_hook_adds_update_to_residual():
    sae = StandardSAE(build_sae_cfg(d_in=16, d_sae=32))
    random_params(sae)
    donor = torch.rand(2, 3, 16)
    mask = torch.zeros(32, dtype=torch.bool)
    mask[7] = True

    hook = make_feature_injection_hook(sae, donor, mask, alpha=0.1)
    residual = torch.rand(2, 3, 16)

    expected = residual + transfer_decomposed_features(sae, donor, mask, alpha=0.1)
    assert_close(hook(residual, None), expected)
