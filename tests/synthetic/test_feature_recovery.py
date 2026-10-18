import math

import pytest
import torch

from sae_lens.saes.standard_sae import StandardTrainingSAE, StandardTrainingSAEConfig
from sae_lens.synthetic import (
    ActivationGenerator,
    ClassificationMetrics,
    FeatureDictionary,
    HierarchyNode,
    SyntheticDataEvalResult,
)
from sae_lens.synthetic.feature_recovery import (
    MERGE_COSINE_FLOOR,
    RecoveryMetrics,
    SyntheticRecoveryEvalResult,
    best_feature_cosines,
    compute_recovery_metrics,
    eval_sae_recovery_on_synthetic_data,
    hierarchy_edges,
)
from tests.helpers import random_params


def test_best_feature_cosines_identical_decoder_is_one():
    gt_features = torch.eye(4, 8)

    best_cos = best_feature_cosines(gt_features, gt_features)

    assert best_cos.shape == (4,)
    assert best_cos.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0], abs=1e-6)


def test_best_feature_cosines_mixture_atom_half_matches_both_features():
    gt_features = torch.eye(4, 8)
    mixture = (gt_features[0] + gt_features[1]) / math.sqrt(2)
    decoder = torch.stack([mixture, gt_features[2], gt_features[3]])

    best_cos = best_feature_cosines(decoder, gt_features)

    assert best_cos[:2].tolist() == pytest.approx(
        [1 / math.sqrt(2), 1 / math.sqrt(2)], abs=1e-6
    )
    assert best_cos[2:].tolist() == pytest.approx([1.0, 1.0], abs=1e-6)


def test_hierarchy_edges_skips_organizational_nodes():
    tree = HierarchyNode(
        feature_index=None,
        children=[
            HierarchyNode(
                feature_index=0,
                children=[
                    HierarchyNode(feature_index=1),
                    HierarchyNode(feature_index=2),
                ],
            ),
            HierarchyNode(feature_index=3),
        ],
    )

    assert hierarchy_edges(tree) == [(0, 1), (0, 2)]
    assert hierarchy_edges([tree, HierarchyNode(feature_index=7)]) == [(0, 1), (0, 2)]


def test_compute_recovery_metrics_perfect_recovery():
    gt_features = torch.eye(5)
    roots = HierarchyNode(
        feature_index=0,
        children=[
            HierarchyNode(feature_index=1),
            HierarchyNode(feature_index=2),
        ],
    )

    metrics = compute_recovery_metrics(gt_features, gt_features, hierarchy_roots=roots)

    assert metrics.median_best_cosine == pytest.approx(1.0, abs=1e-6)
    assert metrics.mean_best_cosine == pytest.approx(1.0, abs=1e-6)
    assert metrics.recovered_fraction == 1.0
    assert metrics.recovery_threshold == 0.95
    assert metrics.num_hierarchy_edges == 2
    assert metrics.merged_edge_fraction == 0.0
    assert metrics.mean_edge_cosine_gap == pytest.approx(0.0, abs=1e-6)


def test_compute_recovery_metrics_detects_merged_edge():
    gt_features = torch.eye(6)
    tilted_mixture = (gt_features[0] + 0.9 * gt_features[1]) / math.sqrt(1.81)
    decoder = torch.stack(
        [tilted_mixture, gt_features[2], gt_features[3], gt_features[4], gt_features[5]]
    )
    roots = HierarchyNode(
        feature_index=0,
        children=[HierarchyNode(feature_index=1), HierarchyNode(feature_index=2)],
    )

    metrics = compute_recovery_metrics(decoder, gt_features, hierarchy_roots=roots)

    parent_cos = 1 / math.sqrt(1.81)
    child_cos = 0.9 / math.sqrt(1.81)
    assert parent_cos > MERGE_COSINE_FLOOR

    # Features 0 and 1 share their best atom (the mixture), features 2-5 are exact
    assert metrics.recovered_fraction == pytest.approx(4 / 6)
    assert metrics.merged_edge_fraction == 0.5
    expected_gap = ((parent_cos - child_cos) + (parent_cos - 1.0)) / 2
    assert metrics.mean_edge_cosine_gap == pytest.approx(expected_gap, abs=1e-6)


def test_compute_recovery_metrics_without_hierarchy_has_no_edge_stats():
    gt_features = torch.eye(5)

    metrics = compute_recovery_metrics(gt_features, gt_features)

    assert metrics.num_hierarchy_edges == 0
    assert metrics.merged_edge_fraction == 0.0
    assert metrics.mean_edge_cosine_gap == 0.0


def test_compute_recovery_metrics_unrelated_decoder_is_not_recovered():
    gt_features = torch.eye(5, 64)
    decoder = torch.randn(10, 64)

    metrics = compute_recovery_metrics(decoder, gt_features)

    assert metrics.recovered_fraction == 0.0
    assert metrics.median_best_cosine < 0.5
    assert metrics.merged_edge_fraction == 0.0


def test_recovery_metrics_to_log_dict_is_flat():
    metrics = RecoveryMetrics(
        median_best_cosine=0.6,
        mean_best_cosine=0.65,
        recovered_fraction=0.1,
        recovery_threshold=0.95,
        num_hierarchy_edges=4,
        merged_edge_fraction=0.25,
        mean_edge_cosine_gap=-0.05,
    )

    assert metrics.to_log_dict(prefix="recovery/") == {
        "recovery/median_best_cosine": 0.6,
        "recovery/mean_best_cosine": 0.65,
        "recovery/recovered_fraction": 0.1,
        "recovery/recovery_threshold": 0.95,
        "recovery/num_hierarchy_edges": 4,
        "recovery/merged_edge_fraction": 0.25,
        "recovery/mean_edge_cosine_gap": -0.05,
    }


def make_aggregate_result(true_l0: float, sae_l0: float) -> SyntheticDataEvalResult:
    return SyntheticDataEvalResult(
        true_l0=true_l0,
        sae_l0=sae_l0,
        dead_latents=0,
        shrinkage=1.0,
        explained_variance=0.99,
        mcc=0.5,
        uniqueness=0.5,
        classification=ClassificationMetrics(
            precision=0.5, recall=0.5, f1_score=0.5, accuracy=0.5
        ),
    )


def make_recovery_metrics(
    recovered_fraction: float, merged_edge_fraction: float
) -> RecoveryMetrics:
    return RecoveryMetrics(
        median_best_cosine=0.6,
        mean_best_cosine=0.6,
        recovered_fraction=recovered_fraction,
        recovery_threshold=0.95,
        num_hierarchy_edges=2,
        merged_edge_fraction=merged_edge_fraction,
        mean_edge_cosine_gap=0.0,
    )


def test_phase_classification_precedence():
    recovered = SyntheticRecoveryEvalResult(
        aggregate=make_aggregate_result(2.0, 4.0),
        recovery=make_recovery_metrics(
            recovered_fraction=1.0, merged_edge_fraction=1.0
        ),
    )
    merged = SyntheticRecoveryEvalResult(
        aggregate=make_aggregate_result(2.0, 4.0),
        recovery=make_recovery_metrics(
            recovered_fraction=0.5, merged_edge_fraction=0.5
        ),
    )
    diffuse = SyntheticRecoveryEvalResult(
        aggregate=make_aggregate_result(2.0, 4.0),
        recovery=make_recovery_metrics(
            recovered_fraction=0.5, merged_edge_fraction=0.0
        ),
    )

    assert recovered.phase == "recovered"
    assert merged.phase == "merged"
    assert diffuse.phase == "diffuse"


def test_density_ratio_and_zero_true_l0():
    result = SyntheticRecoveryEvalResult(
        aggregate=make_aggregate_result(2.0, 4.0),
        recovery=make_recovery_metrics(
            recovered_fraction=0.0, merged_edge_fraction=0.0
        ),
    )
    empty = SyntheticRecoveryEvalResult(
        aggregate=make_aggregate_result(0.0, 3.0),
        recovery=make_recovery_metrics(
            recovered_fraction=0.0, merged_edge_fraction=0.0
        ),
    )

    assert result.density_ratio == pytest.approx(2.0)
    assert math.isinf(empty.density_ratio)


def test_combined_to_log_dict_merges_aggregate_and_recovery():
    result = SyntheticRecoveryEvalResult(
        aggregate=make_aggregate_result(2.0, 4.0),
        recovery=make_recovery_metrics(
            recovered_fraction=0.5, merged_edge_fraction=0.0
        ),
    )

    log_dict = result.to_log_dict(prefix="eval/")

    assert log_dict["eval/true_l0"] == 2.0
    assert log_dict["eval/explained_variance"] == 0.99
    assert log_dict["eval/recovery/recovered_fraction"] == 0.5
    assert log_dict["eval/recovery/median_best_cosine"] == 0.6


def test_eval_sae_recovery_on_synthetic_data_matches_underlying_eval():
    hidden_dim = 8
    num_features = 10

    feature_dict = FeatureDictionary(num_features=num_features, hidden_dim=hidden_dim)
    activations_gen = ActivationGenerator(
        num_features=num_features,
        firing_probabilities=0.2,
    )

    cfg = StandardTrainingSAEConfig(d_in=hidden_dim, d_sae=num_features)
    sae = StandardTrainingSAE(cfg)
    with torch.no_grad():
        sae.W_dec.data = feature_dict.feature_vectors.clone()

    roots = HierarchyNode(
        feature_index=0,
        children=[
            HierarchyNode(feature_index=1),
            HierarchyNode(feature_index=2),
        ],
    )

    result = eval_sae_recovery_on_synthetic_data(
        sae=sae,
        feature_dict=feature_dict,
        activations_generator=activations_gen,
        hierarchy_roots=roots,
        num_samples=50_000,
    )

    assert isinstance(result.aggregate, SyntheticDataEvalResult)
    # Expected L0 = 10 features * 0.2 probability = 2.0
    assert result.aggregate.true_l0 == pytest.approx(2.0, abs=0.03)
    assert result.aggregate.mcc > 0.99
    assert result.recovery.num_hierarchy_edges == 2
    assert result.recovery.recovered_fraction == 1.0
    assert result.phase == "recovered"
    assert result.density_ratio > 0


def test_eval_sae_recovery_on_synthetic_data_reports_diffuse_phase():
    hidden_dim = 64
    num_features = 10

    feature_dict = FeatureDictionary(num_features=num_features, hidden_dim=hidden_dim)
    activations_gen = ActivationGenerator(
        num_features=num_features,
        firing_probabilities=0.2,
    )

    cfg = StandardTrainingSAEConfig(d_in=hidden_dim, d_sae=num_features)
    sae = StandardTrainingSAE(cfg)
    random_params(sae)

    roots = HierarchyNode(
        feature_index=0,
        children=[
            HierarchyNode(feature_index=1),
            HierarchyNode(feature_index=2),
            HierarchyNode(feature_index=3),
        ],
    )

    result = eval_sae_recovery_on_synthetic_data(
        sae=sae,
        feature_dict=feature_dict,
        activations_generator=activations_gen,
        hierarchy_roots=roots,
        num_samples=5_000,
    )

    assert result.recovery.num_hierarchy_edges == 3
    assert result.recovery.recovered_fraction == 0.0
    assert result.recovery.median_best_cosine < 0.5
    assert result.phase == "diffuse"
    assert result.aggregate.explained_variance <= 1.0
