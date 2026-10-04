"""Behavioral tests for benchmark.bench_sparse_probing and its wiring into
the typical evaluation workflow."""

import pytest
import torch

from benchmark.bench_sparse_probing import (
    pca_probing_baseline,
    run_sparse_probing_eval,
    sparse_probing_metrics,
)
from benchmark.test_typical_analysis_workflow import (
    build_concept_sae,
    make_labeled_activations,
    run_typical_evaluation_workflow,
)
from sae_lens.saes.standard_sae import StandardSAE
from tests.helpers import build_sae_cfg, random_params


def test_probe_recovers_label_from_single_deterministic_feature():
    concept = torch.zeros(64, 8)
    concept[:, 0] = 2.0
    baseline = torch.zeros(64, 8)

    metrics = sparse_probing_metrics(
        {"baseline": baseline, "concept": concept}, k_values=[1, 2]
    )

    assert metrics["probe_accuracies"] == pytest.approx([1.0, 1.0])
    assert metrics["probe_accuracy_auc"] == pytest.approx(1.0)
    assert metrics["labels"] == ["baseline", "concept"]


def test_probe_handles_more_than_two_labels():
    features = {}
    for class_index in range(3):
        block = torch.zeros(64, 3)
        block[:, class_index] = 3.0
        features[f"class_{class_index}"] = block

    metrics = sparse_probing_metrics(features, k_values=[3])

    assert metrics["probe_accuracies"][0] == pytest.approx(1.0)


def test_pca_baseline_recovers_top_variance_direction():
    activations = torch.randn(2, 1024, 6)
    activations[0, :, 0] += 6.0

    metrics = pca_probing_baseline(
        {"a": activations[0], "b": activations[1]}, k_values=[1]
    )

    assert metrics["probe_accuracies"][0] >= 0.99


def test_probe_accuracy_scores_held_out_rows_not_training_rows():
    concept = torch.randn(1024, 4)
    concept[:, 0] = 0.0
    concept[0::2, 0] = 5.0
    baseline = torch.randn(1024, 4)

    metrics = sparse_probing_metrics(
        {"baseline": baseline, "concept": concept}, k_values=[1, 4]
    )

    assert all(accuracy < 0.55 for accuracy in metrics["probe_accuracies"])


def test_unstructured_features_score_near_chance():
    features = {"a": torch.randn(2048, 16), "b": torch.randn(2048, 16)}

    metrics = sparse_probing_metrics(features, k_values=[1, 4, 16])

    assert all(accuracy < 0.55 for accuracy in metrics["probe_accuracies"])


def test_run_sparse_probing_eval_probes_the_sae_latents():
    activations = torch.rand(2, 256, 4)
    activations[0, :, 0] += 0.5
    activations_by_label = {"a": activations[0], "b": activations[1]}

    sae = StandardSAE(build_sae_cfg(d_in=4, d_sae=4, device="cpu"))
    with torch.no_grad():
        sae.W_enc.copy_(torch.eye(4))
        sae.b_enc.zero_()

    metrics = run_sparse_probing_eval(sae, activations_by_label, k_values=[1, 2, 4])
    with torch.no_grad():
        expected_features = {
            name: sae.encode(rows) for name, rows in activations_by_label.items()
        }
    expected = sparse_probing_metrics(expected_features, k_values=[1, 2, 4])

    downstream = metrics["downstream_sparse_probing"]
    assert downstream["sae_probe_accuracies"] == pytest.approx(
        expected["probe_accuracies"]
    )
    assert downstream["sae_probe_accuracy_auc"] == pytest.approx(
        expected["probe_accuracy_auc"]
    )
    sae_auc = downstream["sae_probe_accuracy_auc"]
    pca_auc = downstream["pca_probe_accuracy_auc"]
    assert downstream["sae_probe_accuracy_advantage"] == pytest.approx(
        sae_auc - pca_auc
    )
    assert downstream["sae_probe_accuracies"][0] > 0.7


def test_workflow_pairs_proxies_with_downstream_probing():
    activations = make_labeled_activations()

    metrics = run_typical_evaluation_workflow(
        build_concept_sae(), activations, k_values=[1, 2]
    )

    assert set(metrics["unsupervised_proxies"]) >= {
        "encoder_norm",
        "encoder_decoder_cosine_sim",
    }
    downstream = metrics["downstream_sparse_probing"]
    assert downstream["sae_probe_accuracies"][0] >= 0.95

    unstructured_sae = StandardSAE(build_sae_cfg(d_in=128, d_sae=256, device="cpu"))
    random_params(unstructured_sae)
    unstructured_metrics = run_typical_evaluation_workflow(
        unstructured_sae, activations, k_values=[1, 2]
    )
    unstructured_downstream = unstructured_metrics["downstream_sparse_probing"]
    assert (
        unstructured_downstream["sae_probe_accuracies"][0]
        < downstream["sae_probe_accuracies"][0] - 0.15
    )
