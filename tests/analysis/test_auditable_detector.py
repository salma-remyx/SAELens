import pytest
import torch

from sae_lens.analysis import AuditableDetector
from sae_lens.analysis.auditable_detector import (
    description_fidelity,
    document_feature_activations,
    quantile_bin_assignments,
    spearman_correlation,
)
from sae_lens.saes.standard_sae import StandardSAE
from tests.helpers import build_sae_cfg


def marker_docs(num_docs: int) -> list[str]:
    return [
        f"the essay might delve into topic number {index}"
        if index % 2 == 0
        else f"a plain note about the weather number {index}"
        for index in range(num_docs)
    ]


def marker_labels(num_docs: int) -> torch.Tensor:
    return torch.tensor([1 if index % 2 == 0 else 0 for index in range(num_docs)])


def test_spearman_correlation_ranks_and_handles_ties():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert spearman_correlation(values, torch.exp(values)).item() == pytest.approx(1.0)
    assert spearman_correlation(values, -torch.exp(values)).item() == pytest.approx(
        -1.0
    )
    constant = torch.full((4,), 7.0)
    assert spearman_correlation(values, constant).item() == pytest.approx(0.0)

    tied = torch.tensor([1.0, 1.0, 1.0, 2.0])
    increasing = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert spearman_correlation(tied, increasing).item() == pytest.approx(
        3.0 / (15.0**0.5)
    )


def test_quantile_bin_assignments_keeps_inactive_documents_in_bin_zero():
    activations = torch.tensor([0.0, 5.0, 1.0, 0.0, 2.0, 3.0, 4.0])
    expected_bins = [0, 2, 1, 0, 1, 1, 2]
    assert quantile_bin_assignments(activations, num_bins=2).tolist() == expected_bins
    assert quantile_bin_assignments(torch.zeros(5), num_bins=4).tolist() == [0] * 5


def test_description_fidelity_scores_only_descriptions_that_predict_activations():
    docs = marker_docs(40)
    marker_feature = torch.tensor(
        [2.0 if index % 2 == 0 else 0.0 for index in range(40)]
    )
    faithful = description_fidelity("overuses the word delve", marker_feature, docs)
    unfaithful = description_fidelity("discusses gardening mulch", marker_feature, docs)
    assert faithful == pytest.approx(1.0)
    assert unfaithful == pytest.approx(0.0)


def test_detector_drops_unfaithful_features_and_explains_predictions():
    docs = marker_docs(40)
    labels = marker_labels(40)
    feature_acts = torch.zeros(40, 3)
    feature_acts[:, 0] = labels * 2.0
    feature_acts[:, 1] = (1 - labels) * 2.0
    feature_acts[:, 2] = torch.linspace(0.5, 1.0, 40)

    detector = AuditableDetector().fit(
        docs,
        feature_acts,
        labels,
        {
            0: "overuses the word delve",
            1: "overuses the word delve",
            2: "mentions hiking trails",
        },
    )

    # Feature 1 fires on the opposite documents from its description and
    # feature 2's description never matches the corpus, so only feature 0
    # survives the fidelity filter even though all three features are
    # predictive of the label.
    assert detector.retained_features == [0]
    report = {entry.feature: entry for entry in detector.feature_report()}
    assert report[0].fidelity == pytest.approx(1.0)
    assert report[1].fidelity == pytest.approx(-1.0)
    assert report[2].retained is False
    assert detector.predict(feature_acts).tolist() == labels.tolist()

    explanations = detector.explain(feature_acts, top_k=3)
    assert len(explanations) == 40
    assert explanations[0][0].feature == 0
    assert explanations[0][0].description == "overuses the word delve"
    assert explanations[0][0].contribution == pytest.approx(
        (detector.decision_function(feature_acts) - detector.bias)[0].item()
    )
    assert explanations[1][0].contribution == pytest.approx(0.0)


def test_detector_fit_rejects_inputs_it_cannot_use():
    docs = marker_docs(10)
    labels = marker_labels(10)
    feature_acts = torch.ones(10, 4)
    with pytest.raises(ValueError, match="labels must be binary"):
        AuditableDetector().fit(docs, feature_acts, labels * 2, {0: "delve"})
    with pytest.raises(ValueError, match="description references feature"):
        AuditableDetector().fit(docs, feature_acts, labels, {9: "delve"})
    with pytest.raises(ValueError, match="fidelity threshold"):
        AuditableDetector().fit(docs, feature_acts, labels, {0: "gardening mulch"})


def test_detector_on_sae_encoded_document_activations():
    sae = StandardSAE(build_sae_cfg(d_in=8, d_sae=16))
    with torch.no_grad():
        # Deterministic encoder: feature 0 tracks input channel 0, feature 1
        # tracks input channel 1, every other feature stays dead.
        sae.W_enc.zero_()
        sae.W_enc[0, 0] = 1.0
        sae.W_enc[1, 1] = 1.0
        sae.b_enc.zero_()
        sae.b_dec.zero_()

    num_docs, seq_len = 24, 5
    labels = marker_labels(num_docs)
    docs = marker_docs(num_docs)
    activations = torch.zeros(num_docs, seq_len, 8)
    activations[..., 0] = labels.unsqueeze(1).float()
    activations[..., 1] = torch.linspace(0.1, 0.4, seq_len)

    feature_acts = document_feature_activations(sae, activations)

    assert feature_acts.shape == (num_docs, 16)
    assert torch.equal(feature_acts, sae.encode(activations).max(dim=1).values)
    assert feature_acts[:, 0].tolist() == labels.tolist()

    detector = AuditableDetector().fit(
        docs,
        feature_acts,
        labels,
        {0: "overuses the word delve", 5: "mentions hiking trails"},
    )
    assert detector.retained_features == [0]
    assert detector.predict(feature_acts).tolist() == labels.tolist()
