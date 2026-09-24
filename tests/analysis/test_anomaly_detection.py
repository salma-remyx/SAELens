import pytest
import torch

from sae_lens.analysis.anomaly_detection import FreqMaskKMeansAnomalyDetector
from sae_lens.saes.topk_sae import TopKSAE
from tests.helpers import build_topk_sae_cfg, random_params

N_FEATURES = 200
FREQUENT_SUPPORT = list(range(5))
RARE_SUPPORT = list(range(5, 10))
OOD_SUPPORT = list(range(10, 15))


def _blob_activations(
    support: list[int], n_samples: int, value: float
) -> torch.Tensor:
    activations = torch.zeros(n_samples, N_FEATURES)
    activations[:, support] = value + 0.05 * torch.randn(n_samples, len(support))
    return activations


def test_out_of_support_activations_score_above_all_safe_scores():
    safe = torch.cat(
        [
            _blob_activations(FREQUENT_SUPPORT, 400, value=3.0),
            _blob_activations(RARE_SUPPORT, 100, value=6.0),
        ]
    )
    detector = FreqMaskKMeansAnomalyDetector(n_clusters=2, mask_fraction=0.02)
    detector.fit(safe)
    safe_scores = detector.scores(safe)
    ood_scores = detector.scores(_blob_activations(OOD_SUPPORT, 100, value=3.0))
    assert ood_scores.min() > safe_scores.max()


def test_frequency_mask_keeps_most_frequently_active_features():
    activations = torch.zeros(1000, 100)
    activations[:, 0] = 1.0
    activations[::2, 1] = 1.0
    activations[10, 2] = 1.0
    detector = FreqMaskKMeansAnomalyDetector(n_clusters=1, mask_fraction=0.02)
    detector.fit(activations)
    assert detector.mask_indices is not None
    assert set(detector.mask_indices.tolist()) == {0, 1}


def test_threshold_flags_requested_fraction_of_fresh_safe_activations():
    safe = torch.rand(4000, 50)
    fresh_safe = torch.rand(4000, 50)
    detector = FreqMaskKMeansAnomalyDetector(n_clusters=4, mask_fraction=0.02)
    detector.fit(safe)
    flags = detector.scores(fresh_safe) > detector.threshold(false_positive_rate=0.1)
    flagged_fraction = flags.float().mean().item()
    assert flagged_fraction == pytest.approx(0.1, abs=0.02)


def test_topk_sae_inputs_missing_the_safe_direction_score_as_anomalies():
    sae = TopKSAE(build_topk_sae_cfg(d_in=16, d_sae=64, k=8))
    random_params(sae)
    safe_hidden_states = torch.randn(512, 16) + 5.0
    detector = FreqMaskKMeansAnomalyDetector(n_clusters=4, mask_fraction=0.125)
    detector.fit(sae.encode(safe_hidden_states))
    safe_scores = detector.scores(sae.encode(torch.randn(512, 16) + 5.0))
    ood_scores = detector.scores(sae.encode(torch.randn(512, 16)))
    pairwise = (ood_scores.unsqueeze(1) > safe_scores.unsqueeze(0)).float().mean()
    assert pairwise > 0.99
