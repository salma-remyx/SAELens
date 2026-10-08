import numpy as np
import pytest
import torch

from sae_lens.analysis.causal_adjustment import (
    chi2_survival,
    select_adjustment_set,
    subset_sufficiency_pvalue,
)


def test_chi2_survival_matches_closed_form_values():
    for statistic in (0.5, 1.0, 3.0, 7.5, 20.0):
        assert chi2_survival(statistic, 2) == pytest.approx(np.exp(-statistic / 2))
        assert chi2_survival(statistic, 4) == pytest.approx(
            np.exp(-statistic / 2) * (1 + statistic / 2)
        )
    assert chi2_survival(3.841458820694124, 1) == pytest.approx(0.05)
    assert chi2_survival(5.991464547107979, 2) == pytest.approx(0.05)
    assert chi2_survival(0.0, 3) == pytest.approx(1.0)


def test_subset_sufficiency_pvalue_is_one_for_full_and_for_noise_features():
    rng = np.random.default_rng()
    n_samples = 4000
    treatment = (rng.random(n_samples) < 0.5).astype(np.float64)
    noise_features = rng.normal(size=(n_samples, 8))
    n_test = 1200
    pvalue = subset_sufficiency_pvalue(
        treatment[n_test:],
        noise_features[n_test:],
        treatment[:n_test],
        noise_features[:n_test],
        list(range(8)),
    )
    assert pvalue == 1.0
    pvalue_empty = subset_sufficiency_pvalue(
        treatment[n_test:],
        noise_features[n_test:],
        treatment[:n_test],
        noise_features[:n_test],
        [],
    )
    # the treatment is independent of every feature, so the empty subset models
    # it as well as the full set and the test must not reject
    assert pvalue_empty > 0.05


def test_select_adjustment_set_recovers_single_confounding_feature():
    rng = np.random.default_rng()
    n_samples = 4000
    true_effect = 1.5
    confounder = rng.normal(size=n_samples)
    features = rng.normal(size=(n_samples, 12))
    features[:, 7] = confounder
    treatment = (
        rng.random(n_samples) < 1.0 / (1.0 + np.exp(-(2.0 * confounder)))
    ).astype(np.float64)
    outcome = true_effect * treatment + 2.0 * confounder + 0.5 * rng.normal(
        size=n_samples
    )

    result = select_adjustment_set(features, treatment, outcome)

    assert result.ci_test_accepted
    assert 7 in result.feature_indices
    assert len(result.feature_indices) < 6
    # without adjustment the naive estimate is badly biased; adjusting on the
    # recovered confounding feature recovers the true effect
    assert abs(result.unadjusted_effect - true_effect) > 0.5
    assert result.adjusted_effect == pytest.approx(true_effect, abs=0.15)


def test_select_adjustment_set_returns_empty_set_without_confounding():
    rng = np.random.default_rng()
    n_samples = 4000
    treatment = (rng.random(n_samples) < 0.5).astype(np.float64)
    features = rng.normal(size=(n_samples, 8))
    outcome = 1.0 * treatment + 0.5 * rng.normal(size=n_samples)

    result = select_adjustment_set(features, treatment, outcome)

    assert result.feature_indices == []
    assert result.ci_test_accepted
    assert len(result.p_values) == 1
    # with no features admitted, the adjusted estimate is the naive estimate
    assert result.adjusted_effect == pytest.approx(result.unadjusted_effect)
    assert result.unadjusted_effect == pytest.approx(1.0, abs=0.1)


def test_select_adjustment_set_accepts_torch_tensors():
    rng = np.random.default_rng()
    n_samples = 3000
    confounder = rng.normal(size=n_samples)
    features = rng.normal(size=(n_samples, 6))
    features[:, 2] = confounder
    treatment = (rng.random(n_samples) < 1.0 / (1.0 + np.exp(-confounder))).astype(
        np.float64
    )
    outcome = 1.0 * treatment + 1.5 * confounder + 0.3 * rng.normal(size=n_samples)

    result = select_adjustment_set(
        torch.from_numpy(features).float(),
        torch.from_numpy(treatment).float(),
        torch.from_numpy(outcome).float(),
    )

    assert 2 in result.feature_indices
    assert abs(result.adjusted_effect - 1.0) < abs(result.unadjusted_effect - 1.0)
