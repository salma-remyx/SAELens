"""Select minimal SAE-feature adjustment sets for causal confounding adjustment.

Adapted from "Exploring Sparse Autoencoders in Text-Based Causal Confounding
Adjustment" (https://arxiv.org/abs/2609.01322v1), which adjusts for confounding
information in text by growing a minimal set of SAE features until conditional
independence tests indicate the treatment is independent of the remaining
features given the selected ones.

The paper ranks features with a Lasso-regularized logistic-regression path and
estimates effects with Coarsened Exact Matching or DoubleML. This module keeps
the selection mechanism — iterative feature admission judged by a held-out
likelihood-ratio chi-square test at a fixed significance level — but substitutes
the auxiliary components with parameter-free equivalents on this repo's stack:
features are admitted greedily in order of how much they improve an
ordinary-least-squares model of the treatment, and the adjusted effect is an
OLS regression-adjusted treatment coefficient.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class AdjustmentSet:
    """
    Result of selecting a minimal SAE-feature adjustment set.

    Attributes:
        feature_indices: Indices of the selected SAE features, in admission order.
        p_values: Conditional-independence test p-values, where entry i is the
            p-value for the first i admitted features (entry 0 is the empty set).
        ci_test_accepted: Whether the final subset passed the conditional
            independence test at the requested significance level.
        unadjusted_effect: Naive OLS coefficient of the outcome on the treatment.
        adjusted_effect: OLS treatment coefficient controlling for the selected
            features.
    """

    feature_indices: list[int]
    p_values: list[float]
    ci_test_accepted: bool
    unadjusted_effect: float
    adjusted_effect: float


def chi2_survival(statistic: float, dof: int) -> float:
    """
    Upper-tail probability of the chi-square distribution with ``dof`` degrees
    of freedom, implemented via the regularized incomplete gamma function so
    the module needs no scipy dependency.
    """
    if dof <= 0:
        return 1.0
    return _regularized_upper_gamma(dof / 2.0, statistic / 2.0)


def subset_sufficiency_pvalue(
    treatment_train: np.ndarray,
    features_train: np.ndarray,
    treatment_test: np.ndarray,
    features_test: np.ndarray,
    subset_indices: list[int],
) -> float:
    """
    Held-out likelihood-ratio p-value for the conditional independence claim
    that the treatment is independent of the unselected features given the
    selected subset.

    OLS models of the treatment are fit on the training split using either the
    subset or all features; twice their held-out Gaussian log-likelihood ratio
    is ``n_test * log(rss_subset / rss_full)``, which under the null follows a
    chi-square distribution with degrees of freedom equal to the number of
    excluded features. High p-values mean the subset models the treatment as
    well as the full feature set, i.e. the subset is sufficient for adjustment.

    Args:
        treatment_train: Treatment values of shape (n_train,).
        features_train: Feature activations of shape (n_train, n_features).
        treatment_test: Held-out treatment values of shape (n_test,).
        features_test: Held-out feature activations of shape (n_test, n_features).
        subset_indices: Indices of the candidate adjustment features.
    """
    beta_subset = _fit_beta(features_train[:, subset_indices], treatment_train)
    beta_full = _fit_beta(features_train, treatment_train)
    rss_subset = _residual_sum_of_squares(
        beta_subset, features_test[:, subset_indices], treatment_test
    )
    rss_full = _residual_sum_of_squares(beta_full, features_test, treatment_test)
    if rss_full <= 0.0:
        return 0.0
    statistic = len(treatment_test) * math.log(rss_subset / rss_full)
    if statistic <= 0.0:
        return 1.0
    return chi2_survival(statistic, features_train.shape[1] - len(subset_indices))


def select_adjustment_set(
    feature_acts: torch.Tensor | np.ndarray,
    treatment: torch.Tensor | np.ndarray,
    outcome: torch.Tensor | np.ndarray,
    alpha: float = 0.05,
    test_fraction: float = 0.3,
    max_features: int | None = None,
) -> AdjustmentSet:
    """
    Select a minimal set of SAE features for causal confounding adjustment.

    Starting from an empty set, features are admitted one at a time, each time
    adding the feature that most improves an OLS model of the treatment, until
    a held-out likelihood-ratio test fails to reject conditional independence
    of the treatment and the remaining features. The returned effects compare
    the naive outcome-on-treatment coefficient against the coefficient adjusted
    for the selected features; for a binary treatment the unadjusted effect is
    the difference in mean outcomes between treatment arms.

    Args:
        feature_acts: Sparse feature activations of shape (n_samples, n_features),
            e.g. the output of ``sae.encode(activations)``.
        treatment: Treatment values of shape (n_samples,).
        outcome: Outcome values of shape (n_samples,).
        alpha: Significance level for the conditional independence test.
        test_fraction: Fraction of samples held out for the likelihood-ratio test.
        max_features: Optional cap on the number of admitted features.
    """
    features = _to_numpy(feature_acts)
    treatment_values = _to_numpy(treatment).reshape(-1)
    outcome_values = _to_numpy(outcome).reshape(-1)
    if features.ndim != 2:
        raise ValueError("feature_acts must have shape (n_samples, n_features)")
    n_samples, n_features = features.shape
    if len(treatment_values) != n_samples or len(outcome_values) != n_samples:
        raise ValueError(
            "treatment and outcome must have n_samples entries, got "
            f"{len(treatment_values)} and {len(outcome_values)} for {n_samples} samples"
        )

    rng = np.random.default_rng()
    permutation = rng.permutation(n_samples)
    n_test = int(n_samples * test_fraction)
    test_indices = permutation[:n_test]
    train_indices = permutation[n_test:]
    features_train = features[train_indices]
    treatment_train = treatment_values[train_indices]
    features_test = features[test_indices]
    treatment_test = treatment_values[test_indices]

    def sufficiency_pvalue(subset: list[int]) -> float:
        return subset_sufficiency_pvalue(
            treatment_train, features_train, treatment_test, features_test, subset
        )

    feature_cap = n_features if max_features is None else min(max_features, n_features)
    selected: list[int] = []
    p_values: list[float] = []
    while True:
        p_value = sufficiency_pvalue(selected)
        p_values.append(p_value)
        if p_value >= alpha or len(selected) >= feature_cap:
            break
        best_feature = _most_predictive_feature(
            features_train, treatment_train, selected
        )
        if best_feature is None:
            break
        selected.append(best_feature)

    no_features = np.zeros((n_samples, 0))
    return AdjustmentSet(
        feature_indices=selected,
        p_values=p_values,
        ci_test_accepted=p_values[-1] >= alpha,
        unadjusted_effect=_treatment_effect(
            no_features, treatment_values, outcome_values
        ),
        adjusted_effect=_treatment_effect(
            features[:, selected], treatment_values, outcome_values
        ),
    )


def _to_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def _design(features: np.ndarray) -> np.ndarray:
    return np.hstack([features, np.ones((features.shape[0], 1))])


def _fit_beta(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(_design(features), target, rcond=None)
    return beta


def _residual_sum_of_squares(
    beta: np.ndarray, features: np.ndarray, target: np.ndarray
) -> float:
    residuals = target - _design(features) @ beta
    return float(residuals @ residuals)


def _most_predictive_feature(
    features_train: np.ndarray,
    treatment_train: np.ndarray,
    selected: list[int],
) -> int | None:
    """
    Index of the unselected feature that most reduces training residual error
    of the OLS treatment model, or None if no remaining feature helps.
    """
    current_rss = _residual_sum_of_squares(
        _fit_beta(features_train[:, selected], treatment_train),
        features_train[:, selected],
        treatment_train,
    )
    best_feature = None
    best_rss = current_rss
    for feature_index in range(features_train.shape[1]):
        if feature_index in selected:
            continue
        candidate = selected + [feature_index]
        candidate_rss = _residual_sum_of_squares(
            _fit_beta(features_train[:, candidate], treatment_train),
            features_train[:, candidate],
            treatment_train,
        )
        if candidate_rss < best_rss:
            best_rss = candidate_rss
            best_feature = feature_index
    return best_feature


def _treatment_effect(
    features: np.ndarray, treatment: np.ndarray, outcome: np.ndarray
) -> float:
    beta, *_ = np.linalg.lstsq(
        _design(np.column_stack([treatment, features])), outcome, rcond=None
    )
    return float(beta[0])


def _regularized_upper_gamma(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function Q(a, x)."""
    if x <= 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _regularized_lower_gamma_series(a, x)
    return _upper_gamma_continued_fraction(a, x)


def _regularized_lower_gamma_series(a: float, x: float) -> float:
    log_prefactor = a * math.log(x) - x - math.lgamma(a)
    term = 1.0 / a
    total = term
    for n in range(1, 10_000):
        term *= x / (a + n)
        total += term
        if abs(term) < abs(total) * 1e-16:
            break
    return total * math.exp(log_prefactor)


def _upper_gamma_continued_fraction(a: float, x: float) -> float:
    log_prefactor = a * math.log(x) - x - math.lgamma(a)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return min(1.0, max(0.0, math.exp(log_prefactor) * h))
