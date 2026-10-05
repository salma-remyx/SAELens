"""Scaling-law extrapolation analysis for SAE capacity sweeps.

Adapted from ScAn-Bench (arXiv:2609.35707), which builds surrogate benchmarks
for evaluating scaling-analysis methodology itself: which scales you acquire
data at (data acquisition), and how well fits over those scales predict
unseen larger scales (extrapolation).

The SAE-native analogue sweeps one capacity axis (d_sae, k, or training
tokens) through the public training API, records run_evals metrics at each
scale, and then asks the paper's two questions of the resulting points:

- Acquisition: which subset of the smaller scales should a budget be spent
  on before extrapolating to the next scale up?
- Extrapolation: how far off is a power law fitted on that subset when it
  predicts the largest, held-out scale?

Power laws (``value = coefficient * scale ** exponent``) are fitted by
ordinary least squares in log-log space, the standard functional form for
scaling-law analysis.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class ScalingPoint:
    """One sweep point: a scale on the swept axis and the metrics recorded there."""

    scale: float
    metrics: dict[str, float]


@dataclass
class PowerLawFit:
    """Fitted power law ``value = coefficient * scale ** exponent``."""

    coefficient: float
    exponent: float
    train_r2: float

    def predict(self, scale: float) -> float:
        return self.coefficient * scale**self.exponent


def fit_power_law(scales: Sequence[float], values: Sequence[float]) -> PowerLawFit:
    """Fit ``value = coefficient * scale ** exponent`` by OLS in log-log space.

    Raises ValueError if fewer than two points are supplied or any scale or
    value is not positive and finite, since their logs would be undefined.
    """
    if len(scales) != len(values):
        raise ValueError("scales and values must have the same length")
    if len(scales) < 2:
        raise ValueError("power law fits need at least two points")
    scale_array = np.asarray(scales, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    for name, array in (("scale", scale_array), ("value", value_array)):
        if not np.all(np.isfinite(array)) or bool(np.any(array <= 0)):
            raise ValueError(f"every {name} must be positive and finite, got {array}")
    log_scales = np.log(scale_array)
    log_values = np.log(value_array)
    exponent, intercept = np.polyfit(log_scales, log_values, 1)
    residuals = log_values - (intercept + exponent * log_scales)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((log_values - np.mean(log_values)) ** 2))
    train_r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return PowerLawFit(
        coefficient=float(np.exp(intercept)),
        exponent=float(exponent),
        train_r2=train_r2,
    )


@dataclass
class ExtrapolationReport:
    """Result of fitting on acquired points and predicting one held-out scale."""

    metric: str
    fit: PowerLawFit
    held_out_scale: float
    held_out_value: float
    predicted_value: float
    relative_error: float
    n_train_points: int


def _point_value(point: ScalingPoint, metric: str) -> float:
    if metric not in point.metrics:
        raise ValueError(
            f"metric {metric!r} not found; available metrics: "
            f"{sorted(point.metrics)}"
        )
    return point.metrics[metric]


def extrapolation_error(
    points: Sequence[ScalingPoint], metric: str
) -> ExtrapolationReport:
    """Hold out the largest-scale point, fit a power law on the rest, and score.

    This mirrors the setting the paper evaluates: scaling laws are fitted on
    acquired scales and then used to prescribe behaviour at the next, unmeasured
    scale up, so the held-out error is the quantity a methodology should be
    judged by. Requires at least three points.
    """
    ordered = sorted(points, key=lambda point: point.scale)
    if len(ordered) < 3:
        raise ValueError("extrapolation needs at least three sweep points")
    train_points, held_out = ordered[:-1], ordered[-1]
    fit = fit_power_law(
        [point.scale for point in train_points],
        [_point_value(point, metric) for point in train_points],
    )
    held_out_value = _point_value(held_out, metric)
    predicted_value = fit.predict(held_out.scale)
    return ExtrapolationReport(
        metric=metric,
        fit=fit,
        held_out_scale=held_out.scale,
        held_out_value=held_out_value,
        predicted_value=predicted_value,
        relative_error=abs(predicted_value - held_out_value) / held_out_value,
        n_train_points=len(train_points),
    )


def _acquire_all(points: list[ScalingPoint]) -> list[ScalingPoint]:
    return points


def _acquire_even_scales(points: list[ScalingPoint]) -> list[ScalingPoint]:
    return points[::2]


def _acquire_smallest_half(points: list[ScalingPoint]) -> list[ScalingPoint]:
    return points[: (len(points) + 1) // 2]


AcquisitionStrategy = Callable[[list[ScalingPoint]], list[ScalingPoint]]

ACQUISITION_STRATEGIES: dict[str, AcquisitionStrategy] = {
    "all": _acquire_all,
    "even_scales": _acquire_even_scales,
    "smallest_half": _acquire_smallest_half,
}


def compare_acquisition_strategies(
    points: Sequence[ScalingPoint], metric: str
) -> dict[str, ExtrapolationReport]:
    """Judge acquisition strategies by their extrapolation to the largest scale.

    Every strategy selects from the same acquired points (all but the largest
    scale, which is held out for everyone) and is scored against the same
    target, so only the acquisition choice differs. ``even_scales`` and
    ``smallest_half`` spend the same budget — half the acquired scales — but
    spread it across the range versus clustering it far below the frontier.
    Strategies that would leave fewer than two fit points are omitted.
    """
    ordered = sorted(points, key=lambda point: point.scale)
    train_points, held_out = ordered[:-1], ordered[-1]
    reports: dict[str, ExtrapolationReport] = {}
    for name, select in ACQUISITION_STRATEGIES.items():
        acquired = select(list(train_points))
        if len(acquired) < 2:
            continue
        reports[name] = extrapolation_error([*acquired, held_out], metric)
    return reports
