import pytest

from benchmark.scaling_extrapolation import (
    ScalingPoint,
    compare_acquisition_strategies,
    extrapolation_error,
    fit_power_law,
)


def test_fit_power_law_recovers_exact_power_law():
    scales = [16.0, 64.0, 256.0, 1024.0, 4096.0]
    values = [3.0 * scale**-0.5 for scale in scales]
    fit = fit_power_law(scales, values)
    assert fit.coefficient == pytest.approx(3.0, abs=1e-8)
    assert fit.exponent == pytest.approx(-0.5, abs=1e-10)
    assert fit.train_r2 == pytest.approx(1.0, abs=1e-12)
    assert fit.predict(256.0) == pytest.approx(3.0 * 256.0**-0.5, rel=1e-9)


def test_fit_power_law_rejects_points_that_cannot_be_logged():
    with pytest.raises(ValueError):
        fit_power_law([1.0, 2.0, 3.0], [1.0, 0.0, 3.0])
    with pytest.raises(ValueError):
        fit_power_law([1.0, -2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        fit_power_law([1.0], [1.0])
    with pytest.raises(ValueError):
        fit_power_law([1.0, 2.0], [1.0])


def test_extrapolation_error_holds_out_largest_scale_from_unsorted_points():
    points = [
        ScalingPoint(scale=512.0, metrics={"mse": 4.0 * 512.0**-0.8}),
        ScalingPoint(scale=128.0, metrics={"mse": 4.0 * 128.0**-0.8}),
        ScalingPoint(scale=2048.0, metrics={"mse": 4.0 * 2048.0**-0.8}),
        ScalingPoint(scale=256.0, metrics={"mse": 4.0 * 256.0**-0.8}),
        ScalingPoint(scale=1024.0, metrics={"mse": 4.0 * 1024.0**-0.8}),
    ]
    report = extrapolation_error(points, "mse")
    assert report.held_out_scale == 2048.0
    assert report.held_out_value == pytest.approx(4.0 * 2048.0**-0.8)
    assert report.predicted_value == pytest.approx(report.held_out_value, rel=1e-9)
    assert report.relative_error == pytest.approx(0.0, abs=1e-9)
    assert report.n_train_points == 4


def test_acquiring_scales_near_the_frontier_extrapolates_better():
    scales = [128.0, 256.0, 384.0, 512.0, 768.0, 1024.0, 2048.0]
    points = [
        ScalingPoint(scale=scale, metrics={"mse": 2.0 * scale**-0.7 + 0.15})
        for scale in scales
    ]
    reports = compare_acquisition_strategies(points, "mse")
    assert (
        reports["even_scales"].relative_error
        < reports["smallest_half"].relative_error
    )


def test_compare_acquisition_strategies_shares_one_held_out_point():
    scales = [64.0, 128.0, 256.0, 512.0]
    points = [
        ScalingPoint(scale=scale, metrics={"mse": 8.0 * scale**-0.6 + 0.01})
        for scale in scales
    ]
    reports = compare_acquisition_strategies(points, "mse")
    assert set(reports) == {"all", "even_scales", "smallest_half"}
    assert reports["all"].n_train_points == 3
    assert reports["even_scales"].n_train_points == 2
    assert reports["smallest_half"].n_train_points == 2
    assert {report.held_out_scale for report in reports.values()} == {512.0}
    assert all(report.relative_error >= 0.0 for report in reports.values())


def test_missing_metric_raises_with_available_options_listed():
    points = [
        ScalingPoint(scale=float(scale), metrics={"l0": 8.0}) for scale in (1, 2, 3)
    ]
    with pytest.raises(ValueError, match="available metrics"):
        extrapolation_error(points, "mse")
