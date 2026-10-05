import math

import pytest
import torch

from benchmark.scaling_extrapolation import (
    ScalingPoint,
    compare_acquisition_strategies,
    extrapolation_error,
)
from sae_lens.evals import EvalConfig, run_evals
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.training.activation_scaler import ActivationScaler
from tests.helpers import (
    NEEL_NANDA_C4_10K_DATASET,
    TINYSTORIES_MODEL,
    build_topk_runner_cfg,
    load_model_cached,
)


def _flatten_eval_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for category, category_metrics in metrics.items():
        for name, value in category_metrics.items():
            flattened[f"{category}/{name}"] = value
    return flattened


def test_topk_capacity_sweep_scaling_extrapolation():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # one shared model across the sweep so only d_sae varies between points
    model = load_model_cached(TINYSTORIES_MODEL)
    model.to(device)

    sweep_d_sae = [128, 256, 512, 1024]
    points: list[ScalingPoint] = []
    for d_sae in sweep_d_sae:
        cfg = build_topk_runner_cfg(
            d_in=64,
            d_sae=d_sae,
            k=16,
            device=device,
            act_store_device=device,
            training_tokens=8192,
            train_batch_size_tokens=256,
            store_batch_size_prompts=16,
            context_size=16,
            n_batches_in_buffer=2,
            dataset_path=NEEL_NANDA_C4_10K_DATASET,
            hook_name="blocks.0.hook_resid_post",
            model_name=TINYSTORIES_MODEL,
            n_eval_batches=2,
        )
        runner = LanguageModelSAETrainingRunner(cfg, override_model=model)
        trained_sae = runner.run()

        eval_metrics, _ = run_evals(
            sae=trained_sae,
            activation_store=runner.activations_store,
            activation_scaler=ActivationScaler(),
            model=runner.model,
            eval_config=EvalConfig(
                batch_size_prompts=4,
                n_eval_sparsity_variance_batches=2,
                compute_l2_norms=True,
                compute_sparsity_metrics=True,
            ),
        )
        flattened = _flatten_eval_metrics(eval_metrics)
        print(
            f"d_sae={d_sae} mse={flattened['reconstruction_quality/mse']:.4f} "
            f"l0={flattened['sparsity/l0']:.2f}"
        )
        points.append(ScalingPoint(scale=float(d_sae), metrics=flattened))

    reports = compare_acquisition_strategies(points, "reconstruction_quality/mse")
    assert set(reports) == {"all", "even_scales", "smallest_half"}
    assert reports["all"].held_out_scale == 1024.0
    assert reports["all"].n_train_points == 3
    assert reports["even_scales"].n_train_points == 2
    assert reports["smallest_half"].n_train_points == 2
    for report in reports.values():
        assert math.isfinite(report.predicted_value)
        assert math.isfinite(report.relative_error)
        assert report.relative_error >= 0.0

    # TopK keeps exactly k features live per token, so the swept l0 pins to k
    # at every scale: a wiring check that real run_evals metrics flowed through.
    l0_report = extrapolation_error(points, "sparsity/l0")
    assert l0_report.held_out_value == pytest.approx(16.0, abs=1.0)
    for point in points:
        assert point.metrics["sparsity/l0"] == pytest.approx(16.0, abs=1.0)
