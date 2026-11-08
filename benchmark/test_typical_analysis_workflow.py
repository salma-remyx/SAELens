"""Matched-budget selection benchmark: JumpReLU threshold vs calibration-free top-k.

BatchTopK SAEs are saved as JumpReLU SAEs whose threshold is an exponential moving
average learned from the training data, so at inference the feature budget is only
known after the fact and varies from sample to sample. This workflow trains a small
BatchTopK SAE just long enough to learn that threshold, reloads it in the JumpReLU
form it is saved as, and measures what budget each selection policy actually
delivers when both are run at a matched average budget.
"""

from pathlib import Path

import torch

from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE
from sae_lens.saes.jumprelu_sae import JumpReLUSAE
from sae_lens.saes.sae import SAE, TrainStepInput
from sae_lens.saes.topk_selection import (
    compare_threshold_and_topk,
    l0_per_sample,
    matched_budget_k,
    selection_budget,
)
from tests.helpers import build_batchtopk_sae_training_cfg

# The way to run this with this command:
# poetry run py.test benchmark/test_typical_analysis_workflow.py -s
NOMINAL_K = 10


def _train_batchtopk_sae_then_reload_as_inference(
    tmp_path: Path, n_steps: int = 200
) -> JumpReLUSAE:
    cfg = build_batchtopk_sae_training_cfg(
        k=NOMINAL_K,
        d_in=64,
        d_sae=256,
        device="cpu",
        rescale_acts_by_decoder_norm=False,
    )
    training_sae = BatchTopKTrainingSAE(cfg)
    # the SAE sees one fixed batch, so its threshold is calibrated on that data
    # and then judged on activations it never saw
    step_input = TrainStepInput(
        sae_in=torch.randn(256, cfg.d_in),
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=False,
    )
    with torch.no_grad():
        for _ in range(n_steps):
            training_sae.training_forward_pass(step_input)

    training_sae.save_inference_model(str(tmp_path / "inference_sae"))
    inference_sae = SAE.load_from_disk(str(tmp_path / "inference_sae"), device="cpu")
    assert isinstance(inference_sae, JumpReLUSAE)
    return inference_sae


def test_threshold_policy_does_not_fix_the_feature_budget(tmp_path: Path) -> None:
    sae = _train_batchtopk_sae_then_reload_as_inference(tmp_path)
    x = torch.randn(2000, sae.cfg.d_in)

    budget = selection_budget(
        l0_per_sample(sae.encode(x)), sae.cfg.d_sae, nominal_l0=NOMINAL_K
    )
    print(
        f"calibrated for L0 {NOMINAL_K}, delivers {budget.mean_l0:.2f} "
        f"(+/- {budget.std_l0:.2f}, range {budget.min_l0}-{budget.max_l0}), "
        f"budget ratio {budget.budget_ratio}"
    )

    # a threshold fixes which features can fire, not how many do: the per-sample
    # budget swings between empty and several times the calibrated average
    assert budget.std_l0 > 0.25 * budget.mean_l0
    assert budget.max_l0 > 2 * budget.mean_l0
    assert budget.min_l0 < budget.max_l0


def test_topk_policy_matches_the_average_budget_and_caps_the_spend(
    tmp_path: Path,
) -> None:
    sae = _train_batchtopk_sae_then_reload_as_inference(tmp_path)
    x = torch.randn(2000, sae.cfg.d_in)

    comparison = compare_threshold_and_topk(sae, x, nominal_l0=NOMINAL_K)
    threshold_l0 = l0_per_sample(sae.encode(x))
    print(comparison.summary())

    # the top-k budget is matched to the threshold policy's realized average ...
    assert comparison.k == matched_budget_k(threshold_l0, sae.cfg.d_sae)
    # ... caps the spend per sample instead of leaving it free ...
    assert comparison.topk_budget.max_l0 <= comparison.k
    assert comparison.topk_budget.std_l0 == 0.0
    assert comparison.threshold_budget.max_l0 > 2 * comparison.k
    # ... and reconstructs through the same decoder at the same error, so it is a
    # drop-in replacement rather than a trade
    ratio = comparison.topk_mse.item() / comparison.threshold_mse.item()
    print(f"mse ratio topk/threshold at matched budget: {ratio:.4f}")
    assert 0.98 < ratio < 1.02
