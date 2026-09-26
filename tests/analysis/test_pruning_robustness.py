import pytest
import torch
from datasets import Dataset

from sae_lens.analysis.pruning_robustness import (
    compare_pruning_methods,
    estimate_input_second_moment,
    magnitude_pruning_mask,
    perturbation_energy,
    wanda_pruning_mask,
)
from sae_lens.evals import EvalConfig, run_evals
from sae_lens.saes.topk_sae import TopKTrainingSAE
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.activations_store import ActivationsStore
from tests.helpers import TINYSTORIES_MODEL, build_topk_runner_cfg, load_model_cached

TINYSTORIES_TEXT = (
    "Once upon a time there was a little girl who lived in a small house "
    "near a big forest. Every day she liked to play outside with her friends."
)


def test_perturbation_energy_matches_manual_computation():
    # one row: [1, 1] [[1, 0.5], [0.5, 1]] [1, 1]^T = 1 + 0.5 + 0.5 + 1 = 3
    delta = torch.tensor([[1.0, 1.0]])
    sigma = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    assert perturbation_energy(delta, sigma).item() == pytest.approx(3.0)

    # identity delta against a diagonal second moment sums the diagonal
    delta = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sigma = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    assert perturbation_energy(delta, sigma).item() == pytest.approx(5.0)

    # batched deltas against batched second moments
    deltas = torch.stack(
        [
            torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ]
    )
    sigmas = torch.stack(
        [
            torch.tensor([[1.0, 0.5], [0.5, 1.0]]),
            torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
        ]
    )
    assert perturbation_energy(deltas, sigmas).tolist() == pytest.approx([3.0, 5.0])


def test_perturbation_energy_equals_mean_squared_norm_of_output_shift():
    transform = torch.randn(32, 32)
    weight_delta = torch.randn(16, 32) * 0.1
    x_estimate = torch.randn(200_000, 32) @ transform.T
    sigma = estimate_input_second_moment(x_estimate)
    energy = perturbation_energy(weight_delta, sigma).item()

    x_eval = torch.randn(200_000, 32) @ transform.T
    empirical = weight_delta @ x_eval.T  # (16, n_samples)
    empirical_mean_squared_norm = empirical.pow(2).sum(dim=0).mean().item()

    assert empirical_mean_squared_norm == pytest.approx(energy, rel=0.01)


def test_estimate_input_second_moment_recovers_known_second_moment():
    scales = torch.logspace(0, 1, 8)
    activations = torch.randn(500_000, 8) * scales
    sigma = estimate_input_second_moment(activations)

    assert torch.allclose(torch.diag(sigma), scales**2, rtol=0.01)
    off_diagonal = sigma - torch.diag(torch.diag(sigma))
    assert off_diagonal.abs().max().item() == pytest.approx(
        0.0, abs=0.01 * scales.max() ** 2
    )

    # a sheared 2d input has known off-diagonal second moment M M^T = [[2, 1], [1, 1]]
    shear = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    sigma_2d = estimate_input_second_moment(torch.randn(500_000, 2) @ shear.T)
    assert torch.allclose(sigma_2d, shear @ shear.T, rtol=0.01, atol=0.01)


def test_pruning_masks_match_hand_computed_selection():
    weights = torch.tensor([[0.9, 0.8, 0.15, 0.1], [0.06, 0.05, 0.9, 0.04]])
    # column norms: [~0.014, 2, 2, 2]
    activations = torch.tensor([[0.01, 0.0, 0.0, 0.0], [0.01, 2.0, 2.0, 2.0]])

    magnitude = magnitude_pruning_mask(weights, 0.5)
    assert magnitude.tolist() == [
        [True, True, False, False],
        [True, False, True, False],
    ]

    wanda = wanda_pruning_mask(weights, activations, 0.5)
    assert wanda.tolist() == [
        [False, True, True, False],
        [False, True, True, False],
    ]


def test_pruning_masks_remove_expected_fraction_per_row():
    weights = torch.randn(8, 64)
    activations = torch.randn(10_000, 64)
    for mask in (
        magnitude_pruning_mask(weights, 0.75),
        wanda_pruning_mask(weights, activations, 0.75),
    ):
        assert mask.shape == weights.shape
        assert torch.equal(mask.sum(dim=-1), torch.full((8,), 16))


def test_wanda_pruning_has_lower_perturbation_energy_than_magnitude():
    scales = torch.logspace(-2, 1, 64)
    activations = torch.randn(50_000, 64) * scales
    weights = torch.randn(32, 64) * 0.1

    energies = compare_pruning_methods(weights, activations, 0.5)

    assert energies["magnitude"] > 0
    assert energies["wanda"] < energies["magnitude"]


def test_wanda_perturbs_model_sae_input_less_than_magnitude():
    model = load_model_cached(TINYSTORIES_MODEL)
    tokens = model.to_tokens(TINYSTORIES_TEXT)
    _, cache = model.run_with_cache(
        tokens,
        names_filter=["blocks.0.hook_post", "blocks.1.hook_resid_pre"],
    )
    post = cache["blocks.0.hook_post"]
    weight = model.blocks[0].mlp.W_out

    energies = compare_pruning_methods(weight.data, post, 0.9)
    assert energies["wanda"] < energies["magnitude"]

    # pruning W_out of block 0 shifts resid_pre of block 1 by exactly dW x,
    # so the observed mean squared-norm of the SAE input shift equals eps^2
    mask = wanda_pruning_mask(weight.data, post, 0.9)
    dense = weight.data.clone()
    weight.data = dense * mask.to(dense.dtype)
    try:
        _, pruned_cache = model.run_with_cache(
            tokens, names_filter=["blocks.1.hook_resid_pre"]
        )
    finally:
        weight.data = dense
    shift = pruned_cache["blocks.1.hook_resid_pre"] - cache["blocks.1.hook_resid_pre"]
    observed = shift.pow(2).sum(dim=-1).mean().item()
    assert observed == pytest.approx(energies["wanda"].item(), rel=0.02)


def test_run_evals_scores_sae_under_pruned_model():
    cfg = build_topk_runner_cfg(hook_name="blocks.1.hook_resid_pre", d_in=64)
    model = load_model_cached(TINYSTORIES_MODEL)
    sae = TopKTrainingSAE(cfg.sae)
    activation_store = ActivationsStore.from_config(
        model, cfg, override_dataset=Dataset.from_list([{"text": "hello world"}] * 2000)
    )

    tokens = model.to_tokens("hello world")
    _, cache = model.run_with_cache(tokens, names_filter=["blocks.0.hook_post"])
    weight = model.blocks[0].mlp.W_out
    mask = wanda_pruning_mask(weight.data, cache["blocks.0.hook_post"], 0.9)
    dense = weight.data.clone()
    weight.data = dense * mask.to(dense.dtype)
    try:
        metrics, _ = run_evals(
            sae=sae,
            activation_store=activation_store,
            activation_scaler=ActivationScaler(),
            model=model,
            eval_config=EvalConfig(
                batch_size_prompts=4,
                compute_variance_metrics=True,
                compute_sparsity_metrics=True,
                n_eval_sparsity_variance_batches=1,
            ),
        )
    finally:
        weight.data = dense

    assert "explained_variance" in metrics["reconstruction_quality"]
    assert "mse" in metrics["reconstruction_quality"]
    assert "l0" in metrics["sparsity"]
