from pathlib import Path

import pytest
import torch

from sae_lens.registry import get_sae_class, get_sae_training_class
from sae_lens.saes.sae import SAE, TrainStepInput
from sae_lens.saes.sparse_coding_sae import (
    SparseCodingSAE,
    SparseCodingSAEConfig,
    SparseCodingTrainingSAE,
    SparseCodingTrainingSAEConfig,
    _largest_singular_value_squared,
    coactive_similarity_loss,
    sparse_code,
)
from tests.helpers import assert_close, random_params


def make_orthonormal_dictionary(d: int) -> torch.Tensor:
    W = torch.empty(d, d)
    torch.nn.init.orthogonal_(W)
    return W


def make_duplicate_atom_dictionary() -> torch.Tensor:
    # atoms 0 and 1 are identical, atoms 2 and 3 are orthogonal to everything
    W = torch.zeros(4, 4)
    W[0, 0] = 1.0
    W[1, 0] = 1.0
    W[2, 1] = 1.0
    W[3, 2] = 1.0
    return W


def test_sparse_code_matches_closed_form_solution_for_orthonormal_dictionary():
    # for an orthonormal dictionary the nonnegative lasso solution is
    # relu(W^T x - l1) coordinate-wise, and the unit step size reaches it
    W = make_orthonormal_dictionary(10)
    l1 = 0.5
    coeffs = 1.0 + torch.rand(32, 10)
    x = coeffs @ W

    acts = sparse_code(x, W, l1, num_steps=30, refit_steps=0)

    assert_close(acts, (coeffs - l1).relu(), rtol=1e-4, atol=1e-5)


def test_sparse_code_refit_removes_l1_shrinkage_bias_on_the_inferred_support():
    W = make_orthonormal_dictionary(10)
    l1 = 0.5
    coeffs = torch.rand(32, 10)
    x = coeffs @ W

    acts = sparse_code(x, W, l1, num_steps=30, refit_steps=3)

    # the support is exactly {coeffs > l1} and refit recovers the unshrunk
    # coefficient values on it
    expected = coeffs * (coeffs > l1)
    assert_close(acts, expected, rtol=1e-4, atol=1e-5)


def test_sparse_code_finds_better_lasso_solutions_than_planted_or_single_pass():
    d_in = 8
    d_sae = 16
    W = torch.randn(d_sae, d_in)
    W = W / W.norm(dim=-1, keepdim=True)

    true_coeffs = torch.zeros(64, d_sae)
    rows = torch.arange(64).unsqueeze(1)
    cols = torch.rand(64, 3).mul(d_sae).long()
    true_coeffs[rows, cols] = torch.rand(64, 3) + 1.0
    x = true_coeffs @ W
    l1 = 0.05

    def lasso_objective(acts: torch.Tensor) -> float:
        per_sample = 0.5 * (x - acts @ W).pow(2).sum(-1) + l1 * acts.sum(-1)
        return per_sample.mean().item()

    acts = sparse_code(x, W, l1, num_steps=200)
    acts_converged = sparse_code(x, W, l1, num_steps=5000)

    # the solver has converged (running 25x longer does not improve the objective)
    assert lasso_objective(acts) == pytest.approx(
        lasso_objective(acts_converged), rel=1e-3
    )
    # per-sample optimization beats the (feasible) planted coefficients
    assert lasso_objective(acts) < lasso_objective(true_coeffs)
    # per-sample optimization beats a single linear encoder pass, which is the
    # paper's motivation for replacing the encoder with sparse coding
    single_pass_acts = (x @ W.T).relu()
    assert lasso_objective(acts) < lasso_objective(single_pass_acts)


def test_sparse_code_handles_extra_batch_dimensions():
    W = make_orthonormal_dictionary(10)
    x = torch.randn(2, 3, 10)

    acts_3d = sparse_code(x, W, l1_coefficient=0.1, num_steps=30)
    acts_2d = sparse_code(x.reshape(-1, 10), W, l1_coefficient=0.1, num_steps=30)

    assert_close(acts_3d.reshape(-1, 10), acts_2d)


def test_largest_singular_value_squared_of_scaled_orthonormal_dictionary():
    W = 2.5 * make_orthonormal_dictionary(10)

    assert _largest_singular_value_squared(W).item() == pytest.approx(6.25, rel=1e-3)


def test_coactive_similarity_loss_penalizes_co_active_duplicate_atoms():
    # sample 0 activates the duplicate pair {0, 1}; each duplicate sees its
    # partner at cosine similarity 1 > margin, contributing (1 - 0.5)^2 twice
    W = make_duplicate_atom_dictionary()
    acts = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 1.0]])

    loss = coactive_similarity_loss(acts, W, margin=0.5, num_neighbors=3)

    assert loss.item() == pytest.approx(0.25)


def test_coactive_similarity_loss_gradient_pushes_near_duplicate_atoms_apart():
    # atoms 0 and 1 point in nearly the same direction (cosine similarity
    # above the margin but not exactly 1, where the gradient would vanish)
    W = torch.zeros(4, 4)
    W[0, 0] = 1.0
    near_duplicate_direction = torch.tensor([1.0, 0.1, 0.0, 0.0])
    W[1] = near_duplicate_direction / near_duplicate_direction.norm()
    W[2, 2] = 1.0
    W[3, 3] = 1.0
    W = W.requires_grad_(True)
    acts = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 1.0]])

    loss = coactive_similarity_loss(acts, W, margin=0.5, num_neighbors=3)

    similarity = (W[0] @ W[1] / (W[0].norm() * W[1].norm())).item()
    assert loss.item() == pytest.approx((similarity - 0.5) ** 2, rel=1e-5)
    loss.backward()
    assert W.grad[[0, 1]].abs().sum().item() > 0.0
    assert W.grad[2:].abs().max().item() == pytest.approx(0.0, abs=1e-12)


def test_coactive_similarity_loss_ignores_duplicate_atoms_never_co_active_together():
    # atoms 0 and 1 are duplicates but never fire in the same sample
    W = make_duplicate_atom_dictionary()
    acts = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    loss = coactive_similarity_loss(acts, W, margin=0.5, num_neighbors=3)

    assert loss.item() == pytest.approx(0.0)


def test_SparseCodingSAE_encode_and_decode_roundtrip_with_orthonormal_dictionary():
    cfg = SparseCodingSAEConfig(
        d_in=10,
        d_sae=10,
        sparsity_coefficient=0.1,
        fista_steps=30,
        refit_steps=5,
    )
    sae = SparseCodingSAE(cfg)
    with torch.no_grad():
        sae.W_dec.copy_(make_orthonormal_dictionary(10))
        sae.b_dec.copy_(torch.randn(10))

    true_feats = 1.0 + torch.rand(32, 10)
    sae_in = true_feats @ sae.W_dec + sae.b_dec

    feats = sae.encode(sae_in)

    assert_close(feats, true_feats, rtol=1e-4, atol=1e-4)
    assert_close(sae.decode(feats), sae_in, rtol=1e-4, atol=1e-4)


def test_SparseCodingSAE_fold_activation_norm_scaling_factor_preserves_encoding():
    cfg = SparseCodingSAEConfig(
        d_in=10,
        d_sae=10,
        sparsity_coefficient=0.05,
        fista_steps=30,
        refit_steps=5,
    )
    sae = SparseCodingSAE(cfg)
    with torch.no_grad():
        sae.W_dec.copy_(make_orthonormal_dictionary(10))
        sae.b_dec.copy_(0.1 * torch.randn(10))
    inputs = torch.randn(64, 10)

    original_outputs = sae(inputs)
    original_features = sae.encode(inputs)
    assert torch.count_nonzero(original_features).item() > 0

    sae.fold_activation_norm_scaling_factor(2.0)

    assert sae.cfg.sparsity_coefficient == pytest.approx(0.025)
    folded_outputs = 2.0 * sae(inputs / 2.0)
    folded_features = sae.encode(inputs / 2.0)
    assert_close(folded_outputs, original_outputs, rtol=1e-4, atol=1e-5)
    assert_close(folded_features, original_features / 2.0, rtol=1e-4, atol=1e-5)


def test_SparseCodingTrainingSAE_training_forward_pass():
    cfg = SparseCodingTrainingSAEConfig(
        d_in=8,
        d_sae=12,
        decoder_init_norm=1.0,
    )
    sae = SparseCodingTrainingSAE(cfg)
    random_params(sae)
    sae_in = torch.randn(64, 8)

    output = sae.training_forward_pass(
        step_input=TrainStepInput(
            sae_in=sae_in,
            coefficients={"coactive_similarity": 2.0},
            dead_neuron_mask=None,
            n_training_steps=0,
            is_logging_step=False,
        )
    )

    assert set(output.losses.keys()) == {"mse_loss", "coactive_similarity_loss"}
    # coefficients are held fixed per the alternating optimization scheme
    assert not output.feature_acts.requires_grad
    assert torch.count_nonzero(output.hidden_pre).item() == 0

    expected_similarity_loss = 2.0 * coactive_similarity_loss(
        output.feature_acts,
        sae.W_dec,
        margin=cfg.similarity_margin,
        num_neighbors=cfg.num_similarity_neighbors,
    )
    assert_close(
        output.losses["coactive_similarity_loss"],
        expected_similarity_loss,
        rtol=1e-5,
        atol=1e-8,
    )

    output.loss.backward()
    assert sae.W_dec.grad is not None
    assert sae.W_dec.grad.norm().item() > 0.0
    assert sae.b_dec.grad is not None
    assert sae.b_dec.grad.norm().item() > 0.0


def test_sparse_coding_sae_is_registered_in_the_architecture_registry():
    assert get_sae_class("sparse_coding") == (SparseCodingSAE, SparseCodingSAEConfig)
    assert get_sae_training_class("sparse_coding") == (
        SparseCodingTrainingSAE,
        SparseCodingTrainingSAEConfig,
    )


def test_SparseCodingTrainingSAE_save_and_load_inference_sae(tmp_path: Path):
    cfg = SparseCodingTrainingSAEConfig(
        d_in=8,
        d_sae=12,
        fista_steps=20,
        refit_steps=3,
        sparsity_coefficient=0.2,
        decoder_init_norm=1.0,
    )
    training_sae = SparseCodingTrainingSAE(cfg)
    random_params(training_sae)

    model_path = str(tmp_path)
    training_sae.save_inference_model(model_path)

    inference_sae = SAE.load_from_pretrained(model_path, device="cpu")
    assert isinstance(inference_sae, SparseCodingSAE)
    assert inference_sae.cfg.fista_steps == cfg.fista_steps
    assert inference_sae.cfg.refit_steps == cfg.refit_steps
    assert inference_sae.cfg.sparsity_coefficient == cfg.sparsity_coefficient
    assert_close(inference_sae.W_dec, training_sae.W_dec)
    assert_close(inference_sae.b_dec, training_sae.b_dec)

    sae_in = torch.randn(10, cfg.d_in)
    training_feats, _ = training_sae.encode_with_hidden_pre(sae_in)
    inference_feats = inference_sae.encode(sae_in)
    assert_close(inference_feats, training_feats, rtol=1e-4, atol=1e-5)
    assert_close(
        inference_sae.decode(inference_feats),
        training_sae.decode(training_feats),
        rtol=1e-4,
        atol=1e-5,
    )


def test_SparseCodingTrainingSAE_recovers_synthetic_dictionary():
    d_in = 16
    d_sae = 16
    true_dict = make_orthonormal_dictionary(d_sae)

    n_samples = 8192
    true_coeffs = torch.zeros(n_samples, d_sae)
    rows = torch.arange(n_samples).unsqueeze(1)
    cols = torch.rand(n_samples, 3).mul(d_sae).long()
    true_coeffs[rows, cols] = torch.rand(n_samples, 3) + 0.5
    data = true_coeffs @ true_dict

    cfg = SparseCodingTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        sparsity_coefficient=0.4,
        decoder_init_norm=1.0,
    )
    sae = SparseCodingTrainingSAE(cfg)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-2)
    for step in range(3000):
        batch = data[torch.randint(0, n_samples, (256,))]
        output = sae.training_forward_pass(
            step_input=TrainStepInput(
                sae_in=batch,
                coefficients={"coactive_similarity": cfg.similarity_loss_coefficient},
                dead_neuron_mask=None,
                n_training_steps=step,
                is_logging_step=False,
            )
        )
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()

    learned_dict = sae.W_dec.detach()
    learned_dict = learned_dict / learned_dict.norm(dim=-1, keepdim=True)
    best_match_cos = (learned_dict @ true_dict.T).abs().max(dim=1).values

    assert best_match_cos.mean().item() > 0.9
    assert (best_match_cos > 0.99).float().mean().item() > 0.5

    held_out = data[torch.randint(0, n_samples, (512,))]
    with torch.no_grad():
        reconstruction = sae(held_out)
    assert torch.nn.functional.mse_loss(reconstruction, held_out).item() < 1e-3
