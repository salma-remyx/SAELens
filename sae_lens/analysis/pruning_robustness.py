"""Perturbation-energy analysis of SAE robustness under model pruning.

When the weights W of a model layer are pruned to W', activations read by a
fixed SAE shift by the output perturbation (W' - W) x. The paper "When Pruning
Meets Interpretability: Preserving Sparse Autoencoder Robustness in LLMs"
(arXiv:2608.25941) shows that the resulting SAE degradation is governed by the
perturbation energy

    eps^2 = tr(dW Sigma_in dW^T)

where dW = W' - W is the weight delta and Sigma_in = E[x x^T] is the
second-moment matrix of the layer's input activations. Magnitude pruning
ignores Sigma_in and therefore perturbs the SAE input manifold far more than
activation-aware pruning: with a diagonal Sigma_in, squaring Wanda's
per-output score |W_ij| * ||x_j|| gives exactly the per-weight contribution
W_ij^2 Sigma_jj to eps^2, so keeping the top-scoring weights greedily
minimizes eps^2.

Adapted from arXiv:2608.25941, with Wanda as the activation-aware method;
SparseGPT and the paper's layer-wise sparsity allocation policy are out of
scope. Everything here is parameter-free and operates on plain tensors:

- estimate_input_second_moment: Sigma_in from (n_samples, d_in) activations.
- perturbation_energy: eps^2 from a weight delta and Sigma_in.
- magnitude_pruning_mask / wanda_pruning_mask: per-output-row keep masks.
- compare_pruning_methods: eps^2 of each method at matched sparsity.
"""

import torch


def estimate_input_second_moment(activations: torch.Tensor) -> torch.Tensor:
    """Estimate Sigma_in = E[x x^T] from a stack of input activations.

    Args:
        activations: Tensor of shape (..., d_in); leading dims are flattened.

    Returns:
        Tensor of shape (d_in, d_in) in float32.
    """
    flat = activations.reshape(-1, activations.shape[-1]).float()
    return flat.T @ flat / flat.shape[0]


def perturbation_energy(
    weight_delta: torch.Tensor, input_second_moment: torch.Tensor
) -> torch.Tensor:
    """Compute the perturbation energy eps^2 = tr(dW Sigma_in dW^T).

    This equals the expected squared norm E[||dW x||^2] of the layer output
    perturbation, i.e. the mean squared-norm of the shift of the SAE input
    activations when the SAE reads this layer's output.

    Args:
        weight_delta: Tensor of shape (..., d_out, d_in); pruned minus dense
            weights.
        input_second_moment: Tensor of shape (d_in, d_in), or batched with
            shape (..., d_in, d_in) matching weight_delta.

    Returns:
        Scalar tensor, or a tensor of shape (...) for batched inputs, in
        float32.
    """
    delta = weight_delta.float()
    sigma = input_second_moment.float()
    return torch.einsum("...oi,...ij,...oj->...", delta, sigma, delta)


def _keep_top_scores_mask(scores: torch.Tensor, sparsity: float) -> torch.Tensor:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    d_in = scores.shape[-1]
    n_keep = d_in - int(round(d_in * sparsity))
    if n_keep < 1:
        raise ValueError(
            f"sparsity {sparsity} removes every weight in rows of width {d_in}"
        )
    top = torch.topk(scores, k=n_keep, dim=-1)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter_(-1, top.indices, True)


def magnitude_pruning_mask(weights: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Keep-mask for magnitude pruning: keep the largest |W_ij| per output row.

    This baseline ignores activation geometry, which is exactly why it
    perturbs SAE inputs more than activation-aware pruning at the same
    sparsity.

    Args:
        weights: Tensor of shape (..., d_out, d_in).
        sparsity: Fraction of weights removed from every output row, in
            [0, 1).

    Returns:
        Boolean keep-mask with the same shape as weights.
    """
    return _keep_top_scores_mask(weights.float().abs(), sparsity)


def wanda_pruning_mask(
    weights: torch.Tensor, activations: torch.Tensor, sparsity: float
) -> torch.Tensor:
    """Keep-mask for Wanda pruning: keep the largest |W_ij| * ||x_j|| per row.

    ||x_j|| is the L2 norm of input feature j across the calibration
    activations, so weights on low-energy input directions are removed first.
    This is the activation-aware rule of Sun et al. 2023 (Wanda); with a
    diagonal input second moment its per-row ranking is exactly greedy
    minimization of the perturbation energy.

    Args:
        weights: Tensor of shape (..., d_out, d_in).
        activations: Calibration inputs of shape (..., d_in).
        sparsity: Fraction of weights removed from every output row, in
            [0, 1).

    Returns:
        Boolean keep-mask with the same shape as weights.
    """
    feature_norms = activations.reshape(-1, activations.shape[-1]).float().norm(dim=0)
    return _keep_top_scores_mask(weights.float().abs() * feature_norms, sparsity)


def compare_pruning_methods(
    weights: torch.Tensor, activations: torch.Tensor, sparsity: float
) -> dict[str, torch.Tensor]:
    """Compare perturbation energy of pruning methods at matched sparsity.

    Args:
        weights: Dense layer weights of shape (..., d_out, d_in).
        activations: Calibration inputs of shape (..., d_in).
        sparsity: Fraction of weights removed per output row, in [0, 1).

    Returns:
        Mapping from method name to its perturbation energy eps^2, with keys
        "magnitude" and "wanda".
    """
    sigma = estimate_input_second_moment(activations)
    masks = {
        "magnitude": magnitude_pruning_mask(weights, sparsity),
        "wanda": wanda_pruning_mask(weights, activations, sparsity),
    }
    return {
        name: perturbation_energy(weights.float() * (~mask).float(), sigma)
        for name, mask in masks.items()
    }
