"""
Classical dictionary-learning baselines for mode recovery on synthetic data.

Adapted from the synthetic validation experiment in "User Embeddings are
Superpositions of Interpretable Behavioral Modes"
(https://arxiv.org/abs/2611.08487), which generates dense embeddings as sparse
superpositions of ground-truth behavioral modes and compares how well sparse
autoencoders and classical methods (PCA, ICA, K-means, NMF) recover the
underlying modes.

Each baseline fits a dictionary to a batch of dense activations of shape
(num_samples, hidden_dim) and returns unit-norm atoms of shape
(num_components, hidden_dim), matching the shape convention of
FeatureDictionary.feature_vectors and SAE decoder weights, so recovered atoms
can be scored against ground-truth modes with mean_correlation_coefficient or
feature_uniqueness.

The baselines are implemented with torch only, so no scikit-learn dependency
is required. The paper additionally names recovered modes in natural language
and uses them for zero-shot audience retrieval; that requires an external LLM
and real user embeddings, and is out of scope for this module.
"""

import torch

from sae_lens.synthetic.evals import mean_correlation_coefficient


def _normalize_atoms(atoms: torch.Tensor) -> torch.Tensor:
    """Normalize atoms to unit length along the last dimension."""
    return atoms / atoms.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)


def _centered_covariance(activations: torch.Tensor) -> torch.Tensor:
    """Compute the covariance matrix of mean-centered activations."""
    num_samples = activations.shape[0]
    centered = activations - activations.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(1, num_samples - 1)


def _check_num_components(num_components: int, max_components: int) -> None:
    if num_components > max_components:
        raise ValueError(
            f"num_components ({num_components}) cannot exceed {max_components}."
        )


@torch.no_grad()
def _farthest_point_rows(
    activations: torch.Tensor, num_rows: int, normalize: bool = False
) -> torch.Tensor:
    """
    Select well-spread rows by deterministic farthest-point sampling.

    With normalize=True, distances are measured between unit-normalized rows,
    so duplicate directions are at distance zero and the selected rows span
    distinct directions whenever the data does.
    """
    rows = activations
    if normalize:
        rows = rows / rows.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
    selected = [0]
    min_squared = ((rows - rows[0]) ** 2).sum(dim=1)
    for _ in range(num_rows - 1):
        idx = int(min_squared.argmax())
        selected.append(idx)
        min_squared = torch.minimum(min_squared, ((rows - rows[idx]) ** 2).sum(dim=1))
    return activations[torch.tensor(selected, device=activations.device)].clone()


@torch.no_grad()
def pca_atoms(activations: torch.Tensor, num_components: int) -> torch.Tensor:
    """
    Recover dictionary atoms as the top principal components of the activations.

    PCA can produce at most hidden_dim orthogonal directions, so it cannot
    recover more modes than there are embedding dimensions.

    Args:
        activations: Tensor of shape (num_samples, hidden_dim)
        num_components: Number of components to recover

    Returns:
        Unit-norm atoms of shape (num_components, hidden_dim)
    """
    _check_num_components(num_components, activations.shape[1])
    _, eigenvectors = torch.linalg.eigh(_centered_covariance(activations))
    # eigh returns eigenvalues in ascending order, so the top components are last.
    return _normalize_atoms(eigenvectors[:, -num_components:].T)


@torch.no_grad()
def ica_atoms(
    activations: torch.Tensor, num_components: int, max_iters: int = 200
) -> torch.Tensor:
    """
    Recover dictionary atoms with FastICA (symmetric fixed-point, tanh contrast).

    Modes are the columns of the estimated mixing matrix: the directions in
    embedding space along which projections are maximally non-Gaussian (sparse).
    Like PCA, this can produce at most hidden_dim directions.

    Args:
        activations: Tensor of shape (num_samples, hidden_dim)
        num_components: Number of components to recover
        max_iters: Maximum number of fixed-point iterations

    Returns:
        Unit-norm atoms of shape (num_components, hidden_dim)
    """
    hidden_dim = activations.shape[1]
    num_samples = activations.shape[0]
    _check_num_components(num_components, hidden_dim)

    eigenvalues, eigenvectors = torch.linalg.eigh(_centered_covariance(activations))
    top_values = eigenvalues[-num_components:].clamp(min=1e-10)
    top_vectors = eigenvectors[:, -num_components:]
    # Whiten the data to identity covariance in the top component subspace.
    whitened = (
        (activations - activations.mean(dim=0, keepdim=True))
        @ top_vectors
        @ torch.diag(top_values.rsqrt())
    )
    # Maps whitened coordinates back to the embedding space.
    dewhitening = top_vectors @ torch.diag(top_values.sqrt())

    unmixing = torch.linalg.qr(
        torch.randn(
            num_components,
            num_components,
            device=activations.device,
            dtype=activations.dtype,
        )
    ).Q
    for _ in range(max_iters):
        sources = whitened @ unmixing.T
        g = torch.tanh(sources)
        g_prime_mean = (1 - g**2).mean(dim=0)
        updated = g.T @ whitened / num_samples - g_prime_mean.unsqueeze(1) * unmixing
        # Symmetric orthogonalization of the rows.
        new_unmixing = torch.linalg.qr(updated.T).Q.T
        converged = (
            torch.allclose(new_unmixing, unmixing, atol=1e-6)
            or torch.allclose(new_unmixing, -unmixing, atol=1e-6)
        )
        unmixing = new_unmixing
        if converged:
            break

    # Rows of (unmixing @ dewhitening.T) are the columns of the mixing matrix.
    return _normalize_atoms(unmixing @ dewhitening.T)


@torch.no_grad()
def kmeans_atoms(
    activations: torch.Tensor, num_components: int, num_iters: int = 100
) -> torch.Tensor:
    """
    Recover dictionary atoms as normalized K-means cluster centroids.

    Args:
        activations: Tensor of shape (num_samples, hidden_dim)
        num_components: Number of clusters (atoms) to recover
        num_iters: Maximum number of Lloyd's algorithm iterations

    Returns:
        Unit-norm atoms of shape (num_components, hidden_dim)
    """
    num_samples = activations.shape[0]
    _check_num_components(num_components, num_samples)

    centroids = _farthest_point_rows(activations, num_components)
    assignments = torch.full(
        (num_samples,), -1, dtype=torch.long, device=activations.device
    )
    for _ in range(num_iters):
        new_assignments = torch.cdist(activations, centroids).argmin(dim=1)
        if torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments
        counts = torch.bincount(assignments, minlength=num_components)
        sums = torch.zeros_like(centroids).index_add_(0, assignments, activations)
        means = sums / counts.clamp(min=1).unsqueeze(1)
        # Keep the previous centroid for clusters that lost all their points.
        nonempty = (counts > 0).unsqueeze(1)
        centroids = torch.where(nonempty, means, centroids)

    return _normalize_atoms(centroids)


@torch.no_grad()
def nmf_atoms(
    activations: torch.Tensor, num_components: int, num_iters: int = 300
) -> torch.Tensor:
    """
    Recover dictionary atoms with non-negative matrix factorization.

    Activations are clamped at zero before fitting since NMF requires
    non-negative data; superposition embeddings are signed, which is part of
    why NMF struggles in this setting.

    The factor matrix is initialized from well-spread data rows and updated
    with multiplicative updates, alternating several coefficient updates per
    factor update so the factors keep their directions while the coefficients
    converge.

    Args:
        activations: Tensor of shape (num_samples, hidden_dim)
        num_components: Number of factors (atoms) to recover
        num_iters: Number of factor update rounds (each with coefficient updates)

    Returns:
        Unit-norm atoms of shape (num_components, hidden_dim)
    """
    x = activations.clamp(min=0.0)
    num_samples = x.shape[0]
    eps = 1e-10
    factors = _farthest_point_rows(x, num_components, normalize=True)
    # Initialize so that the expected product roughly matches the data scale.
    scale = (x.mean() / num_components).clamp(min=eps).sqrt()
    coefficients = torch.rand(num_samples, num_components, device=x.device) * scale
    for _ in range(num_iters):
        for _ in range(10):
            coefficient_denom = coefficients @ (factors @ factors.T) + eps
            coefficients = coefficients * (x @ factors.T) / coefficient_denom
        factor_denom = (coefficients.T @ coefficients) @ factors + eps
        factors = factors * (coefficients.T @ x) / factor_denom

    return _normalize_atoms(factors)


@torch.no_grad()
def evaluate_classical_baselines(
    activations: torch.Tensor,
    ground_truth_features: torch.Tensor,
    num_components: int | None = None,
) -> dict[str, float]:
    """
    Fit every classical baseline and score mode recovery against ground truth.

    Args:
        activations: Tensor of shape (num_samples, hidden_dim)
        ground_truth_features: Ground truth modes of shape (num_features, hidden_dim)
        num_components: Number of dictionary atoms each baseline should recover.
            Defaults to the number of ground truth features. PCA and ICA are
            capped at hidden_dim components since they cannot produce more
            directions than there are dimensions.

    Returns:
        Dict mapping each baseline name ("pca", "ica", "kmeans", "nmf") to its
        Mean Correlation Coefficient against the ground truth modes.
    """
    if num_components is None:
        num_components = ground_truth_features.shape[0]
    linear_components = min(num_components, activations.shape[1])
    return {
        "pca": mean_correlation_coefficient(
            pca_atoms(activations, linear_components), ground_truth_features
        ),
        "ica": mean_correlation_coefficient(
            ica_atoms(activations, linear_components), ground_truth_features
        ),
        "kmeans": mean_correlation_coefficient(
            kmeans_atoms(activations, num_components), ground_truth_features
        ),
        "nmf": mean_correlation_coefficient(
            nmf_atoms(activations, num_components), ground_truth_features
        ),
    }
