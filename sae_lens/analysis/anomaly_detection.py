"""Unsupervised anomaly detection on SAE feature activations.

Under the linear representation hypothesis, nearby points in SAE feature space
share a small common active support ("local sparsity"). This module implements
FreqMask-KM, an unsupervised detector built on that structure: it models only
safe (in-distribution) activations and flags inputs whose activations deviate
from the safe support, so no unsafe training data is required.

Adapted from "Local Sparsity Enables Unsupervised LLM Safety Detection"
(https://arxiv.org/abs/2609.20129). The paper's FAISS k-means is replaced by a
plain torch implementation; the rest follows the paper's Algorithm 1:

- Fit k-means centroids on safe feature activations to recover neighbourhoods.
- Binarize the safe activations and keep the most frequently active features
  as a global mask (by default 2% of features, matching the 1-2% the paper
  reports needing).
- Score new activations by their L1 distance to the nearest centroid,
  computed only on the masked features.

Typical usage: encode hidden states with an SAE as usual, fit on safe
activations, then score and flag test activations:

    detector = FreqMaskKMeansAnomalyDetector()
    detector.fit(sae.encode(safe_hidden_states))
    scores = detector.scores(sae.encode(test_hidden_states))
    is_anomalous = scores > detector.threshold(false_positive_rate=0.05)
"""

import torch


class FreqMaskKMeansAnomalyDetector:
    """Flags out-of-distribution inputs from their SAE feature activations.

    The detector is fitted on activations from safe inputs only. Scoring uses
    a locally masked L1 distance: the nearest k-means centroid provides the
    local neighbourhood, and the distance is computed only on the features
    that are frequently active across safe data (the frequency mask).

    Args:
        n_clusters: Number of k-means neighbourhoods fitted on safe activations.
        mask_fraction: Fraction of features kept in the frequency mask, in (0, 1].
        max_iterations: Maximum number of Lloyd's k-means update steps.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        mask_fraction: float = 0.02,
        max_iterations: int = 100,
    ) -> None:
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be at least 1, got {n_clusters}.")
        if not 0 < mask_fraction <= 1:
            raise ValueError(f"mask_fraction must be in (0, 1], got {mask_fraction}.")
        self.n_clusters = n_clusters
        self.mask_fraction = mask_fraction
        self.max_iterations = max_iterations
        self.centroids: torch.Tensor | None = None
        self.mask_indices: torch.Tensor | None = None
        self._safe_scores: torch.Tensor | None = None

    def fit(self, safe_activations: torch.Tensor) -> None:
        """Fit centroids and the frequency mask on activations from safe inputs.

        Args:
            safe_activations: Nonnegative feature activations with shape
                (num_samples, num_features), e.g. the output of SAE.encode.
        """
        z = self._validate_activations(safe_activations)
        if self.n_clusters > z.shape[0]:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot exceed the number of safe "
                f"samples ({z.shape[0]})."
            )
        self.centroids = _kmeans(z, self.n_clusters, self.max_iterations)
        frequencies = (z > 0).float().mean(dim=0)
        mask_size = max(1, round(self.mask_fraction * z.shape[1]))
        self.mask_indices = frequencies.topk(mask_size).indices
        self._safe_scores = self.scores(z)

    def scores(self, activations: torch.Tensor) -> torch.Tensor:
        """Return the masked L1 nearest-centroid anomaly score for each input.

        Args:
            activations: Nonnegative feature activations with shape
                (num_samples, num_features) in the fitted feature space.

        Returns:
            Anomaly scores with shape (num_samples,); higher means the input's
            activations deviate more from the safe support.
        """
        z = self._validate_activations(activations)
        if self.centroids is None or self.mask_indices is None:
            raise RuntimeError("fit() must be called with safe activations first.")
        if z.shape[1] != self.centroids.shape[1]:
            raise ValueError(
                f"Expected {self.centroids.shape[1]} features, got {z.shape[1]}."
            )
        centroids = self.centroids.to(z.device)
        mask_indices = self.mask_indices.to(z.device)
        nearest = torch.cdist(z, centroids).argmin(dim=1)
        feature_diffs = (z - centroids[nearest]).abs()[:, mask_indices]
        return feature_diffs.sum(dim=-1)

    def threshold(self, false_positive_rate: float = 0.01) -> float:
        """Return the score threshold for a target safe false-positive rate.

        The threshold is the (1 - false_positive_rate) quantile of the scores
        of the safe activations the detector was fitted on, so flagging inputs
        with a score above it rejects roughly that fraction of safe inputs.

        Args:
            false_positive_rate: Fraction of safe inputs to flag, in (0, 1).
        """
        if not 0 < false_positive_rate < 1:
            raise ValueError(
                f"false_positive_rate must be in (0, 1), got {false_positive_rate}."
            )
        if self._safe_scores is None:
            raise RuntimeError("fit() must be called with safe activations first.")
        return torch.quantile(self._safe_scores, 1 - false_positive_rate).item()

    def _validate_activations(self, activations: torch.Tensor) -> torch.Tensor:
        if activations.dim() != 2:
            raise ValueError(
                "activations must have shape (num_samples, num_features), got "
                f"{tuple(activations.shape)}."
            )
        return activations.detach().float()


def _kmeans(
    x: torch.Tensor,
    n_clusters: int,
    max_iterations: int,
) -> torch.Tensor:
    """Run Lloyd's k-means with random data-point initialization.

    Returns:
        Centroids with shape (n_clusters, num_features). Clusters that lose all
        their points keep their previous centroid.
    """
    init_rows = torch.randperm(x.shape[0], device=x.device)[:n_clusters]
    centroids = x[init_rows].clone()
    for _ in range(max_iterations):
        assignments = torch.cdist(x, centroids).argmin(dim=1)
        counts = torch.bincount(assignments, minlength=n_clusters)
        sums = torch.zeros_like(centroids).index_add_(0, assignments, x)
        occupied = counts > 0
        new_centroids = centroids.clone()
        new_centroids[occupied] = sums[occupied] / counts[occupied].unsqueeze(1)
        converged = torch.allclose(new_centroids, centroids)
        centroids = new_centroids
        if converged:
            break
    return centroids
