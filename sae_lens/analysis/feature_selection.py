"""Supervised top-k SAE feature selection from per-example pooled activations.

Adapted from SAE-FiRE (Cances et al., https://arxiv.org/abs/2505.14420), which
ranks SAE features by how well their per-example pooled activations separate
example labels and keeps only the top-k, rather than handing a full (often
>100k wide) SAE feature vector to a downstream classifier.

The two components ported at full fidelity are the paper's sum-pooling
operator over the context dimension and its per-feature ANOVA F-statistic
(between-label variance over within-label variance) used to rank features.
The paper's gradient-boosted-tree importance and its L2-regularised logistic
regression head are deliberately not ported: neither scikit-learn nor XGBoost
is a dependency of this repo, and the F-statistic is the parameter-free
selection criterion the paper benchmarks against.
"""

import torch


def pool_over_context(
    feature_acts: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Sum-pool SAE feature activations over the context dimension.

    Args:
        feature_acts: Token-level activations of shape (batch_size, ctx, num_features).
            Already-pooled activations of shape (batch_size, num_features) are
            returned unchanged.
        mask: Optional boolean tensor of shape (batch_size, ctx). Positions that
            are False are excluded from the sum (padding, special tokens, ...).

    Returns:
        Pooled activations of shape (batch_size, num_features).
    """
    if feature_acts.ndim == 2:
        if mask is not None:
            raise ValueError(
                "A mask requires token-level activations of shape "
                "(batch_size, ctx, num_features)."
            )
        return feature_acts
    if feature_acts.ndim != 3:
        raise ValueError(
            "feature_acts must have shape (batch_size, ctx, num_features) or "
            f"(batch_size, num_features), got {tuple(feature_acts.shape)}."
        )
    if mask is None:
        return feature_acts.sum(dim=1)
    if tuple(mask.shape) != tuple(feature_acts.shape[:2]):
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match token shape "
            f"{tuple(feature_acts.shape[:2])}."
        )
    return (feature_acts * mask.unsqueeze(-1).to(feature_acts.dtype)).sum(dim=1)


class SupervisedFeatureScorer:
    """Streaming per-feature one-way ANOVA F-statistic over labelled examples.

    Accumulates per-class sufficient statistics batch by batch, mirroring the
    per-feature accumulators in ``sae_lens.evals.get_sparsity_and_variance_metrics``
    but scoring features against example labels instead of counting active
    tokens. Useful when pooled activations arrive in evaluation batches.
    """

    def __init__(self) -> None:
        """Create an empty scorer; the feature count is set by the first batch."""
        self.num_features: int | None = None
        # Class labels key the accumulators and keep the numeric type the caller
        # used, which is an int for integer labels and a float otherwise.
        self._counts: dict[int | float, int] = {}
        self._sums: dict[int | float, torch.Tensor] = {}
        self._sum_squares: dict[int | float, torch.Tensor] = {}

    def add_batch(self, pooled_acts: torch.Tensor, labels: torch.Tensor) -> None:
        """Add a batch of pooled activations and their labels.

        Args:
            pooled_acts: (batch_size, num_features) pooled feature activations.
            labels: (batch_size,) labels; each distinct value is a class.
        """
        if pooled_acts.ndim != 2:
            raise ValueError(
                "pooled_acts must have shape (batch_size, num_features), got "
                f"{tuple(pooled_acts.shape)}."
            )
        if labels.ndim != 1 or labels.shape[0] != pooled_acts.shape[0]:
            raise ValueError(
                f"labels shape {tuple(labels.shape)} does not match batch size "
                f"{pooled_acts.shape[0]}."
            )
        if self.num_features is None:
            self.num_features = pooled_acts.shape[1]
        elif pooled_acts.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {pooled_acts.shape[1]}."
            )

        acts = pooled_acts.to(device="cpu", dtype=torch.float64)
        labels = labels.to("cpu")
        for class_label in torch.unique(labels).tolist():
            class_acts = acts[labels == class_label]
            if class_label not in self._counts:
                self._counts[class_label] = 0
                self._sums[class_label] = torch.zeros_like(acts[0])
                self._sum_squares[class_label] = torch.zeros_like(acts[0])
            self._counts[class_label] += class_acts.shape[0]
            self._sums[class_label] += class_acts.sum(dim=0)
            self._sum_squares[class_label] += class_acts.pow(2).sum(dim=0)

    def f_statistics(self) -> torch.Tensor:
        """Return the per-feature one-way ANOVA F-statistic, shape (num_features,).

        For each feature the statistic is the between-class mean square over the
        within-class mean square, so it is scale-free and invariant to per-feature
        constant offsets. A feature that separates the classes with no
        within-class variance scores ``inf``; a feature with no class signal
        scores 0.
        """
        num_classes = len(self._counts)
        if num_classes < 2:
            raise ValueError(
                "Need at least two distinct labels to score features, got "
                f"{num_classes}."
            )
        keys = sorted(self._counts)
        counts = torch.tensor([self._counts[key] for key in keys], dtype=torch.float64)
        sums = torch.stack([self._sums[key] for key in keys])
        sum_squares = torch.stack([self._sum_squares[key] for key in keys])
        num_examples = int(counts.sum())
        if num_examples <= num_classes:
            raise ValueError(
                "Need more examples than classes to estimate within-class "
                f"variance, got {num_examples} examples over {num_classes} classes."
            )

        class_means = sums / counts.unsqueeze(1)
        overall_mean = sums.sum(dim=0) / num_examples
        sum_squares_between = (
            counts.unsqueeze(1) * (class_means - overall_mean).pow(2)
        ).sum(dim=0)
        sum_squares_within = (
            sum_squares - sums.pow(2) / counts.unsqueeze(1)
        ).sum(dim=0).clamp_min(0.0)

        mean_square_between = sum_squares_between / (num_classes - 1)
        mean_square_within = sum_squares_within / (num_examples - num_classes)
        ratio = mean_square_between / mean_square_within.clamp_min(
            torch.finfo(mean_square_within.dtype).tiny
        )
        return torch.where(
            mean_square_within > 0,
            ratio,
            torch.where(
                sum_squares_between > 0,
                torch.full_like(ratio, float("inf")),
                torch.zeros_like(ratio),
            ),
        )

    def top_features(self, k: int) -> torch.Tensor:
        """Return the indices of the k highest-scoring features, best first.

        Ties are broken by ascending feature index, so the result is
        deterministic. k is capped at the number of features.
        """
        if self.num_features is None:
            raise ValueError("No batches added yet; call add_batch first.")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}.")
        order = torch.argsort(self.f_statistics(), descending=True, stable=True)
        return order[: min(k, self.num_features)]


def select_supervised_features(
    feature_acts: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool activations over context and keep the top-k label-discriminative features.

    Args:
        feature_acts: Token-level activations of shape (batch_size, ctx, num_features)
            or already-pooled activations of shape (batch_size, num_features).
        labels: (batch_size,) labels; each distinct value is a class.
        k: Number of features to keep.
        mask: Optional (batch_size, ctx) boolean mask of positions to pool.

    Returns:
        A tuple of (feature_indices, f_statistics), where feature_indices has
        shape (k,) and f_statistics has shape (num_features,).
    """
    scorer = SupervisedFeatureScorer()
    scorer.add_batch(pool_over_context(feature_acts, mask), labels)
    return scorer.top_features(k), scorer.f_statistics()
