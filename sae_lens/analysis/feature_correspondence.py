"""Feature-level correspondence analysis between two SAEs over a shared corpus.

For each feature of a reference SAE, find its best match among the features of a
comparison SAE by Jaccard similarity of the token positions each feature activates
on, then extract the divergent features whose best match falls below a threshold.

Adapted from "Comparing Latent Concept Formation in State Space Models and
Transformers via Sparse Autoencoders" (arXiv:2609.24440), which uses this
analysis to compare SAEs trained on Mamba-130m and Pythia-70m over a shared
corpus and finds that representational divergence between the architectures is
confined to a microscopic fraction of features.
"""

from dataclasses import dataclass
from typing import Any

import torch

from sae_lens.saes.sae import SAE


@dataclass
class CorrespondenceReport:
    """
    Best-match Jaccard correspondence of one SAE's features against another's.

    Args:
        best_match_similarity (torch.Tensor): (n_features,) the highest Jaccard
            similarity of each reference feature against every comparison feature.
            Features that never activate on the corpus get a similarity of zero.
        best_match_indices (torch.Tensor): (n_features,) index of the comparison
            feature achieving that similarity.
        active_mask (torch.Tensor): (n_features,) whether each reference feature
            activated on at least one token of the corpus. Dead features carry no
            evidence about correspondence and are excluded from divergence stats.
        divergence_threshold (float): an active feature whose best-match
            similarity falls strictly below this threshold is divergent.
    """

    best_match_similarity: torch.Tensor
    best_match_indices: torch.Tensor
    active_mask: torch.Tensor
    divergence_threshold: float

    @property
    def n_active_features(self) -> int:
        """Number of reference features that activated at least once."""
        return int(self.active_mask.sum().item())

    @property
    def divergent_features(self) -> torch.Tensor:
        """Indices of active features whose best match falls below the threshold."""
        divergent = self.active_mask & (
            self.best_match_similarity < self.divergence_threshold
        )
        return torch.nonzero(divergent).flatten()

    @property
    def divergent_fraction(self) -> float:
        """Fraction of active features whose best match falls below the threshold."""
        if self.n_active_features == 0:
            return 0.0
        return len(self.divergent_features) / self.n_active_features


def best_match_jaccard(
    feature_acts_a: torch.Tensor,
    feature_acts_b: torch.Tensor,
    *,
    divergence_threshold: float = 0.5,
    token_batch_size: int = 4096,
    feature_batch_size: int = 1024,
) -> CorrespondenceReport:
    """
    Compute the best-match Jaccard correspondence between two feature activation
    matrices collected over the same token positions.

    Each feature is summarized by the set of token positions it activates on
    (nonzero encode output). For every reference feature the maximum Jaccard
    similarity over all comparison features is reported, giving the correspondence
    distribution used to ask whether two SAEs learned matching features.

    Args:
        feature_acts_a (torch.Tensor): (num_tokens, n_features_a) encode output of
            the reference SAE. Leading dimensions are flattened.
        feature_acts_b (torch.Tensor): (num_tokens, n_features_b) encode output of
            the comparison SAE over the same token positions.
        divergence_threshold (float): an active reference feature whose best-match
            similarity falls strictly below this threshold is divergent.
        token_batch_size (int): number of token positions per intersection matmul,
            bounding peak memory on large corpora.
        feature_batch_size (int): number of reference features whose similarity
            rows are materialized at once.

    Returns:
        CorrespondenceReport: per-feature best matches and the divergent subset.
    """
    mask_a = _flatten_activations(feature_acts_a) != 0
    mask_b = _flatten_activations(feature_acts_b) != 0
    if mask_a.shape[0] != mask_b.shape[0]:
        raise ValueError(
            "Activation matrices must cover the same token positions, got "
            f"{mask_a.shape[0]} and {mask_b.shape[0]}."
        )

    n_features_a = mask_a.shape[1]
    sizes_b = mask_b.sum(dim=0)
    best_similarity = torch.empty(
        n_features_a, dtype=torch.float32, device=mask_a.device
    )
    best_indices = torch.empty(n_features_a, dtype=torch.long, device=mask_a.device)

    for start in range(0, n_features_a, feature_batch_size):
        end = min(start + feature_batch_size, n_features_a)
        mask_a_chunk = mask_a[:, start:end]
        similarities = _jaccard_rows(
            mask_a_chunk,
            mask_b,
            mask_a_chunk.sum(dim=0),
            sizes_b,
            token_batch_size,
        )
        chunk_best, chunk_indices = similarities.max(dim=1)
        best_similarity[start:end] = chunk_best
        best_indices[start:end] = chunk_indices

    return CorrespondenceReport(
        best_match_similarity=best_similarity,
        best_match_indices=best_indices,
        active_mask=mask_a.any(dim=0),
        divergence_threshold=divergence_threshold,
    )


def feature_correspondence(
    sae_a: SAE[Any],
    sae_b: SAE[Any],
    activations_a: torch.Tensor,
    activations_b: torch.Tensor | None = None,
    *,
    divergence_threshold: float = 0.5,
    token_batch_size: int = 4096,
    feature_batch_size: int = 1024,
) -> CorrespondenceReport:
    """
    Compare the features of two SAEs over a shared corpus of activations.

    Both SAEs encode their own view of the same token positions and the reference
    SAE's features are matched against the comparison SAE's by best-match Jaccard
    similarity of their active token position sets. For a cross-architecture
    comparison, each activation matrix is collected from its model over the same
    tokens, so the two matrices may have different input dimensions.

    Args:
        sae_a (SAE): reference SAE whose features are matched.
        sae_b (SAE): comparison SAE providing the candidate matches.
        activations_a (torch.Tensor): (..., d_in) activations encoded by sae_a.
        activations_b (torch.Tensor | None): activations encoded by sae_b, covering
            the same number of token positions. Defaults to activations_a for SAEs
            sharing an input dimension.
        divergence_threshold (float): an active reference feature whose best-match
            similarity falls strictly below this threshold is divergent.
        token_batch_size (int): number of token positions per intersection matmul.
        feature_batch_size (int): number of reference features whose similarity
            rows are materialized at once.

    Returns:
        CorrespondenceReport: per-feature best matches and the divergent subset.
    """
    if activations_b is None:
        activations_b = activations_a
    with torch.no_grad():
        feature_acts_a = sae_a.encode(activations_a)
        feature_acts_b = sae_b.encode(activations_b)
    return best_match_jaccard(
        feature_acts_a,
        feature_acts_b,
        divergence_threshold=divergence_threshold,
        token_batch_size=token_batch_size,
        feature_batch_size=feature_batch_size,
    )


def _flatten_activations(feature_acts: torch.Tensor) -> torch.Tensor:
    """Reshape encode output of shape (..., n_features) to (num_tokens, n_features)."""
    if feature_acts.dim() < 2:
        raise ValueError(
            "Feature activations must have shape (..., n_features), got "
            f"shape {tuple(feature_acts.shape)}."
        )
    return feature_acts.reshape(-1, feature_acts.shape[-1])


def _jaccard_rows(
    mask_a_chunk: torch.Tensor,
    mask_b: torch.Tensor,
    sizes_a_chunk: torch.Tensor,
    sizes_b: torch.Tensor,
    token_batch_size: int,
) -> torch.Tensor:
    """
    Jaccard similarity of each reference feature column in the chunk against every
    comparison feature column, computed from chunked intersection counts.
    """
    intersections = torch.zeros(
        mask_a_chunk.shape[1],
        mask_b.shape[1],
        dtype=torch.int64,
        device=mask_a_chunk.device,
    )
    n_tokens = mask_a_chunk.shape[0]
    for start in range(0, n_tokens, token_batch_size):
        tokens_a = mask_a_chunk[start : start + token_batch_size].to(torch.float32)
        tokens_b = mask_b[start : start + token_batch_size].to(torch.float32)
        intersections += (tokens_a.T @ tokens_b).to(torch.int64)

    sizes_a = sizes_a_chunk.unsqueeze(1)
    sizes_b_column = sizes_b.unsqueeze(0)
    unions = sizes_a + sizes_b_column - intersections
    # Where both sets are empty the intersection is zero too, so clamping the
    # union to one reports a similarity of zero instead of dividing by zero.
    return intersections.float() / unions.clamp(min=1).float()
