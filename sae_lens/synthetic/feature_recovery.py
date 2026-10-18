"""
Ground-truth-conditioned feature recovery diagnostics for SAEs on synthetic data.

Aggregate metrics such as the mean correlation coefficient or feature uniqueness
only show the average shadow that feature absorption casts over recovery: they
cannot say which ground-truth features were recovered, which were merged into a
shared atom, or how nested features fare relative to their parents. This module
adds those diagnostics, following the phase diagram study of the MAIS-O43 open
problem, "A Dominant Diffuse Phase in the Sparse Autoencoder Phase Diagram"
(https://arxiv.org/abs/2609.10299):

- per-feature best cosine similarity against the SAE decoder
- recovery fraction under a strict cosine criterion (0.95 by default)
- hierarchy-conditioned per-edge alignment gaps and merge detection
- a phase label (recovered / merged / diffuse) that contrasts reconstruction
  quality with ground-truth alignment

The study's central observation is that reconstruction quality is a misleading
proxy for recovery: in the diffuse phase reconstruction is nearly perfect while
learned atoms remain far from every true feature and codes are much denser than
the ground truth. Use `eval_sae_recovery_on_synthetic_data` to get both sides of
that contrast in a single result.
"""

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch

from sae_lens.saes.sae import SAE
from sae_lens.synthetic.activation_generator import ActivationGenerator
from sae_lens.synthetic.evals import SyntheticDataEvalResult, eval_sae_on_synthetic_data
from sae_lens.synthetic.feature_dictionary import FeatureDictionary
from sae_lens.synthetic.hierarchy.node import HierarchyNode
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.util import cosine_similarities

DEFAULT_RECOVERY_THRESHOLD = 0.95
"""Cosine criterion for declaring a ground-truth feature recovered (MAIS-O43)."""

MERGE_COSINE_FLOOR = 1 / math.sqrt(2)
"""Minimum best cosine for a shared atom to count as a merged parent-child pair.

A balanced mixture of two orthogonal nested features has cosine 1/sqrt(2) with
each of them, which is the merged atom the exact sparse-coding optimum is known
to prefer in the two-feature case.
"""


def _best_matches(
    sae_decoder: torch.Tensor, gt_features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per ground-truth feature (best |cosine|, best-matching atom index)."""
    if sae_decoder.shape[0] == 0:
        zero = torch.zeros(gt_features.shape[0])
        return zero, zero.long()
    cos_sim = cosine_similarities(gt_features, sae_decoder).abs()
    return cos_sim.amax(dim=1), cos_sim.argmax(dim=1)


def best_feature_cosines(
    sae_decoder: torch.Tensor,
    gt_features: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the best |cosine| between each ground-truth feature and any SAE atom.

    This is the per-feature alignment the MAIS-O43 recovery criterion thresholds:
    a feature counts as recovered when some decoder atom aligns with it closely,
    regardless of what the remaining atoms do.

    Args:
        sae_decoder: SAE decoder weights of shape (num_sae_latents, hidden_dim)
        gt_features: Ground-truth feature vectors of shape (num_gt_features, hidden_dim)

    Returns:
        Tensor of shape (num_gt_features,) with the best absolute cosine similarity
        of each ground-truth feature against the decoder atoms
    """
    return _best_matches(sae_decoder, gt_features)[0]


def hierarchy_edges(
    roots: Sequence[HierarchyNode] | HierarchyNode,
) -> list[tuple[int, int]]:
    """
    Collect (parent, child) feature-index edges from hierarchy trees.

    Organizational nodes without a feature_index do not produce edges but their
    subtrees are still traversed.

    Args:
        roots: One or more root HierarchyNode objects

    Returns:
        List of (parent_index, child_index) pairs in breadth-first order
    """
    if isinstance(roots, HierarchyNode):
        roots = [roots]

    edges: list[tuple[int, int]] = []
    queue: deque[HierarchyNode] = deque(roots)
    while queue:
        node = queue.popleft()
        for child in node.children:
            if node.feature_index is not None and child.feature_index is not None:
                edges.append((node.feature_index, child.feature_index))
            queue.append(child)
    return edges


@dataclass
class RecoveryMetrics:
    """Ground-truth-conditioned recovery diagnostics for an SAE decoder."""

    median_best_cosine: float
    """Median over ground-truth features of the best |cosine| to any decoder atom"""

    mean_best_cosine: float
    """Mean over ground-truth features of the best |cosine| to any decoder atom"""

    recovered_fraction: float
    """Fraction of ground-truth features with best cosine >= recovery_threshold"""

    recovery_threshold: float
    """Cosine criterion used to declare a ground-truth feature recovered"""

    num_hierarchy_edges: int
    """Number of parent-child edges in the evaluated hierarchy (0 if none given)"""

    merged_edge_fraction: float
    """Fraction of hierarchy edges whose parent and child share their best atom"""

    mean_edge_cosine_gap: float
    """Mean over edges of (parent best cosine - child best cosine); positive when
    nested children are worse aligned than their parents, the signature of absorption"""

    def to_log_dict(self, prefix: str = "") -> dict[str, float | int]:
        """Convert to a flat dictionary for logging (e.g. to wandb)."""
        return {f"{prefix}{f.name}": getattr(self, f.name) for f in fields(self)}


@torch.no_grad()
def compute_recovery_metrics(
    sae_decoder: torch.Tensor,
    gt_features: torch.Tensor,
    hierarchy_roots: Sequence[HierarchyNode] | HierarchyNode | None = None,
    recovery_threshold: float = DEFAULT_RECOVERY_THRESHOLD,
) -> RecoveryMetrics:
    """
    Compute ground-truth-conditioned recovery metrics for an SAE decoder.

    Args:
        sae_decoder: SAE decoder weights of shape (num_sae_latents, hidden_dim)
        gt_features: Ground-truth feature vectors of shape (num_gt_features, hidden_dim)
        hierarchy_roots: Optional hierarchy trees used to generate the activations.
            When given, per-edge merge detection and alignment gaps are computed.
        recovery_threshold: Best cosine above which a ground-truth feature counts
            as recovered

    Returns:
        RecoveryMetrics with per-feature alignment aggregates and, when a
        hierarchy is given, per-edge merge and gap statistics
    """
    best_cos, best_atoms = _best_matches(sae_decoder, gt_features)

    edges = hierarchy_edges(hierarchy_roots) if hierarchy_roots is not None else []
    merged_edges = 0
    edge_gaps: list[float] = []
    for parent_index, child_index in edges:
        parent_cos = best_cos[parent_index].item()
        child_cos = best_cos[child_index].item()
        edge_gaps.append(parent_cos - child_cos)
        shares_best_atom = best_atoms[parent_index] == best_atoms[child_index]
        if shares_best_atom and max(parent_cos, child_cos) >= MERGE_COSINE_FLOOR:
            merged_edges += 1

    return RecoveryMetrics(
        median_best_cosine=best_cos.median().item(),
        mean_best_cosine=best_cos.mean().item(),
        recovered_fraction=(best_cos >= recovery_threshold).float().mean().item(),
        recovery_threshold=recovery_threshold,
        num_hierarchy_edges=len(edges),
        merged_edge_fraction=merged_edges / len(edges) if edges else 0.0,
        mean_edge_cosine_gap=sum(edge_gaps) / len(edge_gaps) if edge_gaps else 0.0,
    )


@dataclass
class SyntheticRecoveryEvalResult:
    """Synthetic-data eval result extended with ground-truth recovery diagnostics."""

    aggregate: SyntheticDataEvalResult
    """Aggregate metrics from eval_sae_on_synthetic_data"""

    recovery: RecoveryMetrics
    """Ground-truth-conditioned recovery diagnostics"""

    @property
    def density_ratio(self) -> float:
        """Ratio of SAE L0 to ground-truth L0; inf when the ground truth never fires.

        Learned codes in the diffuse phase are much denser than the ground truth,
        so this ratio is an order of magnitude above 1 in the study's grid.
        """
        if self.aggregate.true_l0 == 0:
            return float("inf")
        return self.aggregate.sae_l0 / self.aggregate.true_l0

    @property
    def phase(self) -> str:
        """Phase label summarizing how the SAE relates to the ground truth.

        Classification, in precedence order:

        - "recovered": every ground-truth feature meets the cosine criterion
        - "merged": some hierarchy edge's parent and child share their best atom
          at mixture-level cosine, i.e. nested features were absorbed together
        - "diffuse": otherwise. Reconstruction may still be near-perfect here,
          which is exactly the failure mode aggregate reconstruction metrics
          cannot see
        """
        if self.recovery.recovered_fraction >= 1.0:
            return "recovered"
        if self.recovery.merged_edge_fraction > 0.0:
            return "merged"
        return "diffuse"

    def to_log_dict(self, prefix: str = "") -> dict[str, float | int]:
        """Convert to a flat dictionary combining aggregate and recovery metrics."""
        result = self.aggregate.to_log_dict(prefix=prefix)
        result.update(self.recovery.to_log_dict(prefix=f"{prefix}recovery/"))
        return result


@torch.no_grad()
def eval_sae_recovery_on_synthetic_data(
    sae: SAE[Any],
    feature_dict: FeatureDictionary,
    activations_generator: ActivationGenerator,
    hierarchy_roots: Sequence[HierarchyNode] | HierarchyNode | None = None,
    num_samples: int = 100_000,
    batch_size: int | None = None,
    activation_scaler: ActivationScaler | None = None,
    recovery_threshold: float = DEFAULT_RECOVERY_THRESHOLD,
) -> SyntheticRecoveryEvalResult:
    """
    Evaluate an SAE on synthetic data with ground-truth recovery diagnostics.

    Runs `eval_sae_on_synthetic_data` for the aggregate metrics (reconstruction,
    sparsity, MCC, ...) and adds the ground-truth-conditioned recovery metrics
    computed from the decoder, so that reconstruction quality and feature
    recovery can be contrasted directly.

    Args:
        sae: The SAE to evaluate
        feature_dict: The feature dictionary used to generate activations
        activations_generator: Generator that produces feature activations
        hierarchy_roots: Optional hierarchy trees the generator was built with.
            Enables per-edge merge detection and alignment-gap metrics.
        num_samples: Number of samples to use for evaluation
        batch_size: Batch size for processing. If None, processes all samples at once.
        activation_scaler: Optional scaler to apply to activations before encoding.
            If None, no scaling is applied.
        recovery_threshold: Best cosine above which a ground-truth feature counts
            as recovered

    Returns:
        SyntheticRecoveryEvalResult containing the aggregate metrics and the
        recovery diagnostics
    """
    aggregate = eval_sae_on_synthetic_data(
        sae=sae,
        feature_dict=feature_dict,
        activations_generator=activations_generator,
        num_samples=num_samples,
        batch_size=batch_size,
        activation_scaler=activation_scaler,
    )

    recovery = compute_recovery_metrics(
        sae_decoder=sae.W_dec,
        gt_features=feature_dict.feature_vectors,
        hierarchy_roots=hierarchy_roots,
        recovery_threshold=recovery_threshold,
    )

    return SyntheticRecoveryEvalResult(aggregate=aggregate, recovery=recovery)
