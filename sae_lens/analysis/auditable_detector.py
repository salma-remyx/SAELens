"""
Auditable classifiers that predict from named SAE features.

Implements the auditable-detector pipeline of "Building Auditable AI Text
Detectors with Sparse Autoencoders" (Dugan et al.,
https://arxiv.org/abs/2610.30006): document-level SAE feature activations are
labeled with natural-language descriptions, only descriptions that pass a
global fidelity threshold are kept, and an L1-regularized logistic regression
is fit on the surviving features so that every prediction decomposes into
contributions from named features.

SAELens ships no auto-interpretation model, so two auxiliary components of the
paper's pipeline are substituted with parameter-free equivalents:

- descriptions are supplied by the caller (from Neuronpedia or the caller's own
  labeling run) instead of being generated from top-activating documents;
- the LLM judge that scores each description is replaced by a lexicon-overlap
  judge, which predicts how strongly a document activates a feature from the
  overlap between the description's content words and the document text.

The fidelity score itself keeps the paper's definition: the Spearman
correlation between predicted and true quantile assignments of the feature's
document-level (max-pooled) activations.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from sae_lens.saes.sae import SAE

Judge = Callable[[str, Sequence[str]], torch.Tensor]
"""Scores each document's predicted activation of a described feature."""

STOP_WORDS = frozenset(
    """
    a about all also an and any are as at be been but by can could does for from
    has have how in into is it its many may more most much no not of on or our
    some such than that the their them then these they this those to very was
    were what when where which who whose why will with would document documents
    feature features phrase phrases text texts use used uses using word words
    mention mentions mentioned containing contains contain
    """.split()
)


def document_feature_activations(
    sae: SAE[Any], activations: torch.Tensor
) -> torch.Tensor:
    """
    Encode activations with an SAE and max-pool token features per document.

    Args:
        sae: SAE used to encode the residual stream.
        activations: Tensor of shape (num_documents, seq_len, d_in) or
            (num_documents, d_in).

    Returns:
        Tensor of shape (num_documents, d_sae) holding each feature's maximum
        activation across the document's tokens.
    """
    feature_acts = sae.encode(activations)
    if feature_acts.dim() == 3:
        feature_acts = feature_acts.max(dim=1).values
    return feature_acts


def quantile_bin_assignments(activations: torch.Tensor, num_bins: int) -> torch.Tensor:
    """
    Assign documents to quantile bins of a single feature's activations.

    Inactive documents (activation <= 0) share bin 0, matching the paper's
    extra bin for documents that never fire the feature. Remaining documents
    are bucketed into ``num_bins`` bins by quantile.

    Args:
        activations: Tensor of shape (num_documents,) of document-level
            activations of one feature.
        num_bins: Number of quantile bins over the active documents.

    Returns:
        Integer tensor of shape (num_documents,) with bin indices in
        [0, num_bins].
    """
    bins = torch.zeros_like(activations, dtype=torch.long)
    active = activations[activations > 0]
    if active.numel() == 0:
        return bins
    edges = torch.quantile(
        active,
        torch.linspace(0.0, 1.0, num_bins + 1, device=activations.device),
    )
    # right=False keeps documents tied with a bin edge in the lower bin, so
    # equal activations are never split across bins.
    active_bins = torch.bucketize(active, edges[1:-1], right=False) + 1
    bins[activations > 0] = active_bins
    return bins


def spearman_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the Spearman rank correlation between two vectors.

    Ties are given the average rank, so constant inputs are handled by
    returning zero rather than an undefined value.
    """
    if a.numel() < 2:
        return torch.zeros((), dtype=torch.float32, device=a.device)
    ranks_a = _average_ranks(a.float())
    ranks_b = _average_ranks(b.float())
    centered_ranks_a = ranks_a - ranks_a.mean()
    centered_ranks_b = ranks_b - ranks_b.mean()
    denominator = centered_ranks_a.norm() * centered_ranks_b.norm()
    if denominator == 0:
        return torch.zeros((), dtype=torch.float32, device=a.device)
    return (centered_ranks_a @ centered_ranks_b) / denominator


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Rank values from 1 to n, giving tied values their average rank."""
    order = values.argsort()
    sorted_values = values[order]
    positions = torch.arange(
        1, values.numel() + 1, dtype=values.dtype, device=values.device
    )
    group_starts = torch.cat(
        [
            torch.ones(1, dtype=torch.bool, device=values.device),
            sorted_values[1:] != sorted_values[:-1],
        ]
    )
    group_ids = torch.cumsum(group_starts, 0) - 1
    num_groups = int(group_ids[-1]) + 1
    sums = torch.zeros(num_groups, dtype=values.dtype, device=values.device)
    sizes = torch.zeros(num_groups, dtype=values.dtype, device=values.device)
    sums.index_add_(0, group_ids, positions)
    sizes.index_add_(0, group_ids, torch.ones_like(positions))
    ranks = torch.empty_like(positions)
    ranks[order] = sums[group_ids] / sizes[group_ids]
    return ranks


class LexiconOverlapJudge:
    """
    Parameter-free stand-in for the paper's LLM judge.

    Predicts a document's activation of a described feature by counting how
    many of the description's content words occur in the document. Any callable
    with the same signature (returning one score per document) can be passed to
    :class:`AuditableDetector` in its place, e.g. an LLM prompted to rate each
    document against the description.
    """

    def __call__(self, description: str, docs: Sequence[str]) -> torch.Tensor:
        content_terms = {
            term for term in _content_terms(description) if term not in STOP_WORDS
        }
        scores = torch.zeros(len(docs), dtype=torch.float32)
        if not content_terms:
            return scores
        for index, doc in enumerate(docs):
            doc_terms = set(_content_terms(doc))
            scores[index] = len(content_terms & doc_terms)
        return scores


def _content_terms(text: str) -> list[str]:
    terms = re.findall(r"[a-z0-9]+", text.lower())
    return [term for term in terms if len(term) >= 3]


def description_fidelity(
    description: str,
    feature_activations: torch.Tensor,
    docs: Sequence[str],
    judge: Judge | None = None,
    num_bins: int = 10,
) -> float:
    """
    Score how well a feature's description predicts where the feature fires.

    Documents are assigned to quantile bins of their activation of the feature,
    and the score is the Spearman correlation between the judge's predicted
    ranking and the true bin assignment. Binning the judge's scores into the
    same quantiles is a monotone transform, so this equals the paper's
    correlation between predicted and true quantile assignments.

    Args:
        description: Natural-language description of the feature.
        feature_activations: Tensor of shape (num_documents,) with the
            feature's document-level activations.
        docs: One string per document, in the same order.
        judge: Scores each document from the description.
        num_bins: Number of quantile bins over active documents.

    Returns:
        Fidelity score in [-1, 1].
    """
    if judge is None:
        judge = LexiconOverlapJudge()
    predicted = judge(description, docs)
    true_bins = quantile_bin_assignments(feature_activations, num_bins)
    return spearman_correlation(predicted, true_bins.float()).item()


@dataclass
class FeatureReport:
    """Fidelity audit trail for one described feature."""

    feature: int
    description: str
    fidelity: float
    retained: bool


@dataclass
class FeatureContribution:
    """A single named feature's contribution to one prediction."""

    feature: int
    description: str
    contribution: float
    """Change in the prediction's logit when the feature is zero-ablated."""


class AuditableDetector:
    """
    Auditable classifier over the SAE features that survive fidelity filtering.

    Fits an L1-regularized logistic regression on the retained features only,
    so a prediction is explained by listing the largest per-feature
    contributions alongside their descriptions.
    """

    def __init__(
        self,
        fidelity_threshold: float = 0.7,
        num_bins: int = 10,
        l1_coefficient: float = 0.01,
        num_steps: int = 500,
        judge: Judge | None = None,
    ) -> None:
        """
        Args:
            fidelity_threshold: Minimum fidelity score for a feature to be kept.
            num_bins: Quantile bins used when scoring description fidelity.
            l1_coefficient: L1 penalty on the logistic head's weights.
            num_steps: Proximal-gradient steps used to fit the head.
            judge: Scores each document against a feature description.
        """
        self.fidelity_threshold = fidelity_threshold
        self.num_bins = num_bins
        self.l1_coefficient = l1_coefficient
        self.num_steps = num_steps
        self.judge = judge if judge is not None else LexiconOverlapJudge()
        self.descriptions: dict[int, str] = {}
        self.fidelities: dict[int, float] = {}
        self.retained_features: list[int] = []
        self.weights = torch.zeros(0)
        self.bias = torch.zeros(())

    def fit(
        self,
        docs: Sequence[str],
        feature_acts: torch.Tensor,
        labels: torch.Tensor,
        descriptions: dict[int, str],
    ) -> "AuditableDetector":
        """
        Filter described features by fidelity, then fit a sparse logistic head.

        Args:
            docs: One string per document.
            feature_acts: Tensor of shape (num_documents, d_sae) of
                document-level feature activations.
            labels: Binary labels of shape (num_documents,).
            descriptions: Mapping from feature index to its description.
                Features without a description are never retained.
        """
        if feature_acts.dim() != 2:
            raise ValueError(
                "feature_acts must be 2D (num_documents, d_sae), got "
                f"{tuple(feature_acts.shape)}"
            )
        if len(docs) != feature_acts.shape[0]:
            raise ValueError(
                f"got {len(docs)} docs but {feature_acts.shape[0]} rows of activations"
            )
        if labels.shape != feature_acts.shape[:1]:
            raise ValueError(
                f"labels must have shape {tuple(feature_acts.shape[:1])}, "
                f"got {tuple(labels.shape)}"
            )
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("labels must be binary (0 or 1)")
        for feature in descriptions:
            if not 0 <= feature < feature_acts.shape[1]:
                raise ValueError(
                    f"description references feature {feature}, but the SAE has "
                    f"{feature_acts.shape[1]} features"
                )

        self.descriptions = dict(descriptions)
        self.fidelities = {
            feature: description_fidelity(
                description,
                feature_acts[:, feature],
                docs,
                judge=self.judge,
                num_bins=self.num_bins,
            )
            for feature, description in self.descriptions.items()
        }
        self.retained_features = [
            feature
            for feature in sorted(self.fidelities)
            if self.fidelities[feature] >= self.fidelity_threshold
        ]
        if not self.retained_features:
            raise ValueError(
                "no feature description reached the fidelity threshold "
                f"{self.fidelity_threshold}"
            )
        self.weights, self.bias = _fit_sparse_logistic_head(
            feature_acts[:, self.retained_features].float(),
            labels.float(),
            self.l1_coefficient,
            self.num_steps,
        )
        return self

    def feature_report(self) -> list[FeatureReport]:
        """Return the fidelity and retention status of every described feature."""
        return [
            FeatureReport(
                feature=feature,
                description=self.descriptions[feature],
                fidelity=self.fidelities[feature],
                retained=feature in self.retained_features,
            )
            for feature in sorted(self.descriptions)
        ]

    def decision_function(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Return the logit score of each document."""
        return self._masked(feature_acts) @ self.weights + self.bias

    def predict(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Return the predicted binary label of each document."""
        return (self.decision_function(feature_acts) > 0).long()

    def explain(
        self, feature_acts: torch.Tensor, top_k: int = 5
    ) -> list[list[FeatureContribution]]:
        """
        Explain each prediction by its largest named feature contributions.

        For this linear head a feature's contribution equals the change in the
        prediction's logit when that feature is zero-ablated.

        Args:
            feature_acts: Tensor of shape (num_documents, d_sae).
            top_k: Number of features to report per document.

        Returns:
            One list of at most ``top_k`` contributions per document, ordered
            by absolute contribution.
        """
        masked = self._masked(feature_acts)
        contributions = masked * self.weights
        explanations: list[list[FeatureContribution]] = []
        for row in contributions:
            order = row.abs().argsort(descending=True)[:top_k]
            explanations.append(
                [
                    FeatureContribution(
                        feature=self.retained_features[index],
                        description=self.descriptions[self.retained_features[index]],
                        contribution=row[index].item(),
                    )
                    for index in order
                ]
            )
        return explanations

    def _masked(self, feature_acts: torch.Tensor) -> torch.Tensor:
        if feature_acts.dim() != 2:
            raise ValueError(
                "feature_acts must be 2D (num_documents, d_sae), got "
                f"{tuple(feature_acts.shape)}"
            )
        return feature_acts[:, self.retained_features].float()


def _fit_sparse_logistic_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    l1_coefficient: float,
    num_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit an L1-regularized logistic regression by proximal gradient descent."""
    num_docs, num_features = features.shape
    weights = torch.zeros(num_features)
    bias = torch.zeros(())
    # Step size 1 / L, with L the Lipschitz constant of the mean BCE gradient,
    # keeps the updates stable whatever the scale of the activations.
    spectral_norm = torch.linalg.matrix_norm(features, ord=2)
    step_size = 1.0 / torch.clamp(spectral_norm.square() / num_docs, min=1e-8)
    for _ in range(num_steps):
        error = torch.sigmoid(features @ weights + bias) - labels
        weights = _soft_threshold(
            weights - step_size * (features.T @ error) / num_docs,
            step_size * l1_coefficient,
        )
        bias = bias - step_size * error.mean()
    return weights, bias


def _soft_threshold(values: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.sign(values) * torch.relu(values.abs() - threshold)
