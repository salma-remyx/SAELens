"""Measuring and transferring bilingual SAE features.

Adapted from "How a Bilingual LM Becomes Bilingual: Tracing Internal
Representations with Sparse Autoencoders" (Inaba, Oseki, Heinzerling and
Takagi, https://arxiv.org/abs/2503.06394). The paper finds that bilingual
language models first learn each language separately and only later form
bilingual feature alignments, that those alignments live mostly in mid layers,
and that adding the bilingual part of a fully trained model's SAE
decomposition into a mid-training model improves it.

This module provides the two pieces of that analysis:

- Measurement: `language_selectivity` labels each feature as language-specific
  or mixed (bilingual) from activations collected on monolingual corpora, and
  `compute_monosemanticity` scores how semantically coherent the tokens
  activating each feature are. `bilingual_alignment` summarises both into the
  per-layer (or per-checkpoint) statistic the paper tracks, and
  `peak_alignment_layer` locates the layer where bilingual alignment peaks.
- Transfer: `transfer_decomposed_features` and `make_feature_injection_hook`
  implement the paper's intervention
  `x_recipient <- x_recipient + alpha * D(mask * E(x_donor))`, restricted to a
  chosen feature kind.

The paper's LLM-jp corpus, SAE training runs and COMET evaluation are not
reproduced here: callers supply their own activations, SAEs and token
embeddings.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from sae_lens.saes.sae import SAE

FeatureKind = Literal["language_a", "language_b", "mixed", "dead"]


@dataclass
class LanguageSelectivity:
    """Per-feature language statistics from two monolingual activation sets.

    Attributes:
        counts_a: (d_sae,) number of activating positions per feature in language_a.
        counts_b: (d_sae,) number of activating positions per feature in language_b.
        p_language_a: (d_sae,) probability a feature's activation came from language_a.
        labels: one FeatureKind per feature.
    """

    counts_a: torch.Tensor
    counts_b: torch.Tensor
    p_language_a: torch.Tensor
    labels: list[FeatureKind]

    @property
    def activation_counts(self) -> torch.Tensor:
        """(d_sae,) total activating positions per feature."""
        return self.counts_a + self.counts_b

    @property
    def live_mask(self) -> torch.Tensor:
        """(d_sae,) True for features that activated at least once."""
        return self.activation_counts > 0

    @property
    def mixed_proportion(self) -> float:
        """Fraction of live features that are mixed (bilingual)."""
        live = self.live_mask
        n_live = int(live.sum())
        if n_live == 0:
            return 0.0
        n_mixed = sum(label == "mixed" for label in self.labels)
        return n_mixed / n_live


def language_selectivity(
    activations_a: torch.Tensor,
    activations_b: torch.Tensor,
    threshold: float = 0.9,
    activation_fraction: float = 0.7,
) -> LanguageSelectivity:
    """Classify features by the language of the positions that activate them.

    Following the paper's token attribution rule, a position activates a
    feature when its activation is at least `activation_fraction` of that
    feature's maximum activation (computed per corpus, so maxima are never
    shared across languages). Features with p(language_a) above `threshold`
    are language_a features, features below `1 - threshold` are language_b
    features, the rest are mixed, and features that activate nowhere are dead.

    Args:
        activations_a: (n_positions_a, d_sae) feature activations from one monolingual corpus.
        activations_b: (n_positions_b, d_sae) feature activations from the other monolingual corpus.
        threshold: language selectivity threshold (the paper uses 0.9).
        activation_fraction: fraction of a feature's maximum activation that
            counts as firing (the paper uses 0.7).
    """

    def activating_counts(activations: torch.Tensor) -> torch.Tensor:
        maxima = activations.max(dim=0).values
        activating = (activations >= activation_fraction * maxima) & (maxima > 0)
        return activating.sum(dim=0)

    counts_a = activating_counts(activations_a)
    counts_b = activating_counts(activations_b)

    # Dead features get p = 0.5 so downstream tensor math stays finite; use
    # live_mask to exclude them.
    probabilities: list[float] = []
    labels: list[FeatureKind] = []
    for n_a, n_b in zip(counts_a.tolist(), counts_b.tolist(), strict=True):
        n_total = n_a + n_b
        if n_total == 0:
            probabilities.append(0.5)
            labels.append("dead")
            continue
        p_a = n_a / n_total
        probabilities.append(p_a)
        if p_a > threshold:
            labels.append("language_a")
        elif p_a < 1 - threshold:
            labels.append("language_b")
        else:
            labels.append("mixed")

    return LanguageSelectivity(
        counts_a,
        counts_b,
        torch.tensor(probabilities),
        labels,
    )


def feature_mask(
    selectivity: LanguageSelectivity,
    kind: Literal["language_a", "language_b", "mixed"] = "mixed",
    m: int | None = None,
) -> torch.Tensor:
    """Build the (d_sae,) boolean mask used to restrict a transfer to one feature kind.

    Features are ranked by how strongly they belong to `kind` (language_a by
    p(language_a), language_b by p(language_b), mixed by how balanced the two
    languages are), and the top `m` are kept. Leaving `m` as None keeps every
    feature of that kind.

    Args:
        selectivity: output of `language_selectivity`.
        kind: which features to select.
        m: number of features to keep.
    """
    if kind == "language_a":
        scores = selectivity.p_language_a
    elif kind == "language_b":
        scores = 1 - selectivity.p_language_a
    else:
        scores = torch.minimum(selectivity.p_language_a, 1 - selectivity.p_language_a)

    mask = torch.tensor([label == kind for label in selectivity.labels])
    if m is None or int(mask.sum()) <= int(m):
        return mask
    if m <= 0:
        return torch.zeros_like(mask)
    masked_scores = torch.where(mask, scores, torch.tensor(-torch.inf))
    top = torch.topk(masked_scores, k=int(m)).indices
    selected = torch.zeros_like(mask)
    selected[top] = True
    return selected


def collect_token_attributions(
    feature_acts: torch.Tensor,
    token_ids: torch.Tensor,
    activation_fraction: float = 0.7,
) -> list[dict[int, int]]:
    """Count which tokens activated each feature.

    A token activates a feature when the feature's activation at that position
    is at least `activation_fraction` of the feature's maximum activation over
    all positions, which is how the paper builds f(v|i).

    Args:
        feature_acts: (n_positions, d_sae) feature activations over an evaluation set.
        token_ids: (n_positions,) token id at each position.
        activation_fraction: fraction of a feature's maximum activation that counts as firing.
    """
    maxima = feature_acts.max(dim=0).values
    activating = (feature_acts >= activation_fraction * maxima) & (maxima > 0)

    attributions: list[dict[int, int]] = []
    for feature in range(feature_acts.shape[1]):
        counts: dict[int, int] = {}
        for token_id in token_ids[activating[:, feature]].tolist():
            counts[token_id] = counts.get(token_id, 0) + 1
        attributions.append(counts)
    return attributions


@dataclass
class Monosemanticity:
    """Per-feature token entropies and monosemanticity scores.

    Attributes:
        token_entropy: (d_sae,) entropy of each feature's token attribution distribution.
        semantic_entropy: (d_sae,) entropy over the semantic clusters of those tokens.
        score: (d_sae,) monosemanticity R_mono in [0, 1], higher is more coherent.
    """

    token_entropy: torch.Tensor
    semantic_entropy: torch.Tensor
    score: torch.Tensor


def compute_monosemanticity(
    token_attributions: Sequence[dict[int, int]],
    token_embeddings: torch.Tensor,
    cos_threshold: float = 0.1,
) -> Monosemanticity:
    """Score how semantically coherent each feature's activating tokens are.

    For each feature, its activating tokens are grouped into semantic clusters
    by cosine similarity of their embeddings (tokens above `cos_threshold`
    share a cluster, via connected components). The paper's monosemanticity is
    then the ratio of the entropy over those clusters to the entropy over the
    raw tokens:

    `R_mono(i) = 1 - H_semantic(i) / H_token(i)`

    A feature that fires on many tokens which all share one meaning scores 1,
    while a feature whose tokens all have distinct meanings scores 0. Features
    firing on a single token type (and features with no activations, which the
    paper scores the same way) get 1.

    Args:
        token_attributions: one {token_id: activation_count} mapping per feature,
            as returned by `collect_token_attributions`.
        token_embeddings: (vocab_size, d_embed) embeddings used to group tokens by meaning.
        cos_threshold: cosine similarity above which two tokens share a semantic
            cluster (the paper uses 0.1).
    """
    token_entropies: list[float] = []
    semantic_entropies: list[float] = []
    scores: list[float] = []

    for counts in token_attributions:
        h_token = _entropy(torch.tensor(list(counts.values()), dtype=torch.float64))
        cluster_totals: dict[int, float] = {}
        if counts:
            token_ids = torch.tensor(sorted(counts))
            raw_counts = torch.tensor(
                [counts[int(token_id)] for token_id in token_ids],
                dtype=torch.float64,
            )
            clusters = _semantic_clusters(token_embeddings[token_ids], cos_threshold)
            for cluster, count in zip(
                clusters.tolist(), raw_counts.tolist(), strict=True
            ):
                cluster_totals[cluster] = cluster_totals.get(cluster, 0.0) + count
        h_semantic = _entropy(
            torch.tensor(list(cluster_totals.values()), dtype=torch.float64)
        )

        token_entropies.append(h_token)
        semantic_entropies.append(h_semantic)
        scores.append(1.0 if h_token == 0 else 1.0 - h_semantic / h_token)

    return Monosemanticity(
        torch.tensor(token_entropies),
        torch.tensor(semantic_entropies),
        torch.tensor(scores),
    )


def _entropy(counts: torch.Tensor) -> float:
    """Natural-log entropy of a count vector, ignoring zero counts."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * probs.log()).sum())


def _semantic_clusters(
    embeddings: torch.Tensor, cos_threshold: float
) -> torch.Tensor:
    """Label each token with a semantic cluster id.

    Tokens are linked when their cosine similarity exceeds `cos_threshold`, and
    clusters are the connected components of that graph.
    """
    n_tokens = embeddings.shape[0]
    parent = list(range(n_tokens))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    if n_tokens > 1:
        norms = embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        normalized = embeddings / norms
        similar = (normalized @ normalized.T) > cos_threshold
        for i in range(n_tokens):
            for j in range(i + 1, n_tokens):
                if bool(similar[i, j]):
                    parent[find(i)] = find(j)

    return torch.tensor([find(i) for i in range(n_tokens)])


@dataclass
class BilingualAlignment:
    """Layer- or checkpoint-level summary of bilingual feature alignment.

    The paper's signature of a bilingual layer is a high mixed-language
    feature proportion together with high monosemanticity, so `score` is their
    product.
    """

    mixed_proportion: float
    mean_monosemanticity: float

    @property
    def score(self) -> float:
        return self.mixed_proportion * self.mean_monosemanticity


def bilingual_alignment(
    selectivity: LanguageSelectivity, monosemanticity: Monosemanticity
) -> BilingualAlignment:
    """Summarize a layer (or checkpoint) from its feature statistics.

    Args:
        selectivity: language statistics for the layer's SAE.
        monosemanticity: token statistics for the same SAE.
    """
    live = selectivity.live_mask
    n_live = int(live.sum())
    if n_live == 0:
        return BilingualAlignment(0.0, 0.0)
    # Dead features are scored 1 by convention, so they are excluded here
    # rather than inflating the mean.
    mean_score = float(monosemanticity.score[live].mean())
    return BilingualAlignment(selectivity.mixed_proportion, mean_score)


def peak_alignment_layer(profiles: Sequence[BilingualAlignment]) -> int:
    """Return the index of the layer with the highest bilingual alignment score."""
    return max(range(len(profiles)), key=lambda layer: profiles[layer].score)


def transfer_decomposed_features(
    sae: SAE[Any],
    donor_activations: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Decompose donor activations and keep only the masked feature directions.

    Implements the update `alpha * D(mask * E(x_donor))` from the paper's
    cross-model intervention, where the SAE is trained on the fully trained
    (donor) model and only features of one kind (language-specific or
    bilingual) survive the mask.

    Args:
        sae: SAE trained on the donor model's activations at the hooked layer.
        donor_activations: (..., d_in) donor model activations for the batch
            being run through the recipient model.
        mask: (d_sae,) boolean feature mask, e.g. from `feature_mask`.
        alpha: injection strength (the paper uses 0.1).
    """
    feature_acts = sae.encode(donor_activations)
    masked_acts = feature_acts * mask.to(feature_acts.dtype)
    return alpha * sae.decode(masked_acts)


def make_feature_injection_hook(
    sae: SAE[Any],
    donor_activations: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.1,
) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    """Build a forward hook that adds transferred donor features to a layer.

    The returned hook is meant for `HookedSAETransformer.run_with_hooks` (or
    `run_with_hooks_with_saes`) at the layer the SAE was trained on, and adds
    `alpha * D(mask * E(x_donor))` to the residual stream of the recipient
    model. Since the donor activations belong to the same input batch as the
    recipient run, the update is computed once, when the hook is built.

    Args:
        sae: SAE trained on the donor model's activations at the hooked layer.
        donor_activations: (..., d_in) donor model activations, matching the
            shape of the hooked activation during the recipient run.
        mask: (d_sae,) boolean feature mask, e.g. from `feature_mask`.
        alpha: injection strength (the paper uses 0.1).
    """
    update = transfer_decomposed_features(sae, donor_activations, mask, alpha)

    def inject(residual: torch.Tensor, _hook: Any) -> torch.Tensor:
        return residual + update

    return inject
