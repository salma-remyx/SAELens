"""Typical analysis workflow: supervised top-k SAE feature selection.

Follows the SAE-FiRE recipe (Cances et al., https://arxiv.org/abs/2505.14420):
run token-level activations through an SAE, sum-pool the feature activations
over the context of each labelled example, rank features by an ANOVA
F-statistic, and keep the top-k rather than the full (very wide) feature
vector. The workflow below checks the part that matters downstream: features
selected on one half of the examples still separate the classes on the
held-out half.
"""

import torch

from sae_lens.analysis.feature_selection import (
    SupervisedFeatureScorer,
    pool_over_context,
    select_supervised_features,
)
from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig
from tests.helpers import random_params

D_IN = 32
D_SAE = 128
CONTEXT_SIZE = 8
NUM_EXAMPLES = 400
FIT_CHUNK_SIZE = 64
TOP_K = 8


def _pooled_feature_acts(
    sae: StandardSAE, labels: torch.Tensor, signal_direction: torch.Tensor
) -> torch.Tensor:
    """Encode label-directed token activations and sum-pool them per example.

    Every token of an example sits near +signal_direction for label 1 and near
    -signal_direction for label 0, with a little per-token noise, so the class
    signal is whatever the SAE makes of a single direction. The final context
    position is padding and is masked out of the pooling.
    """
    signs = torch.where(labels == 1, 1.0, -1.0).view(-1, 1, 1)
    hidden_acts = 2.0 * signs * signal_direction + 0.05 * torch.randn(
        len(labels), CONTEXT_SIZE, D_IN
    )
    token_feature_acts = sae.encode(hidden_acts)
    token_feature_acts[:, -1, :] = 1e6
    mask = torch.ones(len(labels), CONTEXT_SIZE, dtype=torch.bool)
    mask[:, -1] = False
    return pool_over_context(token_feature_acts, mask)


def test_typical_supervised_feature_selection_workflow():
    sae = StandardSAE(
        StandardSAEConfig(
            d_in=D_IN,
            d_sae=D_SAE,
            dtype="float32",
            device="cpu",
            apply_b_dec_to_input=False,
        )
    )
    random_params(sae)

    labels = torch.randint(0, 2, (NUM_EXAMPLES,))
    signal_direction = torch.randn(D_IN)
    signal_direction = signal_direction / signal_direction.norm()
    pooled_acts = _pooled_feature_acts(sae, labels, signal_direction)

    split = NUM_EXAMPLES // 2
    fit_acts, fit_labels = pooled_acts[:split], labels[:split]
    eval_acts, eval_labels = pooled_acts[split:], labels[split:]

    scorer = SupervisedFeatureScorer()
    for start in range(0, split, FIT_CHUNK_SIZE):
        scorer.add_batch(
            fit_acts[start : start + FIT_CHUNK_SIZE],
            fit_labels[start : start + FIT_CHUNK_SIZE],
        )
    f_stats = scorer.f_statistics()
    top_features = scorer.top_features(TOP_K)

    assert len(torch.unique(top_features)) == TOP_K
    assert f_stats[top_features[0]] == f_stats.max()

    # Nearest-centroid classifier on the single best selected feature, with the
    # centroid and its orientation taken from the fit half only.
    best = int(top_features[0])
    fit_class_0 = fit_acts[fit_labels == 0, best].mean()
    fit_class_1 = fit_acts[fit_labels == 1, best].mean()
    threshold = (fit_class_0 + fit_class_1) / 2
    above = eval_acts[:, best] > threshold
    predicted = (above if fit_class_1 > fit_class_0 else above.logical_not()).long()

    assert (predicted == eval_labels).float().mean() > 0.9


def test_typical_streaming_selection_matches_one_shot_api():
    sae = StandardSAE(
        StandardSAEConfig(
            d_in=D_IN,
            d_sae=D_SAE,
            dtype="float32",
            device="cpu",
            apply_b_dec_to_input=False,
        )
    )
    random_params(sae)

    labels = torch.randint(0, 2, (NUM_EXAMPLES,))
    signal_direction = torch.randn(D_IN)
    signal_direction = signal_direction / signal_direction.norm()
    pooled_acts = _pooled_feature_acts(sae, labels, signal_direction)

    streamed = SupervisedFeatureScorer()
    for start in range(0, NUM_EXAMPLES, FIT_CHUNK_SIZE):
        streamed.add_batch(
            pooled_acts[start : start + FIT_CHUNK_SIZE],
            labels[start : start + FIT_CHUNK_SIZE],
        )
    one_shot_indices, one_shot_f_stats = select_supervised_features(
        pooled_acts, labels, TOP_K
    )

    assert torch.equal(streamed.top_features(TOP_K), one_shot_indices)
    assert torch.allclose(streamed.f_statistics(), one_shot_f_stats)
