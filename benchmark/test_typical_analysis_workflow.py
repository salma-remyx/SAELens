"""Typical analysis workflow: SAE features as causal adjustment variables.

Adapted from "Exploring Sparse Autoencoders in Text-Based Causal Confounding
Adjustment" (https://arxiv.org/abs/2609.01322v1): encode documents with an SAE,
then greedily select the minimal set of features that suffices to adjust for
text-based confounding, and compare the adjusted treatment effect against the
naive unadjusted one.

The workflow runs offline against synthetic documents and a freshly initialized
TopKSAE; for real analyses, swap the activations for cached corpus activations
and load a pretrained SAE with ``SAE.load_from_pretrained``.
"""

import numpy as np
import torch

from sae_lens.analysis.causal_adjustment import select_adjustment_set
from sae_lens.saes import TopKSAE, TopKSAEConfig


# Run with: poetry run pytest benchmark/test_typical_analysis_workflow.py -v -s
def test_typical_confounder_adjustment_workflow():
    n_documents = 6000
    d_in = 64
    d_sae = 128
    topk = 16
    true_effect = 0.0

    rng = np.random.default_rng()
    # latent confounders hidden in each document: they drive both the treatment
    # assignment and the outcome, so the naive estimate of the treatment effect
    # is biased unless the analysis adjusts for them
    confounders = rng.normal(size=(n_documents, 3))
    treatment = (
        rng.random(n_documents)
        < 1.0 / (1.0 + np.exp(-(2.5 * confounders[:, 0] + confounders[:, 1])))
    ).astype(np.float64)
    outcome = (
        true_effect * treatment
        + 1.5 * confounders[:, 0]
        + 0.75 * confounders[:, 1]
        + 0.5 * rng.normal(size=n_documents)
    )

    # document activations embedding the confounders plus idiosyncratic content
    mixing = rng.normal(size=(3, d_in))
    activations = confounders @ mixing + 0.5 * rng.normal(size=(n_documents, d_in))

    sae = TopKSAE(TopKSAEConfig(d_in=d_in, d_sae=d_sae, k=topk))
    with torch.no_grad():
        feature_acts = sae.encode(torch.from_numpy(activations).float())

    result = select_adjustment_set(feature_acts, treatment, outcome)

    print(f"selected features: {result.feature_indices}")
    print(f"per-step CI-test p-values: {[round(p, 4) for p in result.p_values]}")
    print(f"unadjusted effect: {result.unadjusted_effect:.4f}")
    print(f"adjusted effect:   {result.adjusted_effect:.4f}")

    assert result.ci_test_accepted
    assert 0 < len(result.feature_indices) < d_sae // 2
    assert abs(result.adjusted_effect - true_effect) < abs(
        result.unadjusted_effect - true_effect
    )
