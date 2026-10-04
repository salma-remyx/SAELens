"""The typical SAE evaluation workflow: unsupervised proxies plus a downstream capability check.

SAEBench (https://arxiv.org/abs/2503.09532) argues that the unsupervised
proxy metrics SAEs are usually judged by (L0, cross-entropy, KL, density,
weight statistics) have unclear practical relevance on their own. The
workflow here therefore pairs the existing ``sae_lens.evals`` proxy metrics
with ``benchmark.bench_sparse_probing``'s downstream sparse-probing eval
over the same labeled activations, so an SAE is scored both on how it
reconstructs and on whether a small budget of its latents linearly recovers
a document-level label.

The synthetic helpers build activations with one known concept direction
and an SAE whose encoder detects it, giving a ground-truth check that the
downstream metric separates a fit SAE from an unstructured one.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch

from benchmark.bench_sparse_probing import run_sparse_probing_eval
from sae_lens.evals import get_featurewise_weight_based_metrics
from sae_lens.saes.sae import SAE
from sae_lens.saes.standard_sae import StandardSAE
from tests.helpers import build_sae_cfg, random_params

CONCEPT_DIM = 0


def make_labeled_activations(
    n_per_label: int = 256,
    d_in: int = 128,
    signal_scale: float = 5.0,
    noise_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Two activation groups differing in exactly one direction.

    "concept" rows add a signal_scale offset along dimension CONCEPT_DIM on
    top of shared Gaussian noise; "baseline" rows carry the noise alone.
    """
    noise = torch.randn(2, n_per_label, d_in) * noise_scale
    noise[0, :, CONCEPT_DIM] += signal_scale
    return {"baseline": noise[1], "concept": noise[0]}


def build_concept_sae(d_in: int = 128, d_sae: int = 256) -> StandardSAE:
    """StandardSAE whose first latent detects the concept dimension.

    The remaining latents get small random detectors so the weight-based
    proxy metrics stay finite while none of them rivals the concept latent.
    """
    sae = StandardSAE(build_sae_cfg(d_in=d_in, d_sae=d_sae, device="cpu"))
    with torch.no_grad():
        sae.W_enc.zero_()
        sae.b_enc.zero_()
        sae.W_dec.zero_()
        sae.b_dec.zero_()
        sae.W_enc[CONCEPT_DIM, 0] = 1.0
        sae.W_dec[0, CONCEPT_DIM] = 1.0
        sae.W_enc[:, 1:] = 0.1 * torch.randn(d_in, d_sae - 1) / d_in**0.5
        sae.W_dec[1:, :] = 0.1 * torch.randn(d_sae - 1, d_in) / d_in**0.5
    return sae


def run_typical_evaluation_workflow(
    sae: SAE[Any],
    activations_by_label: Mapping[str, torch.Tensor],
    k_values: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Score an SAE the standard way: unsupervised proxies from
    sae_lens.evals plus the downstream sparse-probing metric on the same
    activations."""
    proxies = get_featurewise_weight_based_metrics(sae)
    downstream = run_sparse_probing_eval(sae, activations_by_label, k_values=k_values)
    return {"unsupervised_proxies": proxies, **downstream}


def test_typical_evaluation_workflow_discriminates_fit_sae_from_unstructured_sae():
    activations = make_labeled_activations()

    fit_sae = build_concept_sae()
    fit_metrics = run_typical_evaluation_workflow(fit_sae, activations, k_values=[1, 2])
    fit_downstream = fit_metrics["downstream_sparse_probing"]
    assert fit_downstream["sae_probe_accuracies"][0] >= 0.95
    fit_proxies = fit_metrics["unsupervised_proxies"]
    assert fit_proxies["encoder_decoder_cosine_sim"][0] == pytest.approx(1.0)
    assert fit_proxies["encoder_norm"][0] == pytest.approx(1.0)

    unstructured_sae = StandardSAE(build_sae_cfg(d_in=128, d_sae=256, device="cpu"))
    random_params(unstructured_sae)
    unstructured_metrics = run_typical_evaluation_workflow(
        unstructured_sae, activations, k_values=[1, 2]
    )
    unstructured_downstream = unstructured_metrics["downstream_sparse_probing"]
    assert (
        unstructured_downstream["sae_probe_accuracies"][0]
        < fit_downstream["sae_probe_accuracies"][0] - 0.15
    )
