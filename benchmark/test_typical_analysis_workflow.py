import torch

from sae_lens.analysis.feature_correspondence import feature_correspondence
from sae_lens.saes.topk_sae import TopKSAE
from tests.helpers import build_topk_sae_cfg, random_params


def test_typical_feature_correspondence_workflow():
    """
    Typical cross-architecture analysis workflow: encode a shared token corpus
    with two SAEs and summarize their feature correspondence, in the spirit of
    the Mamba-vs-Transformer analysis of arXiv:2609.24440. Two small TopK SAEs
    with different input dims stand in for SAEs trained on different models.

    In a real run the two activation matrices are collected over the same tokens
    with load_model("HookedTransformer", ...) and load_model("HookedMamba", ...),
    e.g. the residual streams of EleutherAI/pythia-70m and state-spaces/mamba-130m.
    """
    transformer_sae = TopKSAE(build_topk_sae_cfg(d_in=64, d_sae=128, k=16))
    ssm_sae = TopKSAE(build_topk_sae_cfg(d_in=32, d_sae=128, k=16))
    random_params(transformer_sae)
    random_params(ssm_sae)

    transformer_view = torch.rand(2048, 64)
    ssm_view = torch.rand(2048, 32)

    report = feature_correspondence(
        transformer_sae,
        ssm_sae,
        transformer_view,
        ssm_view,
        divergence_threshold=0.5,
    )

    active_similarity = report.best_match_similarity[report.active_mask]
    if active_similarity.numel() > 0:
        print(f"median best-match Jaccard: {float(active_similarity.median()):.4f}")
    print(
        "divergent features "
        f"(< {report.divergence_threshold}): {len(report.divergent_features)} "
        f"({report.divergent_fraction:.2%}) of {report.n_active_features} active"
    )

    assert report.best_match_similarity.shape == (128,)
    assert torch.all(report.best_match_indices < 128)
    assert 0.0 <= report.divergent_fraction <= 1.0
