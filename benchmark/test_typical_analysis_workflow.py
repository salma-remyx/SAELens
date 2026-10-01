"""Typical analysis workflow for chunk-level (Mean-Chunk) SAEs.

Exercises the ChunkTrainingSAE added in sae_lens/saes/chunk_sae.py the way a
user would: register the custom architecture through the public registry,
build it from a config dict, train it on activations with chunk-level
(semantic) structure, and check that its features capture what is shared
across a span better than a token-level TopK SAE trained on the same data.

The synthetic corpus follows the setting of "Beyond Token Scale: Chunk-Level
Sparse Autoencoders for Reliable Semantic Feature Discovery"
(https://arxiv.org/abs/2609.35521): each chunk of contiguous tokens carries
one topic direction plus per-token noise, so token-level objectives must
spend their sparse budget on the noise while a chunk-level objective only
sees the shared content.
"""

import torch

from sae_lens.registry import (
    SAE_TRAINING_CLASS_REGISTRY,
    get_sae_training_class,
    register_sae_training_class,
)
from sae_lens.saes.chunk_sae import (
    ChunkTrainingSAE,
    ChunkTrainingSAEConfig,
    chunk_means,
)
from sae_lens.saes.sae import TrainStepInput
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig

D_IN = 16
D_SAE = 64
N_TOPICS = 4
N_CHUNKS = 512
CHUNK_SIZE = 8
K = 4
N_TRAINING_STEPS = 800
TOPIC_NOISE_SCALE = 0.25


def register_chunk_sae() -> None:
    """Register the chunk architecture through the public registry.

    Custom architectures are meant to go through this extension point (added
    in the v6 refactor) rather than the package __init__. Registration happens
    inside the tests, not at import time, so that a full pytest run never sees
    "chunk" leak into tests.helpers.ALL_TRAINING_ARCHITECTURES (snapshotted at
    import time, before any test runs).
    """
    if "chunk" not in SAE_TRAINING_CLASS_REGISTRY:
        register_sae_training_class("chunk", ChunkTrainingSAE, ChunkTrainingSAEConfig)


def train_step_input(sae_in: torch.Tensor) -> TrainStepInput:
    return TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=False,
    )


def make_topic_chunks(
    topics: torch.Tensor, n_chunks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked activations: each chunk is one topic direction plus token noise."""
    topic_ids = torch.randint(0, topics.shape[0], (n_chunks,))
    spans = topics[topic_ids][:, None, :] + TOPIC_NOISE_SCALE * torch.randn(
        n_chunks, CHUNK_SIZE, D_IN
    )
    return topic_ids, spans


def topic_selectivity(
    feature_acts: torch.Tensor, topic_ids: torch.Tensor, n_topics: int
) -> tuple[float, float]:
    """Mean cosine of each chunk's features to its own vs other topics' means."""
    feats = feature_acts.flatten(0, -2)
    topic_means = torch.stack([feats[topic_ids == t].mean(0) for t in range(n_topics)])
    feats_normalized = torch.nn.functional.normalize(feats, dim=-1)
    topic_means_normalized = torch.nn.functional.normalize(topic_means, dim=-1)
    sim = feats_normalized @ topic_means_normalized.T
    same = sim[torch.arange(len(topic_ids)), topic_ids].mean().item()
    off = sim.clone()
    off[torch.arange(len(topic_ids)), topic_ids] = float("nan")
    return same, torch.nanmean(off).item()


def train_sae(sae: torch.nn.Module, sae_in: torch.Tensor) -> None:
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for _ in range(N_TRAINING_STEPS):
        output = sae.training_forward_pass(train_step_input(sae_in))
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_chunk_sae_builds_through_the_public_registry() -> None:
    register_chunk_sae()
    sae_class, config_class = get_sae_training_class("chunk")
    assert sae_class is ChunkTrainingSAE
    assert config_class is ChunkTrainingSAEConfig

    config_dict = ChunkTrainingSAEConfig(
        d_in=D_IN, d_sae=D_SAE, k=float(K), chunk_size=CHUNK_SIZE
    ).to_dict()
    assert config_dict["architecture"] == "chunk"
    assert config_dict["chunk_size"] == CHUNK_SIZE

    sae = sae_class(config_class.from_dict(config_dict))
    spans = torch.randn(3, CHUNK_SIZE, D_IN)
    assert sae(spans).shape == (3, 1, D_IN)


def test_mean_chunk_sae_beats_token_level_topk_on_chunk_level_structure() -> None:
    register_chunk_sae()
    topics = torch.linalg.qr(torch.randn(D_IN, N_TOPICS)).Q.T
    _, train_spans = make_topic_chunks(topics, N_CHUNKS)
    held_topic_ids, held_spans = make_topic_chunks(topics, N_CHUNKS)
    held_means = chunk_means(held_spans, CHUNK_SIZE).flatten(0, -2)

    chunk_sae = ChunkTrainingSAE(
        ChunkTrainingSAEConfig(
            d_in=D_IN, d_sae=D_SAE, k=float(K), chunk_size=CHUNK_SIZE
        )
    )
    token_sae = TopKTrainingSAE(TopKTrainingSAEConfig(d_in=D_IN, d_sae=D_SAE, k=K))
    train_sae(chunk_sae, train_spans)
    train_sae(token_sae, train_spans.reshape(N_CHUNKS * CHUNK_SIZE, D_IN))

    with torch.no_grad():
        chunk_same, chunk_cross = topic_selectivity(
            chunk_sae.encode(held_spans), held_topic_ids, N_TOPICS
        )
        token_same, _ = topic_selectivity(
            token_sae.encode(held_means), held_topic_ids, N_TOPICS
        )
        # both SAEs reconstruct held-out chunk means with the same
        # per-sample sparse budget
        chunk_recon_error = (
            (chunk_sae(held_spans).flatten(0, -2) - held_means).pow(2).sum(-1).mean()
        )
        token_recon_error = (token_sae(held_means) - held_means).pow(2).sum(-1).mean()

    chunk_recon_mse = chunk_recon_error.item()
    token_recon_mse = token_recon_error.item()

    assert chunk_same > 0.9, (chunk_same, chunk_cross)
    assert chunk_cross < 0.1, (chunk_same, chunk_cross)
    assert chunk_same - token_same > 0.05, (chunk_same, token_same)
    assert chunk_recon_mse < 0.8 * token_recon_mse, (chunk_recon_mse, token_recon_mse)
