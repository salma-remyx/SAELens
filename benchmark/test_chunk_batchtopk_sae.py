import torch

from sae_lens.config import SAETrainerConfig
from sae_lens.registry import (
    SAE_TRAINING_CLASS_REGISTRY,
    get_sae_training_class,
    register_sae_training_class,
)
from sae_lens.saes.chunk_batchtopk_sae import (
    ChunkBatchTopKTrainingSAE,
    ChunkBatchTopKTrainingSAEConfig,
    mean_pool_chunks,
)
from sae_lens.saes.sae import TrainingSAEConfig
from sae_lens.training.sae_trainer import SAETrainer


def register_chunk_batchtopk() -> None:
    """Register the architecture via the public extension point.

    Called at test-run time rather than import time so the
    ALL_TRAINING_ARCHITECTURES snapshots taken during collection are
    unaffected.
    """
    if "chunk_batchtopk" not in SAE_TRAINING_CLASS_REGISTRY:
        register_sae_training_class(
            "chunk_batchtopk",
            ChunkBatchTopKTrainingSAE,
            ChunkBatchTopKTrainingSAEConfig,
        )


N_CONCEPTS = 32
D_IN = 16
CHUNK_SIZE = 8
N_TOKENS_PER_BATCH = 512


def make_chunk_structured_batch(concepts: torch.Tensor) -> torch.Tensor:
    """Token activations where each contiguous chunk shares one concept direction."""
    n_chunks = N_TOKENS_PER_BATCH // CHUNK_SIZE
    chunk_concepts = concepts[torch.randint(0, N_CONCEPTS, (n_chunks,))]
    token_signal = chunk_concepts.repeat_interleave(CHUNK_SIZE, dim=0)
    return token_signal + 0.3 * torch.randn_like(token_signal)


def chunk_reconstruction_mse(
    sae: ChunkBatchTopKTrainingSAE, batch: torch.Tensor
) -> float:
    pooled = mean_pool_chunks(batch, CHUNK_SIZE)
    with torch.no_grad():
        recons = sae.decode(sae.encode(pooled))
    return ((recons - pooled) ** 2).sum(dim=-1).mean().item()


def test_chunk_batchtopk_resolves_through_sae_training_registry():
    register_chunk_batchtopk()
    sae_class, sae_config_class = get_sae_training_class("chunk_batchtopk")
    assert sae_class is ChunkBatchTopKTrainingSAE
    assert sae_config_class is ChunkBatchTopKTrainingSAEConfig

    cfg = ChunkBatchTopKTrainingSAEConfig(
        d_in=D_IN,
        d_sae=64,
        k=6,
        chunk_size=CHUNK_SIZE,
        chunk_variant="joint",
    )
    assert cfg.to_dict()["architecture"] == "chunk_batchtopk"
    # this is the dispatch the config loaders and training runners rely on
    roundtrip_cfg = TrainingSAEConfig.from_dict(cfg.to_dict())
    assert isinstance(roundtrip_cfg, ChunkBatchTopKTrainingSAEConfig)
    assert roundtrip_cfg.chunk_variant == "joint"
    assert roundtrip_cfg.chunk_size == CHUNK_SIZE


def test_chunk_batchtopk_learns_chunk_structure_through_sae_trainer():
    register_chunk_batchtopk()
    cfg = ChunkBatchTopKTrainingSAEConfig(
        d_in=D_IN,
        d_sae=64,
        k=6,
        chunk_size=CHUNK_SIZE,
        chunk_variant="mean",
    )
    sae = ChunkBatchTopKTrainingSAE(cfg)
    concepts = torch.randn(N_CONCEPTS, D_IN)
    concepts = concepts / concepts.norm(dim=-1, keepdim=True)

    trainer = SAETrainer(
        SAETrainerConfig(
            total_training_samples=120 * N_TOKENS_PER_BATCH,
            train_batch_size_samples=N_TOKENS_PER_BATCH,
            lr=3e-3,
            lr_end=3e-3,
        ),
        sae,
        data_provider=lambda: iter([]),
    )

    initial_mse = chunk_reconstruction_mse(sae, make_chunk_structured_batch(concepts))
    for _ in range(120):
        trainer.step(make_chunk_structured_batch(concepts))
    final_mse = chunk_reconstruction_mse(sae, make_chunk_structured_batch(concepts))

    # the batchtopk threshold must have engaged during chunk-level training
    assert sae.topk_threshold.item() > 0.0
    # pooled chunks live on the unit sphere, so an untrained SAE pays O(1) error;
    # a chunk-level SAE that has learned the chunk directions pays far less
    assert final_mse < 0.5 * initial_mse
