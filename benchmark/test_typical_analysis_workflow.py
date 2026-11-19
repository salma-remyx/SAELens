"""Typical analysis workflows exercised end to end on synthetic data.

Each test in this file drives a pre-existing library surface through a
realistic workflow shape a researcher would run, without needing a language
model or dataset download.
"""

import copy
from typing import cast

import torch

from sae_lens.registry import (
    SAE_TRAINING_CLASS_REGISTRY,
    get_sae_training_class,
    register_sae_training_class,
)
from sae_lens.saes.contrastive_batchtopk_sae import (
    ContrastiveBatchTopKTrainingSAE,
    ContrastiveBatchTopKTrainingSAEConfig,
)
from sae_lens.saes.sae import TrainStepInput

N_PAIRS = 60
D_IN = 64
D_SAE = 32
K = 4
NOISE = 0.15
N_STEPS = 300
LR = 1e-3
INFONCE_COEFFICIENT = 1.0
INFONCE_TEMPERATURE = 0.1


def _register_contrastive_batchtopk() -> None:
    if "contrastive_batchtopk" not in SAE_TRAINING_CLASS_REGISTRY:
        register_sae_training_class(
            "contrastive_batchtopk",
            ContrastiveBatchTopKTrainingSAE,
            ContrastiveBatchTopKTrainingSAEConfig,
        )


def _build_retrieval_corpus() -> torch.Tensor:
    # each pair is a query and its relevant document: nearby but distinct
    # points, with every other point in the corpus a hard negative
    centers = torch.randn(N_PAIRS, D_IN)
    centers = centers / centers.norm(dim=-1, keepdim=True)
    return torch.cat([centers, centers]) + NOISE * torch.randn(2 * N_PAIRS, D_IN)


def _partner_recall_at_1(feature_acts: torch.Tensor) -> float:
    directions = feature_acts / feature_acts.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    similarities = directions @ directions.T
    similarities.fill_diagonal_(float("-inf"))
    nearest = similarities.argmax(dim=-1)
    partner = (torch.arange(2 * N_PAIRS) + N_PAIRS) % (2 * N_PAIRS)
    return (nearest == partner).float().mean().item()


def _train(
    sae: ContrastiveBatchTopKTrainingSAE,
    corpus: torch.Tensor,
    infonce_coefficient: float,
) -> tuple[float, float]:
    # mirrors what SAETrainer does with get_coefficients(): the scheduled
    # coefficient is delivered to the SAE through TrainStepInput
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
    step_input = TrainStepInput(
        sae_in=corpus,
        coefficients={"infonce": infonce_coefficient},
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=False,
    )
    first_infonce = None
    last_infonce = None
    for _ in range(N_STEPS):
        output = sae.training_forward_pass(step_input)
        infonce = output.losses["infonce_loss"].item()
        first_infonce = infonce if first_infonce is None else first_infonce
        last_infonce = infonce
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    assert first_infonce is not None and last_infonce is not None
    return first_infonce, last_infonce


def test_typical_contrastive_retrieval_workflow() -> None:
    _register_contrastive_batchtopk()
    # construct through the registry, the way custom architectures are
    # consumed by the standard training runners
    training_cls, training_cfg_cls = get_sae_training_class("contrastive_batchtopk")
    sae_cls = cast("type[ContrastiveBatchTopKTrainingSAE]", training_cls)
    cfg_cls = cast("type[ContrastiveBatchTopKTrainingSAEConfig]", training_cfg_cls)

    corpus = _build_retrieval_corpus()

    contrastive_sae = sae_cls(
        cfg_cls(
            d_in=D_IN,
            d_sae=D_SAE,
            k=K,
            infonce_loss_coefficient=INFONCE_COEFFICIENT,
            infonce_temperature=INFONCE_TEMPERATURE,
        )
    )
    baseline_sae = sae_cls(cfg_cls(d_in=D_IN, d_sae=D_SAE, k=K))
    # paired comparison: both arms train from the same initialization
    baseline_sae.load_state_dict(copy.deepcopy(contrastive_sae.state_dict()))

    first_infonce, last_infonce = _train(contrastive_sae, corpus, INFONCE_COEFFICIENT)
    _train(baseline_sae, corpus, 0.0)

    with torch.no_grad():
        contrastive_recall = _partner_recall_at_1(contrastive_sae.encode(corpus))
        baseline_recall = _partner_recall_at_1(baseline_sae.encode(corpus))

    # the contrastive objective collapses the InfoNCE loss it optimizes
    assert last_infonce < first_infonce / 5
    # and the sparse codes retrieve the relevant document for far more
    # queries than an identically-initialized reconstruction-only SAE
    assert contrastive_recall > 0.5
    assert contrastive_recall > baseline_recall + 0.15
