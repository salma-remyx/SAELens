import copy
from collections.abc import Iterator

import pytest
import torch

from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.saes.sae import TrainCoefficientConfig
from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig
from sae_lens.training.multi_sae_trainer import MultiSAETrainer
from sae_lens.training.mutual_feature_regularization import (
    MutualFeatureMultiSAETrainer,
    MutualFeatureSAEMixin,
    mean_max_cosine_sim,
    mutual_feature_penalty,
)
from tests.helpers import assert_close, random_params

D_IN = 16
D_SAE = 8
K = 3
BATCH_SIZE = 64


class TopKMFRTrainingSAE(MutualFeatureSAEMixin, TopKTrainingSAE):
    pass


def _make_sae() -> TopKMFRTrainingSAE:
    cfg = TopKTrainingSAEConfig(
        d_in=D_IN,
        d_sae=D_SAE,
        k=K,
        decoder_init_norm=0.1,
        normalize_activations="none",
        dtype="float32",
        device="cpu",
    )
    sae = TopKMFRTrainingSAE(cfg)
    random_params(sae)
    return sae


def _trainer_cfg(total_samples: int, lr: float = 1e-3) -> SAETrainerConfig:
    return SAETrainerConfig(
        total_training_samples=total_samples,
        train_batch_size_samples=BATCH_SIZE,
        lr=lr,
        lr_end=lr / 10,
        device="cpu",
        n_checkpoints=0,
        save_final_checkpoint=False,
        logger=LoggingConfig(log_to_wandb=False),
        n_batches_for_norm_estimate=2,
    )


def _repeat_provider(batch: torch.Tensor) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield {"hook": batch}


def _symmetric_mmcs(a: torch.Tensor, b: torch.Tensor) -> float:
    return 0.5 * (mean_max_cosine_sim(a, b) + mean_max_cosine_sim(b, a)).item()


def test_mean_max_cosine_sim_is_directional_and_scale_invariant():
    identity = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    duplicate_axis = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    def mmcs(a: torch.Tensor, b: torch.Tensor) -> float:
        return mean_max_cosine_sim(a, b).item()

    assert mmcs(identity, identity) == pytest.approx(1.0)
    assert mmcs(identity, duplicate_axis) == pytest.approx(0.5)
    assert mmcs(duplicate_axis, identity) == pytest.approx(1.0)
    # cosine similarity ignores feature magnitudes
    assert mmcs(identity, 3.0 * duplicate_axis) == pytest.approx(0.5)
    # a row with no counterpart drags the mean down
    assert mmcs(identity, torch.tensor([[0.0, 1.0]])) == pytest.approx(0.5)


def test_mutual_feature_penalty_value_and_peer_detachment():
    features = torch.randn(6, 4)

    def penalty(rows: torch.Tensor, peers: list[torch.Tensor]) -> float:
        return mutual_feature_penalty(rows, peers).item()

    # identical feature sets have zero penalty; a lone SAE also has zero
    assert penalty(features, [features.clone()]) == pytest.approx(0.0, abs=1e-6)
    assert penalty(features, []) == pytest.approx(0.0)

    # orthogonal feature sets are maximally dissimilar
    axis_a = torch.tensor([[1.0, 0.0]])
    axis_b = torch.tensor([[0.0, 1.0]])
    assert penalty(axis_a, [axis_b]) == pytest.approx(1.0)

    own = torch.randn(5, 4, requires_grad=True)
    peer = torch.randn(5, 4, requires_grad=True)
    mutual_feature_penalty(own, [peer]).backward()
    assert own.grad is not None
    # peers are detached: no gradient may leak into the other SAE's optimizer
    assert peer.grad is None


def test_mfr_trainer_rejects_saes_without_the_mixin():
    plain_sae = TopKTrainingSAE(
        TopKTrainingSAEConfig(
            d_in=D_IN, d_sae=D_SAE, k=K, dtype="float32", device="cpu"
        )
    )
    with pytest.raises(TypeError, match="MutualFeatureSAEMixin"):
        MutualFeatureMultiSAETrainer(
            cfg=_trainer_cfg(total_samples=BATCH_SIZE),
            saes={"a": plain_sae, "b": plain_sae},  # type: ignore[dict-item]
            hook_names={"a": "hook", "b": "hook"},
            data_provider=_repeat_provider(torch.randn(BATCH_SIZE, D_IN)),
        )


def test_mfr_with_zero_coefficient_matches_stock_multi_sae_trainer():
    proto_a = _make_sae()
    proto_b = _make_sae()
    batch = torch.randn(BATCH_SIZE, D_IN)
    cfg = _trainer_cfg(total_samples=BATCH_SIZE)

    stock_trainer = MultiSAETrainer(
        cfg=cfg,
        saes={"a": copy.deepcopy(proto_a), "b": copy.deepcopy(proto_b)},
        hook_names={"a": "hook", "b": "hook"},
        data_provider=_repeat_provider(batch),
    )
    mfr_trainer = MutualFeatureMultiSAETrainer(
        cfg=cfg,
        saes={"a": copy.deepcopy(proto_a), "b": copy.deepcopy(proto_b)},
        hook_names={"a": "hook", "b": "hook"},
        data_provider=_repeat_provider(batch),
        mfr_coefficient=0.0,
    )

    for name in ("a", "b"):
        stock_out = stock_trainer.trainers[name].step(batch)
        mfr_out = mfr_trainer.trainers[name].step(batch)
        # the penalty is wired into the existing aux-loss dict but contributes
        # nothing at coefficient zero
        assert "mfr_loss" in mfr_out.losses
        assert mfr_out.losses["mfr_loss"].item() == pytest.approx(0.0)
        assert "mfr_loss" not in stock_out.losses
        assert_close(mfr_trainer.saes[name].W_dec, stock_trainer.saes[name].W_dec)


def test_mfr_coefficient_warm_up_ramps_penalty_from_zero():
    trainer = MutualFeatureMultiSAETrainer(
        cfg=_trainer_cfg(total_samples=8 * BATCH_SIZE),
        saes={"a": _make_sae(), "b": _make_sae()},
        hook_names={"a": "hook", "b": "hook"},
        data_provider=_repeat_provider(torch.randn(BATCH_SIZE, D_IN)),
        mfr_coefficient=TrainCoefficientConfig(value=2.0, warm_up_steps=4),
    )
    batch = next(trainer.data_provider)["hook"]
    outputs = [trainer.trainers["a"].step(batch) for _ in range(4)]

    mfr_losses = [out.losses["mfr_loss"].item() for out in outputs]
    assert mfr_losses[0] == pytest.approx(0.0)
    assert all(loss > 0 for loss in mfr_losses[1:])
    assert mfr_losses[3] > mfr_losses[1]


def test_mfr_coupling_aligns_decoder_features_across_saes():
    n_steps = 100
    batch = torch.randn(BATCH_SIZE, D_IN)
    cfg = _trainer_cfg(total_samples=n_steps * BATCH_SIZE, lr=1e-2)
    proto_a = _make_sae()
    proto_b = _make_sae()
    before = _symmetric_mmcs(proto_a.W_dec, proto_b.W_dec)

    control_trainer = MultiSAETrainer(
        cfg=cfg,
        saes={"a": copy.deepcopy(proto_a), "b": copy.deepcopy(proto_b)},
        hook_names={"a": "hook", "b": "hook"},
        data_provider=_repeat_provider(batch),
    )
    control_trainer.fit()

    mfr_trainer = MutualFeatureMultiSAETrainer(
        cfg=cfg,
        saes={"a": copy.deepcopy(proto_a), "b": copy.deepcopy(proto_b)},
        hook_names={"a": "hook", "b": "hook"},
        data_provider=_repeat_provider(batch),
        mfr_coefficient=30.0,
    )
    mfr_trainer.fit()

    control_after = _symmetric_mmcs(
        control_trainer.saes["a"].W_dec, control_trainer.saes["b"].W_dec
    )
    mfr_after = _symmetric_mmcs(
        mfr_trainer.saes["a"].W_dec, mfr_trainer.saes["b"].W_dec
    )
    # trained independently on the same data, the two SAEs drift apart
    # (each learns partly spurious features); the coupling pulls the
    # decoder feature directions back into near-perfect alignment
    assert mfr_after > before
    assert mfr_after > control_after + 0.2
    assert mfr_after > 0.95
