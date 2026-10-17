from pathlib import Path

import pytest
import torch

from sae_lens.registry import (
    SAE_CLASS_REGISTRY,
    SAE_TRAINING_CLASS_REGISTRY,
    register_sae_class,
    register_sae_training_class,
)
from sae_lens.saes.sae import SAE, TrainStepInput
from sae_lens.saes.sparsemap_sae import (
    SparseMapTopK,
    SparseMapTopKSAE,
    SparseMapTopKSAEConfig,
    SparseMapTopKTrainingSAE,
    SparseMapTopKTrainingSAEConfig,
    entmax15,
    participation_stats,
    sparsemax,
)
from sae_lens.saes.topk_sae import TopK
from tests.helpers import assert_close, random_params


def _bisect_simplex_map(v: torch.Tensor, power: int) -> torch.Tensor:
    # reference implementation: the map is relu(v - tau) ** power where the
    # threshold tau solves sum(relu(v - tau) ** power) == 1, found by bisection
    # (the left-hand side is decreasing in tau)
    v = v - v.max(dim=-1, keepdim=True).values
    lo = v.min(dim=-1, keepdim=True).values - 2
    hi = v.max(dim=-1, keepdim=True).values
    for _ in range(100):
        mid = (lo + hi) / 2
        g = (v - mid).clamp_min(0.0).pow(power).sum(dim=-1, keepdim=True)
        move_lo = g > 1
        lo = torch.where(move_lo, mid, lo)
        hi = torch.where(move_lo, hi, mid)
    p = (v - (lo + hi) / 2).clamp_min(0.0)
    return p if power == 1 else p * p


@pytest.fixture
def registered_sparsemap_classes():
    register_sae_class("sparsemap_topk", SparseMapTopKSAE, SparseMapTopKSAEConfig)
    register_sae_training_class(
        "sparsemap_topk", SparseMapTopKTrainingSAE, SparseMapTopKTrainingSAEConfig
    )
    yield
    del SAE_CLASS_REGISTRY["sparsemap_topk"]
    del SAE_TRAINING_CLASS_REGISTRY["sparsemap_topk"]


@pytest.mark.parametrize("map_fn,power", [(sparsemax, 1), (entmax15, 2)])
def test_probability_maps_solve_their_simplex_projection(map_fn, power):
    inputs = []
    for size in [1, 2, 3, 5, 64, 257]:
        for scale in [0.01, 0.1, 1.0, 5.0, 50.0]:
            inputs.append(torch.randn(64, size, dtype=torch.float64) * scale)
    # many ties / duplicated scores
    inputs.append(torch.randn(64, 32, dtype=torch.float64).round())
    # one dominant entry
    dominant = torch.randn(64, 16, dtype=torch.float64)
    dominant[:, 0] += 50.0
    inputs.append(dominant)

    for v in inputs:
        p = map_fn(v)
        assert (p >= 0).all()
        sums = p.sum(dim=-1)
        assert_close(sums, torch.ones_like(sums), atol=1e-12, rtol=1e-12)
        assert_close(p, _bisect_simplex_map(v, power), atol=1e-12, rtol=1e-12)


def test_probability_maps_hand_computed_values():
    v = torch.tensor([[1.5, 1.0]], dtype=torch.float64)
    assert_close(sparsemax(v), torch.tensor([[0.75, 0.25]], dtype=torch.float64))
    expected_entmax = torch.tensor(
        [[0.8307189138830738, 0.1692810861169262]], dtype=torch.float64
    )
    assert_close(entmax15(v), expected_entmax)
    # both maps drop the smaller entry once the gap reaches 1
    for map_fn in (sparsemax, entmax15):
        kept = (map_fn(torch.tensor([[0.99, 0.0]])) > 0).sum()
        dropped = (map_fn(torch.tensor([[1.01, 0.0]])) > 0).sum()
        assert kept.item() == 2
        assert dropped.item() == 1


@pytest.mark.parametrize("probability_map", ["sparsemax", "entmax15"])
def test_SparseMapTopK_matches_TopK_when_nothing_is_dropped(probability_map):
    x = torch.tensor([[1.0, 1.01, 0.99, -5.0, -6.0]])

    output = SparseMapTopK(k=3, probability_map=probability_map)(x)

    assert torch.equal(output, TopK(k=3)(x))


@pytest.mark.parametrize("probability_map", ["sparsemax", "entmax15"])
def test_SparseMapTopK_drops_selected_features_with_large_gaps(probability_map):
    x = torch.tensor([[10.0, 1.0, 1.0, 1.0]])

    output = SparseMapTopK(k=3, probability_map=probability_map)(x)

    expected = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    assert torch.equal(output, expected)
    # the surviving activation is exactly what plain TopK with k=1 would produce
    assert torch.equal(output, TopK(k=1)(x))


@pytest.mark.parametrize("probability_map", ["sparsemax", "entmax15"])
def test_SparseMapTopK_is_robust_to_raising_k_at_inference(probability_map):
    # each sample has two dominant scores with a small gap between them and
    # everything else far below the map's drop threshold
    num_samples, d_sae = 20000, 32
    x = torch.randn(num_samples, d_sae) * 0.2
    col1 = torch.randint(0, d_sae, (num_samples,))
    col2 = (col1 + 1 + torch.randint(0, d_sae - 1, (num_samples,))) % d_sae
    spikes = torch.randn(num_samples, 2) * 0.2 + 5.0
    x.scatter_(1, torch.stack([col1, col2], dim=1), spikes)

    acts_k2 = SparseMapTopK(k=2, probability_map=probability_map)(x)
    acts_k8 = SparseMapTopK(k=8, probability_map=probability_map)(x)
    sparsemap_identical = (acts_k2 == acts_k8).all(dim=-1).float().mean().item()

    topk_acts_k2 = TopK(k=2)(x)
    topk_acts_k8 = TopK(k=8)(x)
    topk_identical = (topk_acts_k2 == topk_acts_k8).all(dim=-1).float().mean().item()

    # raising the budget from 2 to 8 barely changes the sparse-map activations
    # (extra selected scores fall below the map's threshold and get exact zeros)
    assert sparsemap_identical > 0.999
    # while plain TopK output changes on essentially every sample
    assert topk_identical < 0.05


def test_SparseMapTopK_participation_is_adaptive_and_entmax15_drops_less():
    x = torch.randn(50000, 64)
    sparse_acts = SparseMapTopK(k=16, probability_map="sparsemax")(x)
    entmax_acts = SparseMapTopK(k=16, probability_map="entmax15")(x)

    for acts in (sparse_acts, entmax_acts):
        active = (acts > 0).sum(dim=-1)
        assert (active >= 1).all()
        assert (active <= 16).all()

    sparse_l0 = (sparse_acts > 0).sum(dim=-1).float().mean().item()
    entmax_l0 = (entmax_acts > 0).sum(dim=-1).float().mean().item()
    # on standard normal scores sparsemax keeps ~3.2 of the 16 selected
    # features on average, and entmax15 keeps ~4.6 (it drops less)
    assert 3.0 < sparse_l0 < 3.5
    assert 4.3 < entmax_l0 < 4.9
    assert entmax_l0 > sparse_l0


def test_SparseMapTopKTrainingSAE_training_forward_pass():
    cfg = SparseMapTopKTrainingSAEConfig(
        d_in=8, d_sae=24, k=4, probability_map="entmax15", device="cpu"
    )
    sae = SparseMapTopKTrainingSAE(cfg)
    random_params(sae)

    sae_in = torch.randn(64, sae.cfg.d_in)
    step_input = TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=False,
    )

    output = sae.training_forward_pass(step_input)

    assert torch.isfinite(output.loss)
    assert "mse_loss" in output.losses
    assert "auxiliary_reconstruction_loss" in output.losses
    active = (output.feature_acts > 0).sum(dim=-1)
    # the map never activates more than the k selected features, and samples
    # whose selected pre-activations are all negative have none active (the
    # same ReLU behaviour as plain TopK)
    assert (active <= sae.cfg.k).all()
    assert active.sum() > 0
    assert output.metrics["mean_active_features"] == pytest.approx(
        active.float().mean().item()
    )

    output.loss.backward()
    assert sae.W_enc.grad is not None
    assert torch.isfinite(sae.W_enc.grad).all()


@pytest.mark.usefixtures("registered_sparsemap_classes")
def test_SparseMapTopKTrainingSAE_save_and_load_inference_sae(tmp_path: Path):
    cfg = SparseMapTopKTrainingSAEConfig(
        d_in=8, d_sae=24, k=2, probability_map="sparsemax", device="cpu"
    )
    training_sae = SparseMapTopKTrainingSAE(cfg)
    random_params(training_sae)

    sae_in = torch.randn(32, training_sae.cfg.d_in)

    training_sae.save_inference_model(str(tmp_path))
    inference_sae = SAE.load_from_disk(str(tmp_path), device="cpu")

    assert isinstance(inference_sae, SparseMapTopKSAE)
    assert inference_sae.cfg.k == cfg.k
    assert inference_sae.cfg.probability_map == cfg.probability_map

    training_acts, _ = training_sae.encode_with_hidden_pre(sae_in)
    assert_close(inference_sae.encode(sae_in), training_acts)
    assert_close(inference_sae(sae_in), training_sae(sae_in))


@pytest.mark.usefixtures("registered_sparsemap_classes")
def test_sparsemap_configs_roundtrip_through_the_registry():
    cfg = SparseMapTopKTrainingSAEConfig(
        d_in=8, d_sae=24, k=4, probability_map="entmax15", device="cpu"
    )

    loaded = SparseMapTopKTrainingSAEConfig.from_dict(cfg.to_dict())
    assert isinstance(loaded, SparseMapTopKTrainingSAEConfig)
    assert loaded.probability_map == "entmax15"
    assert loaded.k == 4

    inference_cfg_dict = cfg.get_inference_sae_cfg_dict()
    assert inference_cfg_dict["architecture"] == "sparsemap_topk"
    assert inference_cfg_dict["probability_map"] == "entmax15"
    assert inference_cfg_dict["k"] == 4


def test_SparseMapTopKTrainingSAEConfig_rejects_use_sparse_activations():
    with pytest.raises(ValueError, match="use_sparse_activations"):
        SparseMapTopKTrainingSAEConfig(
            d_in=8, d_sae=24, k=4, use_sparse_activations=True
        )


def test_participation_stats():
    feature_acts = torch.tensor(
        [
            [1.0, 2.0, 0.0, 0.0],  # 2 active
            [3.0, 0.0, 0.0, 0.0],  # 1 active
            [1.0, 0.5, 2.0, 0.0],  # 3 active (nothing dropped)
        ]
    )

    stats = participation_stats(feature_acts, k=3)

    assert stats["mean_active_features"] == pytest.approx(2.0)
    assert stats["frac_samples_below_budget"] == pytest.approx(2 / 3)
    assert stats["frac_single_feature_samples"] == pytest.approx(1 / 3)
