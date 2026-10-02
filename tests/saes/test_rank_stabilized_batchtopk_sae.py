import pytest
import torch

from sae_lens.saes.batchtopk_sae import (
    BatchTopK,
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.rank_stabilized_batchtopk_sae import (
    RankStabilizedBatchTopKTrainingSAE,
    RankStabilizedBatchTopKTrainingSAEConfig,
    active_margin,
    batchtopk_cutoff,
    feature_sensitivity,
    mean_active_margin,
    pairwise_rank_stabilization_loss,
)
from sae_lens.saes.sae import TrainStepInput
from tests.helpers import assert_close, random_params


def build_step_input(sae_in: torch.Tensor) -> TrainStepInput:
    return TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=None,
        n_training_steps=0,
        is_logging_step=False,
    )


def test_batchtopk_cutoff_matches_smallest_selected_activation():
    pre_acts = torch.randn(6, 10) + 1.0
    k = 3.0

    cutoff = batchtopk_cutoff(pre_acts, k)
    selected = BatchTopK(k)(pre_acts)

    assert_close(cutoff, selected[selected > 0].min())


def test_active_margin_and_mean_active_margin_on_hand_computed_example():
    # with k=2 over 2 samples, 4 of the 8 ReLU'd values are selected, so the
    # cutoff is the 4th largest of [5, 3, 0, 0, 4, 0.25, 1.5, 0] = 1.5
    pre_acts = torch.tensor([[5.0, 3.0, 0.0, 0.0], [4.0, 0.25, 1.5, 0.0]])

    cutoff = batchtopk_cutoff(pre_acts, k=2)
    assert cutoff.item() == pytest.approx(1.5)

    margins = active_margin(pre_acts, cutoff)
    assert_close(
        margins,
        torch.tensor([[3.5, 1.5, -1.5, -1.5], [2.5, -1.25, 0.0, -1.5]]),
    )

    # only the three strictly-selected features (5, 3, 4) count towards the mean
    assert mean_active_margin(pre_acts, cutoff).item() == pytest.approx(
        (3.5 + 1.5 + 2.5) / 3
    )


def test_pairwise_rank_stabilization_loss_is_zero_when_ranks_are_stable():
    # identical pair members can never disagree about the selection boundary
    identical = torch.randn(5, 2, 8) + 1.0
    identical[:, 1] = identical[:, 0]
    assert pairwise_rank_stabilization_loss(identical, k=2).item() == 0.0

    # features kept in both members (even with different margins) or in
    # neither member contribute nothing: both members keep features 0 and 1,
    # with member b's feature 1 sitting exactly on the cutoff
    stable = torch.tensor([[[5.0, 4.0, 0.0, 0.0], [6.0, 2.0, 0.0, 0.0]]])
    # k=2 over 2 samples selects 4 of the 8 values, cutoff = 2.0
    assert pairwise_rank_stabilization_loss(stable, k=2).item() == pytest.approx(0.0)


def test_pairwise_rank_stabilization_loss_penalizes_boundary_ordering_failures():
    # member a keeps feature 1 with margin 0.5 while member b drops it 1.25
    # below the cutoff: the penalty is capped by the keeping margin (0.5)
    capped = torch.tensor([[[5.0, 2.0, 0.0, 0.0], [4.0, 0.25, 1.5, 0.0]]])
    assert pairwise_rank_stabilization_loss(capped, k=2).item() == pytest.approx(0.5)

    # member a keeps feature 1 with margin 0.25 while member b drops it only
    # 0.125 below the cutoff: the deficit itself is charged (uncapped)
    uncapped = torch.tensor([[[5.0, 3.0, 0.0, 0.0], [4.0, 2.625, 2.75, 0.0]]])
    assert pairwise_rank_stabilization_loss(uncapped, k=2).item() == pytest.approx(
        0.125
    )


def test_pairwise_rank_stabilization_loss_only_pushes_dropped_features_up():
    # the deficit (0.125) is smaller than the keeping margin (0.25), so the
    # gradient pushes the dropped feature back up towards the cutoff
    uncapped = torch.tensor(
        [[[5.0, 3.0, 0.0, 0.0], [4.0, 2.625, 2.75, 0.0]]], requires_grad=True
    )
    pairwise_rank_stabilization_loss(uncapped, k=2).backward()
    assert uncapped.grad is not None
    assert_close(
        uncapped.grad,
        torch.tensor([[[0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]]]),
    )

    # when the deficit exceeds the keeping margin the penalty is capped by the
    # (detached) margin, so no gradient flows at all
    capped = torch.tensor(
        [[[5.0, 2.0, 0.0, 0.0], [4.0, 0.25, 1.5, 0.0]]], requires_grad=True
    )
    pairwise_rank_stabilization_loss(capped, k=2).backward()
    assert capped.grad is not None
    assert_close(capped.grad, torch.zeros_like(capped))


def test_feature_sensitivity_on_hand_computed_example():
    feature_acts = torch.tensor(
        [
            [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]],  # overlap 2 / union 3
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],  # overlap 0 / union 2
        ]
    )
    assert feature_sensitivity(feature_acts).item() == pytest.approx(
        (2 / 3 + 0.0) / 2
    )

    identical = torch.rand(4, 2, 6) + 1.0
    identical[:, 1] = identical[:, 0]
    assert feature_sensitivity(identical).item() == pytest.approx(1.0)

    # pairs where neither member activates anything are excluded
    assert feature_sensitivity(torch.zeros(3, 2, 4)).item() == 0.0


def test_rank_stabilized_sae_adds_pairwise_loss_and_metrics_to_training_step():
    cfg = RankStabilizedBatchTopKTrainingSAEConfig(
        d_in=8,
        d_sae=32,
        k=4,
        decoder_init_norm=0.1,
        pairwise_rank_loss_coefficient=0.5,
    )
    sae = RankStabilizedBatchTopKTrainingSAE(cfg)
    random_params(sae)
    sae_in = torch.randn(16, 8)

    output = sae.training_forward_pass(build_step_input(sae_in))

    assert set(output.losses.keys()) == {
        "mse_loss",
        "auxiliary_reconstruction_loss",
        "pairwise_rank_stabilization_loss",
    }
    # the aux loss dict contract sums every entry into the total loss
    assert_close(output.loss, torch.stack(list(output.losses.values())).sum())
    assert_close(
        output.losses["pairwise_rank_stabilization_loss"],
        0.5
        * pairwise_rank_stabilization_loss(
            output.hidden_pre.reshape(-1, 2, cfg.d_sae), cfg.k
        ),
    )
    assert "active_cutoff" in output.metrics
    assert "mean_active_margin" in output.metrics
    sensitivity_metric = output.metrics["pairwise_feature_sensitivity"]
    assert isinstance(sensitivity_metric, torch.Tensor)
    assert sensitivity_metric.item() == pytest.approx(
        feature_sensitivity(output.feature_acts).item()
    )
    assert 0.0 <= sensitivity_metric.item() <= 1.0


def test_rank_stabilized_sae_with_zero_coefficient_matches_plain_batchtopk():
    sae = RankStabilizedBatchTopKTrainingSAE(
        RankStabilizedBatchTopKTrainingSAEConfig(
            d_in=8, d_sae=32, k=4, decoder_init_norm=0.1
        )
    )
    random_params(sae)
    baseline = BatchTopKTrainingSAE(
        BatchTopKTrainingSAEConfig(d_in=8, d_sae=32, k=4, decoder_init_norm=0.1)
    )
    baseline.load_state_dict(sae.state_dict())

    sae_in = torch.randn(16, 8)
    output = sae.training_forward_pass(build_step_input(sae_in))
    baseline_output = baseline.training_forward_pass(build_step_input(sae_in))

    assert set(output.losses.keys()) == {
        "mse_loss",
        "auxiliary_reconstruction_loss",
    }
    assert set(baseline_output.losses.keys()) == set(output.losses.keys())
    for loss_name in output.losses:
        assert_close(output.losses[loss_name], baseline_output.losses[loss_name])
    assert_close(output.loss, baseline_output.loss)
    # boundary metrics still get logged without the stabilization loss
    assert "active_cutoff" in output.metrics
    assert "mean_active_margin" in output.metrics
    assert "pairwise_feature_sensitivity" not in output.metrics


def test_rank_stabilized_sae_raises_on_unpaired_batch():
    cfg = RankStabilizedBatchTopKTrainingSAEConfig(
        d_in=8,
        d_sae=32,
        k=4,
        decoder_init_norm=0.1,
        pairwise_rank_loss_coefficient=0.5,
    )
    sae = RankStabilizedBatchTopKTrainingSAE(cfg)

    with pytest.raises(ValueError, match="even number of samples"):
        sae.training_forward_pass(build_step_input(torch.randn(3, 8)))
