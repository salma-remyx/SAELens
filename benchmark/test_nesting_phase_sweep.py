import math

import pytest
import torch

from benchmark.nesting_phase_sweep import (
    NestingSweepCell,
    NestingSweepConfig,
    build_nested_synthetic_model,
    compute_recovery_metrics,
    nested_pairs,
    nested_pairs_modifier,
    run_nesting_sweep,
    run_single_cell,
)
from sae_lens.saes.standard_sae import StandardTrainingSAE, StandardTrainingSAEConfig
from sae_lens.synthetic.activation_generator import ActivationGenerator
from sae_lens.synthetic.evals import eval_sae_on_synthetic_data


def test_nested_pairs_covers_requested_fraction_of_features():
    assert nested_pairs(16, 0.0) == []
    assert nested_pairs(16, 0.5) == [(0, 1), (2, 3), (4, 5), (6, 7)]
    assert len(nested_pairs(16, 1.0)) == 8

    involved = {index for pair in nested_pairs(16, 1.0) for index in pair}
    assert involved == set(range(16))

    with pytest.raises(ValueError, match="nesting_fraction"):
        nested_pairs(16, 1.5)


def test_nested_pairs_modifier_gates_children_on_parents():
    generator = ActivationGenerator(
        num_features=8,
        firing_probabilities=0.3,
        modify_activations=nested_pairs_modifier([(0, 1), (2, 3)]),
    )
    activations = generator.sample(200_000)

    for child, parent in [(0, 1), (2, 3)]:
        children_without_parent = activations[:, child][activations[:, parent] == 0]
        assert torch.all(children_without_parent == 0)

    # Parents and unpaired features keep firing at p, children at p^2.
    for feature in [1, 3, 7]:
        firing_rate = activations[:, feature].gt(0).float().mean().item()
        assert firing_rate == pytest.approx(0.3, abs=0.004)
    for feature in [0, 2]:
        firing_rate = activations[:, feature].gt(0).float().mean().item()
        assert firing_rate == pytest.approx(0.09, abs=0.004)


def test_recovery_metrics_reports_full_recovery_for_matched_dictionary():
    gt_features = torch.eye(4)
    metrics = compute_recovery_metrics(torch.eye(4), gt_features)

    assert metrics.phase == "recovered"
    assert metrics.recovered_fraction == 1.0
    assert metrics.merged_fraction == 0.0
    assert metrics.median_best_cosine == pytest.approx(1.0)


def test_recovery_metrics_flags_atom_shared_by_two_features_as_merged():
    gt_features = torch.eye(4)
    merged_atom = (gt_features[0] + gt_features[1]) / math.sqrt(2)
    decoder = torch.stack([merged_atom, gt_features[2], gt_features[3]])

    metrics = compute_recovery_metrics(decoder, gt_features)

    assert metrics.phase == "merged"
    assert metrics.recovered_fraction == pytest.approx(0.5)
    assert metrics.merged_fraction == pytest.approx(0.5)
    assert metrics.best_cosines.tolist() == pytest.approx(
        [1 / math.sqrt(2), 1 / math.sqrt(2), 1.0, 1.0]
    )


def test_recovery_metrics_reports_diffuse_when_atoms_split_three_features():
    gt_features = torch.eye(4)
    decoder = torch.stack(
        [
            (gt_features[0] + gt_features[1] + gt_features[2]) / math.sqrt(3),
            (gt_features[1] + gt_features[2] + gt_features[3]) / math.sqrt(3),
            (gt_features[2] + gt_features[3] + gt_features[0]) / math.sqrt(3),
            (gt_features[3] + gt_features[0] + gt_features[1]) / math.sqrt(3),
        ]
    )

    metrics = compute_recovery_metrics(decoder, gt_features)

    assert metrics.phase == "diffuse"
    assert metrics.recovered_fraction == 0.0
    assert metrics.merged_fraction == 0.0
    assert metrics.median_best_cosine == pytest.approx(1 / math.sqrt(3))


def test_existing_synthetic_evaluator_sees_nesting_through_model_builder():
    model = build_nested_synthetic_model(
        num_features=8,
        hidden_dim=8,
        nesting_fraction=0.5,
        firing_probability=0.3,
    )
    sae = StandardTrainingSAE(
        StandardTrainingSAEConfig(d_in=8, d_sae=8, l1_coefficient=0.01)
    )

    result = eval_sae_on_synthetic_data(
        sae=sae,
        feature_dict=model.feature_dict,
        activations_generator=model.activation_generator,
        num_samples=100_000,
        batch_size=10_000,
    )

    # Two parents and four unpaired features fire at 0.3; the two nested
    # children only fire alongside their parents, at 0.3^2 = 0.09.
    expected_l0 = 6 * 0.3 + 2 * 0.09
    assert result.true_l0 == pytest.approx(expected_l0, abs=0.015)


def test_run_single_cell_trains_through_synthetic_runner_and_reports_phase():
    cell = run_single_cell(
        num_features=8,
        nesting_fraction=0.5,
        l1_coefficient=0.01,
        firing_probability=0.3,
        training_samples=400,
        batch_size=40,
        eval_samples=50_000,
    )

    assert isinstance(cell, NestingSweepCell)
    assert cell.num_features == 8
    assert cell.nesting_fraction == 0.5
    assert cell.l1_coefficient == 0.01
    assert cell.phase in ("recovered", "merged", "diffuse")
    assert 0.0 <= cell.recovered_fraction <= 1.0
    assert 0.0 <= cell.merged_fraction <= 1.0
    assert 0.0 <= cell.median_best_cosine <= 1.0
    assert cell.true_l0 == pytest.approx(1.98, abs=0.02)
    assert cell.sae_l0 > 0.0
    assert cell.density_ratio > 0.0

    logged = cell.to_dict()
    assert logged["phase"] == cell.phase
    assert logged["true_l0"] == cell.true_l0


def test_run_nesting_sweep_covers_grid_and_orders_cells():
    config = NestingSweepConfig(
        nesting_fractions=(0.0, 1.0),
        l1_coefficients=(0.01,),
        num_features_grid=(8,),
        seeds=(0,),
        firing_probability=0.3,
        training_samples=400,
        batch_size=40,
        eval_samples=20_000,
    )

    cells = run_nesting_sweep(config)

    assert len(cells) == 2
    assert [cell.nesting_fraction for cell in cells] == [0.0, 1.0]
    for cell in cells:
        assert cell.phase in ("recovered", "merged", "diffuse")
        assert cell.true_l0 > 0.0
        assert cell.sae_l0 > 0.0
