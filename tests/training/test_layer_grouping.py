import pytest
import torch

from sae_lens.training.layer_grouping import (
    amad_score,
    angular_distance_matrix,
    collect_paired_activations,
    complete_linkage_groups,
    mean_angular_distance,
    select_hook_groups,
)


def test_mean_angular_distance_for_identical_orthogonal_and_opposed_streams():
    stream = torch.tensor([[1.0, 0.0]]).repeat(64, 1)
    orthogonal = torch.tensor([[0.0, 1.0]]).repeat(64, 1)
    opposed = torch.tensor([[-1.0, 0.0]]).repeat(64, 1)

    assert mean_angular_distance(stream, stream) == pytest.approx(0.0, abs=1e-6)
    assert mean_angular_distance(stream, orthogonal) == pytest.approx(0.5)
    assert mean_angular_distance(stream, opposed) == pytest.approx(1.0)


def test_mean_angular_distance_of_independent_streams_is_half():
    x = torch.randn(20_000, 256)
    y = torch.randn(20_000, 256)

    assert mean_angular_distance(x, y) == pytest.approx(0.5, abs=0.01)


def test_mean_angular_distance_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="row-aligned"):
        mean_angular_distance(torch.randn(8, 4), torch.randn(9, 4))


def test_angular_distance_matrix_is_symmetric_with_zero_diagonal():
    streams = {
        "a": torch.randn(128, 16),
        "b": torch.randn(128, 16) * 3.0,
        "c": torch.randn(128, 16) + 5.0,
    }

    distances = angular_distance_matrix(streams)

    assert distances.shape == (3, 3)
    assert torch.allclose(distances, distances.T)
    assert torch.allclose(distances.diagonal(), torch.zeros(3))
    off_diagonal = distances[~torch.eye(3, dtype=torch.bool)]
    assert torch.all((off_diagonal > 0.0) & (off_diagonal < 1.0))


def test_angular_distance_matrix_rejects_mismatched_token_counts():
    streams = {"a": torch.randn(10, 4), "b": torch.randn(12, 4)}

    with pytest.raises(ValueError, match="row-aligned"):
        angular_distance_matrix(streams)


def test_complete_linkage_groups_clusters_known_distance_matrix():
    distances = torch.tensor(
        [
            [0.0, 0.1, 0.8, 0.9],
            [0.1, 0.0, 0.7, 0.8],
            [0.8, 0.7, 0.0, 0.1],
            [0.9, 0.8, 0.1, 0.0],
        ]
    )

    assert complete_linkage_groups(distances, 2) == [[0, 1], [2, 3]]
    assert complete_linkage_groups(distances, 3) == [[0, 1], [2], [3]]
    assert complete_linkage_groups(distances, 4) == [[0], [1], [2], [3]]

    with pytest.raises(ValueError, match="n_groups"):
        complete_linkage_groups(distances, 5)


def test_amad_score_averages_group_diameters():
    distances = torch.tensor(
        [
            [0.0, 0.1, 0.8, 0.9],
            [0.1, 0.0, 0.7, 0.8],
            [0.8, 0.7, 0.0, 0.1],
            [0.9, 0.8, 0.1, 0.0],
        ]
    )

    assert amad_score(distances, [[0, 1], [2, 3]]) == pytest.approx(0.1)
    assert amad_score(distances, [[0], [1], [2], [3]]) == pytest.approx(0.0)
    assert amad_score(distances, [[0, 1, 2, 3]]) == pytest.approx(0.9)


def test_select_hook_groups_recovers_two_similar_clusters():
    n_tokens, d_in = 512, 32
    base_a = torch.randn(n_tokens, d_in)
    base_b = torch.randn(n_tokens, d_in)
    streams = {
        "layer0": base_a,
        "layer0_post": base_a + 0.01 * torch.randn(n_tokens, d_in),
        "layer1": base_b,
        "layer1_post": base_b + 0.01 * torch.randn(n_tokens, d_in),
    }

    groups = select_hook_groups(streams, threshold=0.2)

    assert groups == [["layer0", "layer0_post"], ["layer1", "layer1_post"]]


def test_select_hook_groups_returns_singletons_for_nonpositive_threshold():
    streams = {"a": torch.randn(64, 8), "b": torch.randn(64, 8)}

    assert select_hook_groups(streams, threshold=0.0) == [["a"], ["b"]]


def test_collect_paired_activations_concatenates_batches_in_order():
    def provider():
        for step in range(3):
            yield {
                "h0": torch.full((2, 3), float(step)),
                "h1": torch.full((2, 3), float(step) + 10.0),
            }

    streams = collect_paired_activations(iter(provider()), ["h0", "h1"], 3)

    assert streams["h0"].shape == (6, 3)
    assert torch.equal(
        streams["h0"],
        torch.tensor(
            [[0.0] * 3, [0.0] * 3, [1.0] * 3, [1.0] * 3, [2.0] * 3, [2.0] * 3]
        ),
    )
    assert torch.equal(streams["h1"], streams["h0"] + 10.0)
