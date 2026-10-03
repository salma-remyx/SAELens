"""
Grouping of hookpoints by residual-stream similarity, so that one SAE can be
trained per group of similar hooks instead of one SAE per hook.

Adapted from "Group-SAE: Efficient Training of Sparse Autoencoders for Large
Language Models via Layer Groups" (https://arxiv.org/abs/2410.21508). The
paper observes that residual-stream representations of nearby layers are
strongly aligned, and scores a partition of layers with the Average Maximum
Angular Distance (AMAD): within each group the mean angular distance between
every pair of layer activation streams is computed, the group's diameter is
the maximum such pairwise distance, and AMAD is the mean diameter over
groups. Layers are grouped with complete-linkage agglomerative clustering,
and the smallest number of groups whose AMAD falls below a threshold (0.2 in
the paper) is selected.

All functions expect paired activation streams: one (n_tokens, d_in) tensor
per hookpoint, with rows at the same index coming from the same token
position. `collect_paired_activations` gathers such streams from a multi-hook
data provider (e.g. `ActivationsStore.get_multi_hook_data_loader()`), which
applies a single shared shuffle to every hook and therefore preserves row
alignment across hooks.
"""

import math
from collections.abc import Iterator, Mapping, Sequence

import torch


def mean_angular_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Mean angular distance between paired rows of two activation streams.

    Both tensors must have the same shape (n_tokens, d_in); row i of `x` is
    compared with row i of `y`. The angular distance of a pair is arccos of
    their cosine similarity divided by pi, so identical directions map to
    0.0, orthogonal directions to 0.5 and opposite directions to 1.0.
    """
    if x.shape != y.shape:
        raise ValueError(
            f"activation streams must have the same shape to be row-aligned; "
            f"got {tuple(x.shape)} and {tuple(y.shape)}"
        )
    cosine = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    angular = torch.arccos(cosine.clamp(-1.0, 1.0)) / math.pi
    return angular.mean().item()


def angular_distance_matrix(
    hook_activations: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """
    Pairwise mean angular distances between hookpoint activation streams.

    Takes one row-aligned (n_tokens, d_in) stream per hookpoint and returns a
    symmetric (n_hooks, n_hooks) matrix with zeros on the diagonal.
    Row/column order follows the mapping's key order.
    """
    names = list(hook_activations)
    n_tokens = hook_activations[names[0]].shape[0] if names else 0
    for name, activations in hook_activations.items():
        if activations.shape[0] != n_tokens:
            raise ValueError(
                "activation streams must have the same number of tokens to be "
                f"row-aligned; hook {name!r} has {activations.shape[0]} rows, "
                f"expected {n_tokens}"
            )

    distances = torch.zeros(len(names), len(names))
    for i, name_i in enumerate(names):
        for j in range(i + 1, len(names)):
            distance = mean_angular_distance(
                hook_activations[name_i], hook_activations[names[j]]
            )
            distances[i, j] = distance
            distances[j, i] = distance
    return distances


def complete_linkage_groups(
    distance_matrix: torch.Tensor, n_groups: int
) -> list[list[int]]:
    """
    Bottom-up agglomerative clustering with complete linkage.

    Starts from singletons and repeatedly merges the two clusters with the
    smallest complete-linkage distance (the maximum pairwise distance between
    their members) until `n_groups` clusters remain. Ties break to the
    lowest-index pair, so the result is deterministic.
    """
    n = distance_matrix.shape[0]
    if distance_matrix.ndim != 2 or distance_matrix.shape[1] != n:
        raise ValueError(
            f"distance_matrix must be square; got shape {tuple(distance_matrix.shape)}"
        )
    if not 1 <= n_groups <= n:
        raise ValueError(f"n_groups must be in [1, {n}]; got {n_groups}")

    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > n_groups:
        best_distance: float | None = None
        best_pair = (0, 1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = distance_matrix[clusters[i]][:, clusters[j]].max().item()
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)
        i, j = best_pair
        clusters[i] = clusters[i] + clusters[j]
        clusters.pop(j)
    return clusters


def amad_score(distance_matrix: torch.Tensor, groups: Sequence[Sequence[int]]) -> float:
    """
    Average Maximum Angular Distance of a partition of hookpoints.

    Each group's diameter is the maximum pairwise distance between its
    members (0.0 for singletons); the score is the mean diameter over all
    groups. Lower scores mean more internally similar groups.
    """
    diameters = [
        distance_matrix[list(group)][:, list(group)].max().item() for group in groups
    ]
    return sum(diameters) / len(diameters)


def select_hook_groups(
    hook_activations: Mapping[str, torch.Tensor],
    threshold: float = 0.2,
) -> list[list[str]]:
    """
    Select the smallest grouping of hookpoints whose AMAD is below threshold.

    Hookpoints are grouped by complete-linkage clustering of their pairwise
    mean angular distances, increasing the number of groups until the AMAD
    score drops below `threshold` (0.2 in the paper). Hooks within each group
    are listed in the mapping's key order. If the threshold can never be met
    (threshold <= 0), every hookpoint becomes its own group.
    """
    names = list(hook_activations)
    distances = angular_distance_matrix(hook_activations)
    for n_groups in range(1, len(names) + 1):
        groups = complete_linkage_groups(distances, n_groups)
        if amad_score(distances, groups) < threshold:
            return [[names[i] for i in sorted(group)] for group in groups]
    return [[name] for name in names]


def collect_paired_activations(
    data_provider: Iterator[dict[str, torch.Tensor]],
    hook_names: Sequence[str],
    n_batches: int,
) -> dict[str, torch.Tensor]:
    """
    Stack batches from a multi-hook data provider into one stream per hook.

    Pulls `n_batches` batches and concatenates them along the token axis, so
    the returned (n_batches * batch_tokens, d_in) streams stay row-aligned
    across hooks.
    """
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in hook_names}
    for _ in range(n_batches):
        batch = next(data_provider)
        for name in hook_names:
            collected[name].append(batch[name])
    return {name: torch.cat(chunks, dim=0) for name, chunks in collected.items()}
