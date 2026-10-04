"""Downstream-capability evaluation for SAEs via sparse linear probing.

Adapted from the Sparse Probing metric of SAEBench ("SAEBench: A
Comprehensive Benchmark for Sparse Autoencoders in Language Model
Interpretability", https://arxiv.org/abs/2503.09532). SAEBench's central
criticism is that SAEs are usually scored with unsupervised proxy metrics
(L0, cross-entropy, KL, density) whose practical relevance is unclear;
sparse probing instead asks whether a small budget of SAE latents suffices
to linearly recover a downstream, document-level label.

The paper's core mechanism is kept intact: for a sweep of feature budgets
k, keep only the top-k SAE latents (ranked by mean absolute training
activation), fit a linear probe, score it on held-out rows, and summarize
accuracy across the sweep as a log-k normalized AUC, against a
matched-budget PCA baseline computed on the raw activations.

Adaptations for this repo (paper auxiliaries replaced by target-native
equivalents):

- sklearn logistic regression is replaced by a torch linear probe trained
  full-batch with Adam from a zero init (deterministic, no new dependency);
  the paper's binary dataset pairs generalize to any label count via a
  softmax probe.
- The paper's HuggingFace dataset pairs streamed through a hooked model are
  replaced by caller-supplied activation tensors, so the metric runs on any
  activations at the SAE input site (cached, streamed, or synthetic).
- The paper's benchmark harness (15 dataset pairs, wandb logging) is cut;
  results come back as a run_evals-style metric dict.

Row convention: within each label group, even-indexed rows train the probe
and odd-indexed rows are scored, so results are deterministic and need no
random seed.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from sae_lens.saes.sae import SAE


def _stack_and_split(
    tensors_by_label: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Stack label groups into train/score tensors of shape (n_rows, n_features).

    Even-indexed rows of each group go to the training split, odd-indexed
    rows to the scored split. Label ids follow the sorted label names.
    """
    if len(tensors_by_label) < 2:
        raise ValueError("probing needs activations for at least two labels")
    names = sorted(tensors_by_label)
    train_rows: list[torch.Tensor] = []
    train_ids: list[torch.Tensor] = []
    score_rows: list[torch.Tensor] = []
    score_ids: list[torch.Tensor] = []
    n_features: int | None = None
    for label_id, name in enumerate(names):
        group = tensors_by_label[name].detach().float()
        if group.ndim != 2:
            raise ValueError(
                f"label {name!r} must be a 2D (n_samples, n_features) tensor"
            )
        if group.shape[0] < 4:
            raise ValueError(f"label {name!r} needs at least 4 rows to split")
        if n_features is None:
            n_features = group.shape[1]
        elif group.shape[1] != n_features:
            raise ValueError("all labels must share the same feature dimension")
        train_group, score_group = group[0::2], group[1::2]
        train_rows.append(train_group)
        score_rows.append(score_group)
        train_ids.append(torch.full((train_group.shape[0],), label_id))
        score_ids.append(torch.full((score_group.shape[0],), label_id))
    return (
        torch.cat(train_rows),
        torch.cat(train_ids),
        torch.cat(score_rows),
        torch.cat(score_ids),
        names,
    )


def _standardize(
    train: torch.Tensor, score: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Z-score columns with training-split statistics only."""
    mean = train.mean(dim=0)
    std = train.std(dim=0)
    std = torch.where(std > 0, std, torch.ones_like(std))
    return (train - mean) / std, (score - mean) / std


def _probe_accuracy(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    score_x: torch.Tensor,
    score_y: torch.Tensor,
    probe_steps: int,
    probe_lr: float,
) -> float:
    """Fit a linear probe full-batch with Adam and return held-out accuracy."""
    n_classes = int(train_y.max().item()) + 1
    weight = torch.zeros((train_x.shape[1], n_classes), requires_grad=True)
    bias = torch.zeros(n_classes, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=probe_lr)
    for _ in range(probe_steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(train_x @ weight + bias, train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predictions = (score_x @ weight + bias).argmax(dim=1)
        return (predictions == score_y).float().mean().item()


def _resolve_k_values(requested: Sequence[int] | None, max_k: int) -> list[int]:
    """Default to powers of two up to max_k (plus max_k), or validate a sweep."""
    if max_k < 1:
        raise ValueError(
            "need at least one usable feature and more training rows than features"
        )
    if requested is None:
        ks = []
        k = 1
        while k < max_k:
            ks.append(k)
            k *= 2
        ks.append(max_k)
        return ks
    ks = sorted({int(k) for k in requested})
    if not ks or ks[0] < 1:
        raise ValueError("k_values must be positive integers")
    if ks[-1] > max_k:
        raise ValueError(
            f"k_values must not exceed {max_k} (limited by features and training rows)"
        )
    return ks


def _normalized_auc(k_values: Sequence[int], accuracies: Sequence[float]) -> float:
    """Mean accuracy across the sweep, trapezoid-integrated over log2(k)."""
    if len(k_values) == 1:
        return float(accuracies[0])
    xs = torch.log2(torch.tensor(k_values, dtype=torch.float64))
    ys = torch.tensor(accuracies, dtype=torch.float64)
    return float(torch.trapezoid(ys, xs) / (xs[-1] - xs[0]))


def sparse_probing_metrics(
    features_by_label: Mapping[str, torch.Tensor],
    k_values: Sequence[int] | None = None,
    *,
    probe_steps: int = 200,
    probe_lr: float = 0.1,
) -> dict[str, Any]:
    """Probe label accuracy from feature tensors at matched budgets of k features.

    Features are ranked once by mean absolute training activation; for each
    budget k only the top-k ranked features are standardized (training-split
    statistics) and handed to the probe.
    """
    train_x, train_y, score_x, score_y, names = _stack_and_split(features_by_label)
    train_z, score_z = _standardize(train_x, score_x)
    ks = _resolve_k_values(k_values, min(train_x.shape[1], train_x.shape[0] - 1))
    ranking = torch.argsort(train_x.abs().mean(dim=0), descending=True)
    accuracies = [
        _probe_accuracy(
            train_z[:, ranking[:k]],
            train_y,
            score_z[:, ranking[:k]],
            score_y,
            probe_steps,
            probe_lr,
        )
        for k in ks
    ]
    return {
        "labels": names,
        "k_values": ks,
        "probe_accuracies": accuracies,
        "probe_accuracy_auc": _normalized_auc(ks, accuracies),
    }


def pca_probing_baseline(
    activations_by_label: Mapping[str, torch.Tensor],
    k_values: Sequence[int] | None = None,
    *,
    probe_steps: int = 200,
    probe_lr: float = 0.1,
) -> dict[str, Any]:
    """Matched-budget baseline: probe the top-k principal components instead.

    The components come from the centered (but not rescaled) training
    activations, so the baseline sees the raw variance structure that PCA is
    meant to rank; the SAE and PCA arms are scored on the same rows at the
    same budgets k.
    """
    train_x, train_y, score_x, score_y, names = _stack_and_split(activations_by_label)
    ks = _resolve_k_values(k_values, min(train_x.shape[1], train_x.shape[0] - 1))
    mean = train_x.mean(dim=0)
    _, _, components = torch.linalg.svd(train_x - mean, full_matrices=False)
    train_projection = (train_x - mean) @ components.T
    score_projection = (score_x - mean) @ components.T
    accuracies = [
        _probe_accuracy(
            train_projection[:, :k],
            train_y,
            score_projection[:, :k],
            score_y,
            probe_steps,
            probe_lr,
        )
        for k in ks
    ]
    return {
        "labels": names,
        "k_values": ks,
        "probe_accuracies": accuracies,
        "probe_accuracy_auc": _normalized_auc(ks, accuracies),
    }


def run_sparse_probing_eval(
    sae: SAE[Any],
    activations_by_label: Mapping[str, torch.Tensor],
    k_values: Sequence[int] | None = None,
    *,
    probe_steps: int = 200,
    probe_lr: float = 0.1,
) -> dict[str, Any]:
    """Sparse-probing eval of an SAE against the PCA baseline, SAEBench-style.

    Encodes each label's activations with ``sae.encode``, probes the latents
    at each budget k, and repeats the same sweep over principal components
    of the raw activations. Returns a run_evals-style metric group
    "downstream_sparse_probing" whose ``sae_probe_accuracy_advantage`` is the
    SAE AUC minus the PCA AUC: positive means a sparse budget of SAE latents
    recovers the label better than equally many principal components.
    """
    with torch.no_grad():
        features_by_label = {
            name: sae.encode(activations)
            for name, activations in activations_by_label.items()
        }
    if k_values is None:
        n_train_rows = sum(
            (activations.shape[0] + 1) // 2
            for activations in activations_by_label.values()
        )
        k_values = _resolve_k_values(
            None, min(sae.cfg.d_sae, sae.cfg.d_in, n_train_rows - 1)
        )
    sae_metrics = sparse_probing_metrics(
        features_by_label, k_values, probe_steps=probe_steps, probe_lr=probe_lr
    )
    pca_metrics = pca_probing_baseline(
        activations_by_label, k_values, probe_steps=probe_steps, probe_lr=probe_lr
    )
    return {
        "downstream_sparse_probing": {
            "labels": sae_metrics["labels"],
            "k_values": sae_metrics["k_values"],
            "sae_probe_accuracies": sae_metrics["probe_accuracies"],
            "pca_probe_accuracies": pca_metrics["probe_accuracies"],
            "sae_probe_accuracy_auc": sae_metrics["probe_accuracy_auc"],
            "pca_probe_accuracy_auc": pca_metrics["probe_accuracy_auc"],
            "sae_probe_accuracy_advantage": (
                sae_metrics["probe_accuracy_auc"] - pca_metrics["probe_accuracy_auc"]
            ),
        }
    }
