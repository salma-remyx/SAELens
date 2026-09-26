"""Benchmark: SAE robustness under model pruning, scored by perturbation energy.

Runs the suggested experiment of arXiv:2608.25941 on top of the repo's eval
stack. For each requested block L, a fixed pretrained SAE reads
blocks.{L}.hook_resid_pre while the MLP output weights W_out of block L-1 are
pruned at matched sparsity with a magnitude mask and a Wanda mask. Because
resid_pre(L) receives mlp_out(L-1) directly, the SAE input shift is exactly
dW x, so eps^2 = tr(dW Sigma dW^T) is the mean squared-norm of the shift of
the SAE's own input activations. Each row reports eps^2 next to the SAE's eval metrics
under the pruned model: the method with lower eps^2 should degrade the SAE
less, and sweeping L exposes which layers' SAEs are most pruning-sensitive.
"""

import argparse
from typing import Any

import torch
from tabulate import tabulate
from transformer_lens import HookedTransformer

from sae_lens import SAE, ActivationsStore
from sae_lens.analysis.pruning_robustness import (
    compare_pruning_methods,
    magnitude_pruning_mask,
    wanda_pruning_mask,
)
from sae_lens.evals import EvalConfig, run_evals
from sae_lens.training.activation_scaler import ActivationScaler

torch.set_grad_enabled(False)


def collect_input_activations(
    model: HookedTransformer,
    activation_store: ActivationsStore,
    hook_name: str,
    n_batches: int,
    batch_size_prompts: int,
) -> torch.Tensor:
    """Stack (n_tokens, d_in) activations at hook_name from store batches."""
    chunks = []
    for _ in range(n_batches):
        tokens = activation_store.get_batch_tokens(batch_size_prompts)
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        chunks.append(cache[hook_name].reshape(-1, cache[hook_name].shape[-1]))
    return torch.cat(chunks, dim=0)


def eval_sae_under_mask(
    model: HookedTransformer,
    sae: SAE[Any],
    activation_store: ActivationsStore,
    weight: torch.nn.Parameter,
    mask: torch.Tensor | None,
    eval_config: EvalConfig,
) -> dict[str, Any]:
    """Run run_evals with weight pruned by mask (None leaves it dense)."""
    dense = weight.data.clone()
    if mask is not None:
        weight.data = dense * mask.to(dense.dtype)
    try:
        metrics, _ = run_evals(
            sae=sae,
            activation_store=activation_store,
            activation_scaler=ActivationScaler(),
            model=model,
            eval_config=eval_config,
        )
    finally:
        weight.data = dense
    return metrics


def bench_block(
    model: HookedTransformer,
    release: str,
    block: int,
    sparsity: float,
    n_calibration_batches: int,
    eval_config: EvalConfig,
    batch_size_prompts: int,
    device: str,
) -> list[dict[str, Any]]:
    """Prune block (block - 1).mlp.W_out and evaluate the SAE at block's hook."""
    if block < 1:
        raise ValueError(
            f"block must be >= 1 (block 0 has no block below it), got {block}"
        )
    sae = SAE.from_pretrained(release, f"blocks.{block}.hook_resid_pre", device=device)
    activation_store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        dataset=sae.cfg.metadata.dataset_path,
        streaming=True,
        store_batch_size_prompts=batch_size_prompts,
        device=device,
    )

    weight = model.blocks[block - 1].mlp.W_out
    activations = collect_input_activations(
        model,
        activation_store,
        f"blocks.{block - 1}.hook_post",
        n_calibration_batches,
        batch_size_prompts,
    )
    energies = compare_pruning_methods(weight.data, activations, sparsity)

    rows: list[dict[str, Any]] = []
    for method, mask in (
        ("dense", None),
        ("magnitude", magnitude_pruning_mask(weight.data, sparsity)),
        ("wanda", wanda_pruning_mask(weight.data, activations, sparsity)),
    ):
        metrics = eval_sae_under_mask(
            model, sae, activation_store, weight, mask, eval_config
        )
        rows.append(
            {
                "block": block,
                "method": method,
                "eps2": 0.0 if method == "dense" else energies[method].item(),
                "explained_variance": metrics.get("reconstruction_quality", {}).get(
                    "explained_variance"
                ),
                "mse": metrics.get("reconstruction_quality", {}).get("mse"),
                "l0": metrics.get("sparsity", {}).get("l0"),
                "ce_loss_score": metrics.get("model_performance_preservation", {}).get(
                    "ce_loss_score"
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--release", default="gpt2-small-res-jb")
    parser.add_argument(
        "--blocks",
        default="1,6,11",
        help="Comma-separated block indices whose hook_resid_pre SAEs are evaluated "
        "(each prunes the W_out of the block below)",
    )
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--n-calibration-batches", type=int, default=8)
    parser.add_argument("--n-eval-batches", type=int, default=5)
    parser.add_argument("--batch-size-prompts", type=int, default=8)
    parser.add_argument("--compute-ce-loss", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    eval_config = EvalConfig(
        batch_size_prompts=args.batch_size_prompts,
        compute_variance_metrics=True,
        compute_sparsity_metrics=True,
        compute_ce_loss=args.compute_ce_loss,
        n_eval_reconstruction_batches=args.n_eval_batches,
        n_eval_sparsity_variance_batches=args.n_eval_batches,
    )

    rows: list[dict[str, Any]] = []
    for block in [int(part) for part in args.blocks.split(",")]:
        rows.extend(
            bench_block(
                model,
                args.release,
                block,
                args.sparsity,
                args.n_calibration_batches,
                eval_config,
                args.batch_size_prompts,
                args.device,
            )
        )

    print(
        tabulate(
            rows,
            headers="keys",
            floatfmt=".5g",
        )
    )
    for block in [int(part) for part in args.blocks.split(",")]:
        block_rows = {
            row["method"]: row["eps2"] for row in rows if row["block"] == block
        }
        ratio = block_rows["magnitude"] / block_rows["wanda"]
        print(
            f"block {block}: magnitude eps^2 / wanda eps^2 = {ratio:.2f} "
            f"(activation-aware pruning perturbs the SAE input this many times less)"
        )


if __name__ == "__main__":
    main()
