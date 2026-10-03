"""
End-to-end Group-SAE workflow: measure pairwise angular distances between
hookpoint activation streams, select hook groups by AMAD, then train one SAE
per group of hooks with each batch drawn from a randomly selected member hook
of the group.

The way to run this with this command:
poetry run py.test benchmark/test_grouped_sae_training.py --profile-svg -s
"""

import torch
from datasets import Dataset
from transformer_lens.HookedTransformer import HookedRootModule

from sae_lens.config import LoggingConfig
from sae_lens.load_model import load_model
from sae_lens.multi_sae_training_runner import MultiSAETrainingRunnerConfig
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.grouped_sae_training_runner import GroupedSAETrainingRunner
from sae_lens.training.layer_grouping import (
    amad_score,
    angular_distance_matrix,
    collect_paired_activations,
    select_hook_groups,
)

HOOKS = [
    "blocks.0.hook_resid_pre",
    "blocks.0.hook_resid_post",
    "blocks.1.hook_resid_pre",
    "blocks.1.hook_resid_post",
]


def _multi_hook_store(
    model: HookedRootModule, dataset: Dataset, hooks: list[str], device: str
) -> ActivationsStore:
    d_in = model.cfg.d_model
    return ActivationsStore.from_config_multi_hook(
        model=model,
        dataset=dataset,
        hook_names=hooks,
        hook_d_ins={name: d_in for name in hooks},
        streaming=False,
        context_size=8,
        n_batches_in_buffer=2,
        total_training_tokens=1024,
        store_batch_size_prompts=4,
        train_batch_size_tokens=4,
        prepend_bos=True,
        normalize_activations="none",
        device=torch.device(device),
        dtype="float32",
        seqpos_slice=(None,),
        activations_mixing_fraction=0.0,
    )


def test_grouped_sae_training_workflow():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = load_model(
        "HookedTransformer",
        "tiny-stories-1M",
        device=device,
        model_from_pretrained_kwargs={"center_writing_weights": False},
    )
    dataset = Dataset.from_list(
        [{"text": f"the quick brown fox {i} jumps over"} for i in range(200)]
    )
    d_in = model.cfg.d_model

    # Phase 1: measure inter-hook angular distances and let AMAD pick groups.
    measure_store = _multi_hook_store(model, dataset, HOOKS, device)
    paired = collect_paired_activations(
        measure_store.get_multi_hook_data_loader(), HOOKS, n_batches=8
    )
    distances = angular_distance_matrix(paired)
    groups = select_hook_groups(paired, threshold=0.2)
    print(f"AMAD-selected groups: {groups}")

    assert sorted(hook for group in groups for hook in group) == sorted(HOOKS)
    indices = [[HOOKS.index(hook) for hook in group] for group in groups]
    assert amad_score(distances, indices) < 0.2

    # Phase 2: train one SAE per fixed pair of layer hooks (two SAEs cover
    # four hookpoints, halving the SAE count of one-SAE-per-hook training).
    hook_groups = {
        "layer0": ["blocks.0.hook_resid_pre", "blocks.0.hook_resid_post"],
        "layer1": ["blocks.1.hook_resid_pre", "blocks.1.hook_resid_post"],
    }
    cfg = MultiSAETrainingRunnerConfig(
        saes={
            "layer0": StandardTrainingSAEConfig(
                d_in=d_in,
                d_sae=64,
                l1_coefficient=1e-3,
                decoder_init_norm=0.1,
                normalize_activations="none",
                dtype="float32",
                device=device,
            ),
            "layer1": StandardTrainingSAEConfig(
                d_in=d_in,
                d_sae=64,
                l1_coefficient=1e-3,
                decoder_init_norm=0.1,
                normalize_activations="none",
                dtype="float32",
                device=device,
            ),
        },
        hook_names={name: hooks[0] for name, hooks in hook_groups.items()},
        model_name="tiny-stories-1M",
        dataset_path="placeholder",  # override_dataset is used
        streaming=False,
        context_size=8,
        n_batches_in_buffer=2,
        training_tokens=512,
        store_batch_size_prompts=4,
        train_batch_size_tokens=4,
        prepend_bos=True,
        device=device,
        dtype="float32",
        seqpos_slice=(None,),
        activations_mixing_fraction=0.0,
        lr=1e-3,
        logger=LoggingConfig(log_to_wandb=False),
    )
    runner = GroupedSAETrainingRunner(
        cfg, hook_groups, override_model=model, override_dataset=dataset
    )
    saes = runner.run()

    assert set(saes) == set(hook_groups)
    assert len(saes) < len(HOOKS)
    for name, hooks in hook_groups.items():
        assert saes[name].cfg.metadata.hook_group == hooks
        assert saes[name].cfg.metadata.hook_name == hooks[0]
        assert saes[name].W_dec.abs().sum().item() > 0

    # Reconstruction sanity check per member hook (the runner's store has
    # already been exhausted by fit(), so build a fresh one).
    eval_store = _multi_hook_store(model, dataset, HOOKS, device)
    batch = next(eval_store.get_multi_hook_data_loader())
    for name, hooks in hook_groups.items():
        sae = saes[name]
        for hook in hooks:
            activations = batch[hook]
            recon = sae(activations)
            per_sample_mse = (recon - activations).pow(2).sum(-1).mean()
            baseline = activations.pow(2).sum(-1).mean()
            assert per_sample_mse < baseline, (
                f"{name}/{hook}: reconstruction worse than zero baseline: "
                f"{per_sample_mse.item()} vs {baseline.item()}"
            )
