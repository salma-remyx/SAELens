from collections.abc import Mapping

import pytest
import torch
from datasets import Dataset
from transformer_lens import HookedTransformer

from sae_lens.config import LoggingConfig
from sae_lens.multi_sae_training_runner import MultiSAETrainingRunnerConfig
from sae_lens.saes.sae import TrainingSAEConfig
from sae_lens.saes.standard_sae import StandardTrainingSAE, StandardTrainingSAEConfig
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
from sae_lens.training.grouped_sae_training_runner import (
    GroupedHookDataProvider,
    GroupedSAETrainingRunner,
)
from tests.helpers import TINYSTORIES_MODEL, load_model_cached

HOOK_GROUPS = {
    "resid_pre_group": ["blocks.0.hook_resid_pre", "blocks.1.hook_resid_pre"],
    "resid_post_group": ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"],
}


@pytest.fixture
def ts_model() -> HookedTransformer:
    return load_model_cached(TINYSTORIES_MODEL)


@pytest.fixture
def dataset() -> Dataset:
    return Dataset.from_list(
        [{"text": f"the quick brown fox {i} jumps over"} for i in range(200)]
    )


def _build_cfg(
    *,
    saes: Mapping[str, TrainingSAEConfig],
    hook_names: dict[str, str],
    training_tokens: int = 32,
) -> MultiSAETrainingRunnerConfig:
    return MultiSAETrainingRunnerConfig(
        saes=saes,
        hook_names=hook_names,
        model_name=TINYSTORIES_MODEL,
        dataset_path="placeholder",  # override_dataset is used
        streaming=False,
        context_size=8,
        n_batches_in_buffer=2,
        training_tokens=training_tokens,
        store_batch_size_prompts=4,
        train_batch_size_tokens=4,
        prepend_bos=True,
        device="cpu",
        dtype="float32",
        seqpos_slice=(None,),
        activations_mixing_fraction=0.0,
        lr=1e-3,
        logger=LoggingConfig(log_to_wandb=False),
    )


def _std_sae_cfg(d_in: int) -> StandardTrainingSAEConfig:
    return StandardTrainingSAEConfig(
        d_in=d_in,
        d_sae=32,
        l1_coefficient=1e-3,
        decoder_init_norm=0.1,
        normalize_activations="none",
        dtype="float32",
        device="cpu",
    )


def test_grouped_hook_provider_samples_each_member_hook_uniformly():
    n_steps = 2000
    source = iter(
        [
            {
                "h0": torch.full((4, 3), 0.0),
                "h1": torch.full((4, 3), 1.0),
                "h2": torch.full((4, 3), 2.0),
            }
            for _ in range(n_steps)
        ]
    )
    provider = GroupedHookDataProvider(source, {"a": ["h0", "h1"], "b": ["h2"]})

    h0_steps = 0
    for _ in range(n_steps):
        batch = next(provider)
        assert set(batch) == {"a", "b"}
        # the singleton group always routes to its only hook
        assert torch.equal(batch["b"], torch.full((4, 3), 2.0))
        if batch["a"][0, 0].item() == 0.0:
            h0_steps += 1
        else:
            assert torch.equal(batch["a"], torch.full((4, 3), 1.0))

    assert h0_steps == pytest.approx(n_steps / 2, abs=n_steps * 0.05)


def test_grouped_hook_provider_rejects_empty_group():
    source = iter([{"h0": torch.zeros(4, 3)}])
    with pytest.raises(ValueError, match="must not be empty"):
        GroupedHookDataProvider(source, {"a": []})


def test_grouped_sae_runner_rejects_mismatched_group_keys():
    cfg = _build_cfg(
        saes={"a": _std_sae_cfg(64), "b": _std_sae_cfg(64)},
        hook_names={"a": "blocks.0.hook_resid_pre", "b": "blocks.1.hook_resid_pre"},
    )
    with pytest.raises(ValueError, match="same keys as cfg.saes"):
        GroupedSAETrainingRunner(cfg, {"a": ["blocks.0.hook_resid_pre"]})


def test_grouped_sae_runner_rejects_representative_hook_outside_group():
    cfg = _build_cfg(
        saes={"a": _std_sae_cfg(64), "b": _std_sae_cfg(64)},
        hook_names={"a": "blocks.0.hook_resid_pre", "b": "blocks.1.hook_resid_pre"},
    )
    with pytest.raises(ValueError, match="must be a member of hook group"):
        GroupedSAETrainingRunner(
            cfg,
            {
                "a": ["blocks.0.hook_resid_post"],
                "b": ["blocks.1.hook_resid_pre"],
            },
        )


def test_grouped_sae_runner_rejects_hook_in_multiple_groups():
    cfg = _build_cfg(
        saes={"a": _std_sae_cfg(64), "b": _std_sae_cfg(64)},
        hook_names={"a": "blocks.0.hook_resid_pre", "b": "blocks.1.hook_resid_pre"},
    )
    with pytest.raises(ValueError, match="more than one group"):
        GroupedSAETrainingRunner(
            cfg,
            {
                "a": ["blocks.0.hook_resid_pre", "blocks.1.hook_resid_pre"],
                "b": ["blocks.1.hook_resid_pre", "blocks.1.hook_resid_post"],
            },
        )


def test_grouped_sae_runner_rejects_mismatched_override_saes():
    cfg = _build_cfg(
        saes={"a": _std_sae_cfg(64), "b": _std_sae_cfg(64)},
        hook_names={"a": "blocks.0.hook_resid_pre", "b": "blocks.1.hook_resid_pre"},
    )
    bad_override = {"a": StandardTrainingSAE(_std_sae_cfg(64))}
    with pytest.raises(ValueError, match="override_saes keys must match"):
        GroupedSAETrainingRunner(
            cfg,
            {
                "a": ["blocks.0.hook_resid_pre"],
                "b": ["blocks.1.hook_resid_pre"],
            },
            override_saes={**bad_override, "c": bad_override["a"]},
        )


def test_grouped_sae_runner_trains_one_sae_per_hook_group(
    ts_model: HookedTransformer, dataset: Dataset
):
    d_in = ts_model.cfg.d_model
    cfg = _build_cfg(
        saes={
            "resid_pre_group": _std_sae_cfg(d_in),
            "resid_post_group": TopKTrainingSAEConfig(
                d_in=d_in,
                d_sae=32,
                k=4,
                decoder_init_norm=0.1,
                normalize_activations="none",
                dtype="float32",
                device="cpu",
            ),
        },
        hook_names={name: hooks[0] for name, hooks in HOOK_GROUPS.items()},
    )
    runner = GroupedSAETrainingRunner(
        cfg, HOOK_GROUPS, override_model=ts_model, override_dataset=dataset
    )
    saes = runner.run()

    assert set(saes) == set(HOOK_GROUPS)
    for name, hooks in HOOK_GROUPS.items():
        assert saes[name].cfg.metadata.hook_group == hooks
        assert saes[name].cfg.metadata.hook_name == hooks[0]
        assert saes[name].W_dec.abs().sum().item() > 0
    # topk with k=4 caps the mean l0 at 4
    assert 0 < saes["resid_post_group"].cfg.metadata["l0"] <= 4
