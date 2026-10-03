"""
Public entrypoint for training one SAE per group of hookpoints from a shared
LLM forward pass.

Use `GroupedSAETrainingRunner(cfg, hook_groups).run()` to train one SAE per
group of similar hookpoints instead of one SAE per hookpoint, adapting
"Group-SAE: Efficient Training of Sparse Autoencoders for Large Language
Models via Layer Groups" (https://arxiv.org/abs/2410.21508) onto the
multi-hook training stack: every SAE's batch is drawn from a randomly
selected member hook of its group on each step, so a single SAE learns a
dictionary shared by all the hooks it covers. Which hooks should share an
SAE can be decided with the AMAD grouping in `sae_lens.training.layer_grouping`.

Reuses `MultiSAETrainingRunnerConfig` (one entry in `saes` per group,
`hook_names` mapping each group to a representative member hook) and
delegates training to `MultiSAETrainer`, so every group SAE gets the same
per-SAE optimizer, learning-rate schedule, activation scaler and sparsity
tracking as in single-hook training.

V1 limitations:

    - No wandb logging, evals, checkpointing, resume or output saving (the
      underlying `MultiSAETrainer` supports them; wire them up in a follow-up
      if needed).
    - No prefetching of LLM batches (`prefetch_llm_batches` is ignored).
    - `sae.cfg.metadata.hook_name` records the group's representative hook;
      the full group is stored in `sae.cfg.metadata.hook_group`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformer_lens.HookedTransformer import HookedRootModule

from sae_lens import logger
from sae_lens.config import HfDataset
from sae_lens.load_model import load_model
from sae_lens.multi_sae_training_runner import MultiSAETrainingRunnerConfig
from sae_lens.registry import get_sae_training_class
from sae_lens.saes.sae import TrainingSAE
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.multi_sae_trainer import MultiSAETrainer
from sae_lens.training.types import MultiHookDataProvider
from sae_lens.util import get_special_token_ids


class GroupedHookDataProvider:
    """
    Routes a multi-hook activation stream to one SAE per group of hooks.

    Every step pulls one multi-hook batch from `source` and yields, per SAE
    name, the slice of a randomly selected member hook of that SAE's group —
    the layer-mixing rule of Group-SAE.
    """

    def __init__(
        self,
        source: MultiHookDataProvider,
        hook_groups: Mapping[str, Sequence[str]],
    ) -> None:
        for name, hooks in hook_groups.items():
            if not hooks:
                raise ValueError(f"hook group {name!r} must not be empty")
        self._source = source
        self._hook_groups = {name: list(hooks) for name, hooks in hook_groups.items()}

    def __iter__(self) -> GroupedHookDataProvider:
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        multi_batch = next(self._source)
        return {
            name: multi_batch[hooks[int(torch.randint(len(hooks), (1,)))]]
            for name, hooks in self._hook_groups.items()
        }


class GroupedSAETrainingRunner:
    """
    Orchestrator that wires the model, multi-hook ActivationsStore, one
    `TrainingSAE` per hook group, and `MultiSAETrainer`. Public surface:
    `__init__(cfg, hook_groups, ...)` and `run() -> dict[str, TrainingSAE]`.
    """

    cfg: MultiSAETrainingRunnerConfig
    hook_groups: dict[str, list[str]]
    model: HookedRootModule
    saes: dict[str, TrainingSAE[Any]]
    activations_store: ActivationsStore
    trainer: MultiSAETrainer

    def __init__(
        self,
        cfg: MultiSAETrainingRunnerConfig,
        hook_groups: Mapping[str, Sequence[str]],
        override_dataset: HfDataset | None = None,
        override_model: HookedRootModule | None = None,
        override_saes: dict[str, TrainingSAE[Any]] | None = None,
    ) -> None:
        if set(hook_groups) != set(cfg.saes):
            raise ValueError(
                f"hook_groups must have the same keys as cfg.saes; got "
                f"{sorted(hook_groups)} vs {sorted(cfg.saes)}"
            )
        all_hooks: list[str] = []
        seen_hooks: set[str] = set()
        for name, hooks in hook_groups.items():
            if not hooks:
                raise ValueError(
                    f"hook group {name!r} must not be empty; every SAE needs at "
                    "least one hook to train on"
                )
            if cfg.hook_names_per_sae[name] not in hooks:
                raise ValueError(
                    f"cfg.hook_names[{name!r}] must be a member of hook group "
                    f"{name!r}; got {cfg.hook_names_per_sae[name]!r} not in {list(hooks)}"
                )
            for hook in hooks:
                if hook in seen_hooks:
                    raise ValueError(
                        f"hook {hook!r} appears in more than one group; groups "
                        "must partition the hooks"
                    )
                seen_hooks.add(hook)
            all_hooks.extend(hooks)

        if override_saes is not None:
            extra = set(override_saes) - set(cfg.saes)
            missing = set(cfg.saes) - set(override_saes)
            if extra or missing:
                raise ValueError(
                    f"override_saes keys must match cfg.saes; extra: {sorted(extra)}; "
                    f"missing: {sorted(missing)}"
                )

        self.cfg = cfg
        self.hook_groups = {name: list(hooks) for name, hooks in hook_groups.items()}

        if override_dataset is not None:
            logger.warning(
                f"override_dataset overrides cfg.dataset_path={cfg.dataset_path!r}; "
                "this run will not be reproducible from configuration alone."
            )
        if override_model is not None:
            logger.warning(
                f"override_model overrides cfg.model_name={cfg.model_name!r}; "
                "this run will not be reproducible from configuration alone."
            )

        llm_device = cfg.llm_device
        assert llm_device is not None  # set in __post_init__

        self.model = (
            override_model
            if override_model is not None
            else load_model(
                cfg.model_class_name,
                cfg.model_name,
                device=llm_device,
                model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
                hook_names=all_hooks,
            )
        )

        # Per-hook store parameters derived from each hook's group SAE, so the
        # single shared forward pass captures every member hook.
        hook_d_ins = {
            hook: cfg.saes[name].d_in
            for name, hooks in self.hook_groups.items()
            for hook in hooks
        }
        hook_head_indices = {
            hook: cfg.hook_head_indices_per_sae[name]
            for name, hooks in self.hook_groups.items()
            for hook in hooks
        }

        # The store does not apply normalization itself (each SAE's trainer
        # scales its own activations), so "none" keeps the store a no-op here.
        self.activations_store = ActivationsStore.from_config_multi_hook(
            model=self.model,
            dataset=override_dataset
            if override_dataset is not None
            else cfg.dataset_path,
            hook_names=all_hooks,
            hook_d_ins=hook_d_ins,
            hook_head_indices=hook_head_indices,
            streaming=cfg.streaming,
            context_size=cfg.context_size,
            n_batches_in_buffer=cfg.n_batches_in_buffer,
            total_training_tokens=cfg.training_tokens,
            store_batch_size_prompts=cfg.store_batch_size_prompts,
            train_batch_size_tokens=cfg.train_batch_size_tokens,
            prepend_bos=cfg.prepend_bos,
            normalize_activations="none",
            device=torch.device(cfg.act_store_device),  # type: ignore[arg-type]
            dtype=cfg.dtype,
            model_kwargs=cfg.model_kwargs,
            autocast_lm=cfg.autocast_lm,
            dataset_trust_remote_code=cfg.dataset_trust_remote_code,
            seqpos_slice=cfg.seqpos_slice,
            exclude_special_tokens=_resolve_exclude_special_tokens(
                cfg.exclude_special_tokens,
                self.model,
                torch.device(cfg.act_store_device),  # type: ignore[arg-type]
            ),
            disable_concat_sequences=cfg.disable_concat_sequences,
            sequence_separator_token=cfg.sequence_separator_token,
            activations_mixing_fraction=cfg.activations_mixing_fraction,
            use_chat_formatting=cfg.use_chat_formatting,
        )

        if override_saes is None:
            saes: dict[str, TrainingSAE[Any]] = {}
            for name, sae_cfg in cfg.saes.items():
                sae_class, _ = get_sae_training_class(sae_cfg.architecture())
                saes[name] = sae_class(sae_cfg)
            self.saes = saes
        else:
            self.saes = override_saes

        for sae in self.saes.values():
            sae.to(cfg.device)

        # The trainer looks up each SAE's batch in the yielded dict by
        # hook_names[name]; the grouped provider yields one batch per SAE
        # name, so each SAE is keyed by its own (group) name.
        self.trainer = MultiSAETrainer(
            cfg=cfg.to_sae_trainer_config(),
            saes=self.saes,
            hook_names={name: name for name in self.saes},
            data_provider=GroupedHookDataProvider(
                self.activations_store.get_multi_hook_data_loader(),
                self.hook_groups,
            ),
        )

    def run(self) -> dict[str, TrainingSAE[Any]]:
        self._set_sae_metadata()
        return self.trainer.fit()

    def _set_sae_metadata(self) -> None:
        for name, sae in self.saes.items():
            sae.cfg.metadata.dataset_path = self.cfg.dataset_path
            sae.cfg.metadata.hook_name = self.cfg.hook_names_per_sae[name]
            sae.cfg.metadata.hook_group = self.hook_groups[name]
            sae.cfg.metadata.model_name = self.cfg.model_name
            sae.cfg.metadata.model_class_name = self.cfg.model_class_name
            sae.cfg.metadata.hook_head_index = self.cfg.hook_head_indices_per_sae[name]
            sae.cfg.metadata.context_size = self.cfg.context_size
            sae.cfg.metadata.seqpos_slice = self.cfg.seqpos_slice
            sae.cfg.metadata.model_from_pretrained_kwargs = (
                self.cfg.model_from_pretrained_kwargs
            )
            sae.cfg.metadata.prepend_bos = self.cfg.prepend_bos
            sae.cfg.metadata.exclude_special_tokens = self.cfg.exclude_special_tokens
            sae.cfg.metadata.sequence_separator_token = (
                self.cfg.sequence_separator_token
            )
            sae.cfg.metadata.disable_concat_sequences = (
                self.cfg.disable_concat_sequences
            )


def _resolve_exclude_special_tokens(
    raw: bool | list[int],
    model: HookedRootModule,
    device: torch.device,
) -> torch.Tensor | None:
    if raw is False:
        return None
    ids = (
        list(get_special_token_ids(model.tokenizer))  # type: ignore[arg-type]
        if raw is True
        else list(raw)
    )
    return torch.tensor(ids, dtype=torch.long, device=device)
