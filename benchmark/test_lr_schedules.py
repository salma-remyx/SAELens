"""
Benchmark for the learning-rate schedules a training config produces.

Builds the scheduler the same way SAETrainer does and steps it across a whole
run, so a config field that stops reaching the scheduler shows up here.
"""

import pytest
import torch
from torch.optim import Adam

from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig
from sae_lens.training.optim import get_lr_scheduler

LR = 1e-3


def _lr_curve(cfg: LanguageModelSAERunnerConfig) -> list[float]:
    assert cfg.lr_end is not None  # set in config post-init
    optimizer = Adam([torch.zeros(1, cfg.sae.d_in)], lr=cfg.lr)
    scheduler = get_lr_scheduler(
        cfg.lr_scheduler_name,
        lr=cfg.lr,
        optimizer=optimizer,
        warm_up_steps=cfg.lr_warm_up_steps,
        decay_steps=cfg.lr_decay_steps,
        training_steps=cfg.total_training_steps,
        lr_end=cfg.lr_end,
        num_cycles=cfg.n_restart_cycles,
    )
    lrs = [scheduler.get_last_lr()[0]]
    for _ in range(cfg.total_training_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    return lrs


def _runner_config(
    total_training_steps: int,
    lr_scheduler_name: str,
    lr_warm_up_steps: int,
    lr_decay_steps: int,
    n_restart_cycles: int,
) -> LanguageModelSAERunnerConfig:
    return LanguageModelSAERunnerConfig(
        sae=StandardTrainingSAEConfig(d_sae=8, d_in=8),
        lr=LR,
        lr_end=0.0,
        lr_scheduler_name=lr_scheduler_name,
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_decay_steps,
        n_restart_cycles=n_restart_cycles,
        train_batch_size_tokens=8,
        training_tokens=total_training_steps * 8,
        logger=LoggingConfig(log_to_wandb=False),
    )


def test_wsd_sqrt_lr_curve_from_runner_config():
    cfg = _runner_config(
        total_training_steps=100,
        lr_scheduler_name="wsdsqrt",
        lr_warm_up_steps=10,
        lr_decay_steps=20,
        n_restart_cycles=1,
    )
    lrs = _lr_curve(cfg)
    print(f"peak lr: {max(lrs)}, final lr: {lrs[-1]}")
    assert lrs[0] == pytest.approx(LR / 10)
    assert max(lrs) == pytest.approx(LR)
    # the stable phase runs from the end of warmup to the start of the decay
    assert lrs[10:80] == pytest.approx([LR] * 70)
    # 1-sqrt decay: halfway through the decay the lr is sqrt(1/2) of the peak
    assert lrs[90] == pytest.approx(LR * 0.5**0.5)
    assert lrs[100] == pytest.approx(0.0)


def test_wsd_schedule_jump_lr_curve_from_runner_config():
    cfg = _runner_config(
        total_training_steps=100,
        lr_scheduler_name="wsdlinear",
        lr_warm_up_steps=10,
        lr_decay_steps=20,
        n_restart_cycles=2,
    )
    lrs = _lr_curve(cfg)
    print(f"lr at the two cycle boundaries: {lrs[49]}, {lrs[99]}")
    # both cycles decay over the last 20 steps of their 50-step cycle
    assert lrs[30:50] == pytest.approx(lrs[80:100])
    assert lrs[49] == pytest.approx(LR / 20)
    # after the first decay the learning rate jumps back to the peak
    assert lrs[50] == pytest.approx(LR)
    assert lrs[99] == pytest.approx(LR / 20)
