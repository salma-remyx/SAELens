"""
Warmup-stable-decay (WSD) learning-rate schedules.

Adapted from "Scaling Laws and Compute-Optimal Training Beyond Fixed Training
Durations" (Hägele et al., 2024, https://arxiv.org/abs/2405.18392), which
replaces cosine annealing with a warmup-stable-decay schedule so that a run can
be decayed at several different training durations, or extended past its
original horizon, without restarting from scratch.

The schedule is expressed as a multiplier of the peak learning rate:

- Warmup: linear ramp from 0 to 1 across `warm_up_steps`.
- Stable: constant at 1.
- Decay: from 1 down to `lr_end / lr` across `decay_steps`, following
  `(1 - t / decay_steps) ** decay_exponent`. An exponent of 1 is the paper's
  linear decay, 0.5 its 1-sqrt decay, which stays near the peak learning rate
  for longer and is the shape the paper recommends for short decay phases.

With `num_cycles > 1` the stable + decay pair repeats: once a decay completes
the learning rate jumps back to the peak and decays again ("schedule jumping"
in the paper). Every cycle boundary then holds a fully decayed checkpoint, so a
single run yields several usable SAEs at different training durations. The
paper re-warms the learning rate over roughly 10% of the remaining steps after
each jump; we jump straight back to the peak so the schedule stays a pure
function of the step count, which is what the trainer's checkpoint save/load
path round-trips.
"""

from collections.abc import Callable

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Decay shape for each WSD scheduler name: the exponent applied to the fraction
# of the decay phase that remains.
WSD_DECAY_EXPONENTS: dict[str, float] = {
    "wsdlinear": 1.0,
    "wsdsqrt": 0.5,
}


def wsd_multiplier(
    training_steps: int,
    warm_up_steps: int,
    decay_steps: int,
    decay_exponent: float,
    lr_end_ratio: float,
    num_cycles: int = 1,
) -> Callable[[int], float]:
    """
    Returns a `step -> multiplier` function for the WSD schedule family.

    Args:
        training_steps (int): Total number of training steps.
        warm_up_steps (int): Number of linear warmup steps at the start of the run.
        decay_steps (int): Length of each decay phase.
        decay_exponent (float): Shape of the decay, 1.0 for linear and 0.5 for 1-sqrt.
        lr_end_ratio (float): Learning rate held after a decay, relative to the peak.
        num_cycles (int): Number of stable + decay cycles; above 1 the learning rate
            jumps back to the peak after each decay.
    """
    cycle_steps = training_steps // max(num_cycles, 1)
    if decay_steps <= 0:
        raise ValueError(f"decay_steps must be > 0, got {decay_steps}.")
    if warm_up_steps + decay_steps > cycle_steps:
        raise ValueError(
            f"warmup ({warm_up_steps}) and decay ({decay_steps}) steps must fit "
            f"within a cycle of {cycle_steps} steps."
        )

    def multiplier(step: int) -> float:
        if step < warm_up_steps:
            return (step + 1) / warm_up_steps
        cycle = min(step // cycle_steps, num_cycles - 1)
        # each cycle ends with a decay phase of `decay_steps`
        decay_offset = step - (cycle_steps * (cycle + 1) - decay_steps)
        if decay_offset < 0:
            return 1.0
        if decay_offset >= decay_steps:
            return lr_end_ratio
        remaining = 1.0 - decay_offset / decay_steps
        return lr_end_ratio + (1.0 - lr_end_ratio) * remaining**decay_exponent

    return multiplier


def get_wsd_lr_scheduler(
    scheduler_name: str,
    optimizer: optim.Optimizer,
    training_steps: int,
    warm_up_steps: int,
    decay_steps: int,
    lr_end: float,
    lr: float,
    num_cycles: int,
) -> lr_scheduler.LRScheduler:
    """
    Builds a warmup-stable-decay scheduler selected by name.

    Args:
        scheduler_name (str): "wsdlinear" for linear decay, "wsdsqrt" for 1-sqrt decay.
        optimizer (optim.Optimizer): Optimizer to schedule.
        training_steps (int): Total number of training steps.
        warm_up_steps (int): Number of linear warmup steps.
        decay_steps (int): Length of each decay phase.
        lr_end (float): Learning rate held after a decay phase.
        lr (float): Peak learning rate.
        num_cycles (int): Number of stable + decay cycles (schedule jumps when > 1).
    """
    if scheduler_name not in WSD_DECAY_EXPONENTS:
        raise ValueError(f"Unsupported WSD scheduler: {scheduler_name}")
    if not 0.0 <= lr_end <= lr:
        raise ValueError(f"lr_end must be in [0, lr], got {lr_end} with lr={lr}.")
    return lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=wsd_multiplier(
            training_steps=training_steps,
            warm_up_steps=warm_up_steps,
            decay_steps=decay_steps,
            decay_exponent=WSD_DECAY_EXPONENTS[scheduler_name],
            lr_end_ratio=lr_end / lr,
            num_cycles=num_cycles,
        ),
    )
