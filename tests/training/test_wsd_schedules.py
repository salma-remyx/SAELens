import math

import pytest
import torch
from torch.optim import Adam

from sae_lens.training.wsd_schedules import get_wsd_lr_scheduler, wsd_multiplier

LR = 0.1


def test_wsd_multiplier_warms_up_linearly():
    multiplier = wsd_multiplier(
        training_steps=12,
        warm_up_steps=4,
        decay_steps=2,
        decay_exponent=1.0,
        lr_end_ratio=0.0,
    )
    assert [multiplier(step) for step in range(4)] == [0.25, 0.5, 0.75, 1.0]


def test_wsd_multiplier_holds_peak_through_the_stable_phase():
    multiplier = wsd_multiplier(
        training_steps=12,
        warm_up_steps=2,
        decay_steps=4,
        decay_exponent=1.0,
        lr_end_ratio=0.0,
    )
    # the stable phase runs from the end of warmup to the start of the decay
    assert [multiplier(step) for step in range(2, 8)] == [1.0] * 6


def test_wsd_multiplier_linear_decay_reaches_zero():
    multiplier = wsd_multiplier(
        training_steps=12,
        warm_up_steps=0,
        decay_steps=4,
        decay_exponent=1.0,
        lr_end_ratio=0.0,
    )
    assert [multiplier(step) for step in range(8, 13)] == [1.0, 0.75, 0.5, 0.25, 0.0]


def test_wsd_multiplier_sqrt_decay_shape():
    multiplier = wsd_multiplier(
        training_steps=12,
        warm_up_steps=0,
        decay_steps=4,
        decay_exponent=0.5,
        lr_end_ratio=0.0,
    )
    expected = [1.0, math.sqrt(0.75), math.sqrt(0.5), math.sqrt(0.25), 0.0]
    assert [multiplier(step) for step in range(8, 13)] == pytest.approx(expected)


def test_wsd_multiplier_sqrt_decay_stays_above_linear_decay():
    linear = wsd_multiplier(
        training_steps=12,
        warm_up_steps=0,
        decay_steps=4,
        decay_exponent=1.0,
        lr_end_ratio=0.0,
    )
    sqrt = wsd_multiplier(
        training_steps=12,
        warm_up_steps=0,
        decay_steps=4,
        decay_exponent=0.5,
        lr_end_ratio=0.0,
    )
    multipliers = [(linear(step), sqrt(step)) for step in range(12)]
    assert all(linear_value <= sqrt_value for linear_value, sqrt_value in multipliers)
    # 1-sqrt spends more of the decay near the peak learning rate
    assert linear(10) == pytest.approx(0.5)
    assert sqrt(10) == pytest.approx(math.sqrt(0.5))


def test_wsd_multiplier_holds_lr_end_ratio_after_the_decay():
    multiplier = wsd_multiplier(
        training_steps=12,
        warm_up_steps=0,
        decay_steps=4,
        decay_exponent=1.0,
        lr_end_ratio=0.2,
    )
    assert [multiplier(step) for step in range(8, 13)] == pytest.approx(
        [1.0, 0.8, 0.6, 0.4, 0.2]
    )


def test_wsd_multiplier_schedule_jump_returns_to_the_peak_after_a_decay():
    multiplier = wsd_multiplier(
        training_steps=20,
        warm_up_steps=2,
        decay_steps=5,
        decay_exponent=1.0,
        lr_end_ratio=0.0,
        num_cycles=2,
    )
    # the first cycle decays over steps 5..9
    assert [multiplier(step) for step in range(5, 10)] == pytest.approx(
        [1.0, 0.8, 0.6, 0.4, 0.2]
    )
    # the learning rate jumps straight back to the peak for the second cycle
    assert [multiplier(step) for step in range(10, 15)] == [1.0] * 5
    # the second cycle decays over the same shape, ending at the same value
    assert [multiplier(step) for step in range(15, 20)] == pytest.approx(
        [1.0, 0.8, 0.6, 0.4, 0.2]
    )


def test_wsd_multiplier_cycles_repeat_the_same_decay_curve():
    two_cycles = wsd_multiplier(
        training_steps=20,
        warm_up_steps=2,
        decay_steps=5,
        decay_exponent=1.0,
        lr_end_ratio=0.0,
        num_cycles=2,
    )
    one_cycle = wsd_multiplier(
        training_steps=10,
        warm_up_steps=2,
        decay_steps=5,
        decay_exponent=1.0,
        lr_end_ratio=0.0,
        num_cycles=1,
    )
    first_cycle = [two_cycles(step) for step in range(2, 10)]
    assert first_cycle == pytest.approx([one_cycle(step) for step in range(2, 10)])
    # the warmup runs once per run, not once per cycle
    assert two_cycles(10) == 1.0
    # the second cycle replays the first cycle's stable and decay phases, so each
    # cycle boundary is a decayed checkpoint reached the same way
    assert [two_cycles(step) for step in range(12, 20)] == pytest.approx(first_cycle)


def test_wsd_multiplier_rejects_decay_longer_than_the_cycle():
    with pytest.raises(ValueError, match="must fit within a cycle"):
        wsd_multiplier(
            training_steps=10,
            warm_up_steps=4,
            decay_steps=8,
            decay_exponent=1.0,
            lr_end_ratio=0.0,
        )


def test_wsd_multiplier_rejects_zero_decay_steps():
    with pytest.raises(ValueError, match="decay_steps must be > 0"):
        wsd_multiplier(
            training_steps=10,
            warm_up_steps=2,
            decay_steps=0,
            decay_exponent=1.0,
            lr_end_ratio=0.0,
        )


def test_get_wsd_lr_scheduler_steps_the_learning_rate():
    optimizer = Adam([torch.tensor(1.0)], lr=LR)
    scheduler = get_wsd_lr_scheduler(
        "wsdsqrt",
        optimizer,
        training_steps=12,
        warm_up_steps=0,
        decay_steps=4,
        lr_end=0.0,
        lr=LR,
        num_cycles=1,
    )
    assert scheduler.get_last_lr() == [LR]
    for _ in range(8):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr() == [LR]
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == [pytest.approx(LR * math.sqrt(0.75))]
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr() == [pytest.approx(LR * math.sqrt(0.25))]
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == [0.0]


def test_get_wsd_lr_scheduler_state_dict_round_trips():
    optimizer = Adam([torch.tensor(1.0)], lr=LR)
    scheduler = get_wsd_lr_scheduler(
        "wsdlinear",
        optimizer,
        training_steps=12,
        warm_up_steps=4,
        decay_steps=4,
        lr_end=0.0,
        lr=LR,
        num_cycles=1,
    )
    for _ in range(7):
        optimizer.step()
        scheduler.step()

    resumed_optimizer = Adam([torch.tensor(1.0)], lr=LR)
    resumed = get_wsd_lr_scheduler(
        "wsdlinear",
        resumed_optimizer,
        training_steps=12,
        warm_up_steps=4,
        decay_steps=4,
        lr_end=0.0,
        lr=LR,
        num_cycles=1,
    )
    resumed.load_state_dict(scheduler.state_dict())
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        resumed_optimizer.step()
        resumed.step()
        assert resumed.get_last_lr() == pytest.approx(scheduler.get_last_lr())


def test_get_wsd_lr_scheduler_rejects_unknown_name():
    optimizer = Adam([torch.tensor(1.0)], lr=LR)
    with pytest.raises(ValueError, match="Unsupported WSD scheduler: wsdother"):
        get_wsd_lr_scheduler(
            "wsdother",
            optimizer,
            training_steps=12,
            warm_up_steps=2,
            decay_steps=4,
            lr_end=0.0,
            lr=LR,
            num_cycles=1,
        )


def test_get_wsd_lr_scheduler_rejects_lr_end_above_lr():
    optimizer = Adam([torch.tensor(1.0)], lr=LR)
    with pytest.raises(ValueError, match="lr_end must be in"):
        get_wsd_lr_scheduler(
            "wsdlinear",
            optimizer,
            training_steps=12,
            warm_up_steps=2,
            decay_steps=4,
            lr_end=LR * 2,
            lr=LR,
            num_cycles=1,
        )
