"""Tests for the drift-triggered retrain controller. Pure and deterministic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.drift import DriftMonitor
from app.retrain import (
    DriftSignal,
    RetrainController,
    signal_from_monitor,
)


def _controller(min_consecutive=3, cooldown=5, min_samples=200):
    calls: list[DriftSignal] = []
    ctrl = RetrainController(
        retrain_fn=calls.append,
        min_consecutive=min_consecutive,
        cooldown=cooldown,
        min_samples=min_samples,
    )
    return ctrl, calls


def _drift(n=500):
    return DriftSignal(drifted=True, sample_count=n, psi=0.42, kl=0.2)


def _clean(n=500):
    return DriftSignal(drifted=False, sample_count=n, psi=0.01, kl=0.0)


def test_single_drift_does_not_trigger():
    ctrl, calls = _controller(min_consecutive=3)
    decision = ctrl.observe(_drift())
    assert decision.triggered is False
    assert decision.consecutive_drift == 1
    assert calls == []


def test_sustained_drift_triggers_once():
    ctrl, calls = _controller(min_consecutive=3, cooldown=5)
    d1 = ctrl.observe(_drift())
    d2 = ctrl.observe(_drift())
    d3 = ctrl.observe(_drift())
    assert (d1.triggered, d2.triggered, d3.triggered) == (False, False, True)
    assert len(calls) == 1
    assert ctrl.retrain_count == 1
    assert "sustained drift" in d3.reason


def test_clean_evaluation_resets_streak():
    ctrl, calls = _controller(min_consecutive=3)
    ctrl.observe(_drift())
    ctrl.observe(_drift())
    reset = ctrl.observe(_clean())
    assert reset.consecutive_drift == 0
    # Two more drifts should NOT trigger yet — the streak restarted.
    ctrl.observe(_drift())
    again = ctrl.observe(_drift())
    assert again.triggered is False
    assert calls == []


def test_insufficient_samples_never_advances_streak():
    ctrl, calls = _controller(min_consecutive=2, min_samples=200)
    d1 = ctrl.observe(DriftSignal(drifted=True, sample_count=50))
    d2 = ctrl.observe(DriftSignal(drifted=True, sample_count=50))
    assert (d1.triggered, d2.triggered) == (False, False)
    assert "insufficient samples" in d1.reason
    assert calls == []


def test_cooldown_suppresses_immediate_retrigger():
    ctrl, calls = _controller(min_consecutive=2, cooldown=3)
    ctrl.observe(_drift())
    fired = ctrl.observe(_drift())
    assert fired.triggered is True

    # During cooldown, even sustained drift must not retrigger.
    suppressed = [ctrl.observe(_drift()) for _ in range(3)]
    assert all(not d.triggered for d in suppressed)
    assert suppressed[0].cooldown_remaining == 2
    assert len(calls) == 1

    # After cooldown drains, a fresh sustained streak triggers again.
    ctrl.observe(_drift())
    second = ctrl.observe(_drift())
    assert second.triggered is True
    assert len(calls) == 2


@pytest.mark.parametrize("bad", [{"min_consecutive": 0}, {"cooldown": -1}, {"min_samples": -5}])
def test_invalid_config_rejected(bad):
    with pytest.raises(ValueError):
        RetrainController(retrain_fn=lambda s: None, **bad)


def test_signal_from_monitor_bridges_drift_monitor():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"f": rng.normal(0, 1, 500)})
    # Shift the current distribution hard so PSI clears the threshold.
    cur = pd.DataFrame({"f": rng.normal(5, 1, 500)})

    monitor = DriftMonitor(feature_names=["f"], psi_threshold=0.2, kl_threshold=0.1)
    monitor.set_reference(ref, rng.normal(0, 1, 500))
    monitor.update(cur, rng.normal(5, 1, 500))

    signal = signal_from_monitor(monitor, sample_count=len(cur))
    assert signal.drifted is True
    assert signal.sample_count == 500
    assert signal.psi is not None and signal.psi > 0.2
