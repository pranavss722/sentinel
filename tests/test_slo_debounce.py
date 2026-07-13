"""Tests for the hardened, noise-resistant autonomous SLO rollback.

These lock in the two guardrails added to ``CanaryController``:

  * a **minimum-sample floor** — the SLO is not evaluated for rollback until the
    latency/error window holds at least ``slo_min_samples`` observations, and
  * a **sustained-breach debounce** — the SLO must breach on
    ``rollback_consecutive_breaches`` CONSECUTIVE polls before a rollback fires;
    any healthy poll resets the counter.

Drift remains an immediate trigger by design (it is already batch-debounced
inside ``DriftMonitor``); the final test documents that choice.
"""
from __future__ import annotations

import numpy as np

from app.canary import CanaryController
from app.metrics import canary_rollback_total


class FakeModel:
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])


class FakeRegistry:
    def __init__(self):
        self._model = FakeModel()

    def get_champion(self):
        return self._model

    def get_challenger(self):
        return self._model


class FakeDriftMonitor:
    def __init__(self, should_roll: bool = False):
        self._should_roll = should_roll

    def should_rollback(self) -> bool:
        return self._should_roll


class ResettableFakeDriftMonitor:
    """Drift double that records reset() and clears its verdict when reset."""

    def __init__(self, should_roll: bool = False):
        self._should_roll = should_roll
        self.reset_called = False

    def should_rollback(self) -> bool:
        return self._should_roll

    def reset(self) -> None:
        self.reset_called = True
        self._should_roll = False


def make_controller(
    canary_weight: float = 0.5,
    slo_min_samples: int = 200,
    consecutive: int = 3,
    drift: bool = False,
) -> CanaryController:
    return CanaryController(
        registry=FakeRegistry(),
        drift_monitor=FakeDriftMonitor(drift),
        canary_weight=canary_weight,
        slo_p99_ms=200.0,
        slo_error_rate_pct=1.0,
        slo_min_samples=slo_min_samples,
        rollback_consecutive_breaches=consecutive,
    )


def _load_latencies(controller: CanaryController, n: int, latency_s: float) -> None:
    for _ in range(n):
        controller.record_latency(latency_s)


# --------------------------------------------------------------------------- #
# NEGATIVE: a single/transient breach must not roll back
# --------------------------------------------------------------------------- #
def test_single_transient_slo_breach_does_not_rollback():
    controller = make_controller(consecutive=3)
    # A genuine, well-sampled p99 breach (1000ms >> 200ms), but only ONE poll.
    _load_latencies(controller, 300, 1.0)
    assert controller.check_slo() is False

    before = canary_rollback_total._value.get()
    rolled = controller.poll_once()

    assert rolled is False
    assert controller.canary_weight == 0.5
    assert controller._consecutive_slo_breaches == 1
    assert canary_rollback_total._value.get() == before


# --------------------------------------------------------------------------- #
# MIN-SAMPLE: a breach computed from too few observations must not roll back
# --------------------------------------------------------------------------- #
def test_slo_breach_below_min_samples_does_not_rollback():
    # consecutive=1 isolates the min-sample guard: even with no debounce, a thin
    # window must be treated as healthy.
    controller = make_controller(slo_min_samples=200, consecutive=1)
    _load_latencies(controller, 50, 5.0)  # 50 wildly-slow samples, but < 200

    # Not enough data → treated as healthy, never a breach.
    assert controller.check_slo() is True

    before = canary_rollback_total._value.get()
    for _ in range(5):
        assert controller.poll_once() is False

    assert controller.canary_weight == 0.5
    assert controller._consecutive_slo_breaches == 0
    assert canary_rollback_total._value.get() == before


# --------------------------------------------------------------------------- #
# POSITIVE: a breach sustained across the required consecutive polls rolls back
# --------------------------------------------------------------------------- #
def test_sustained_slo_breach_rolls_back():
    controller = make_controller(consecutive=3)
    _load_latencies(controller, 300, 1.0)

    before = canary_rollback_total._value.get()

    # First two breaching polls are withheld ...
    assert controller.poll_once() is False
    assert controller.poll_once() is False
    assert controller.canary_weight == 0.5
    assert canary_rollback_total._value.get() == before

    # ... the third consecutive breach fires the rollback.
    assert controller.poll_once() is True
    assert controller.canary_weight == 0.0
    assert canary_rollback_total._value.get() == before + 1

    # Fire-exactly-once: further polls at weight 0 do not re-increment.
    assert controller.poll_once() is False
    assert canary_rollback_total._value.get() == before + 1


# --------------------------------------------------------------------------- #
# RESET: a healthy poll between breaches resets the debounce counter
# --------------------------------------------------------------------------- #
def test_healthy_poll_resets_debounce_counter():
    controller = make_controller(consecutive=3)

    # Two consecutive breaches (counter → 2, no rollback yet).
    _load_latencies(controller, 300, 1.0)
    assert controller.poll_once() is False
    assert controller.poll_once() is False
    assert controller._consecutive_slo_breaches == 2

    # The window recovers to healthy latencies → next poll resets the counter.
    controller._latencies.clear()
    _load_latencies(controller, 300, 0.010)  # 10ms, well within 200ms SLO
    assert controller.check_slo() is True
    assert controller.poll_once() is False
    assert controller._consecutive_slo_breaches == 0
    assert controller.canary_weight == 0.5

    # A fresh breach now starts counting from zero: two breaches still no rollback.
    controller._latencies.clear()
    _load_latencies(controller, 300, 1.0)
    before = canary_rollback_total._value.get()
    assert controller.poll_once() is False
    assert controller.poll_once() is False
    assert controller.canary_weight == 0.5
    assert canary_rollback_total._value.get() == before

    # Only the third fresh consecutive breach rolls back.
    assert controller.poll_once() is True
    assert controller.canary_weight == 0.0
    assert canary_rollback_total._value.get() == before + 1


# --------------------------------------------------------------------------- #
# ERROR-RATE path is debounced the same way
# --------------------------------------------------------------------------- #
def test_sustained_error_rate_breach_rolls_back_but_single_does_not():
    controller = make_controller(consecutive=3)
    # 300 requests, ~10% erroring (>> 1% SLO), above the min-sample floor.
    for i in range(300):
        controller.record_request(error=(i % 10 == 0))
    assert controller.check_slo() is False

    before = canary_rollback_total._value.get()
    assert controller.poll_once() is False  # transient — withheld
    assert controller.canary_weight == 0.5

    rolled = False
    for _ in range(controller._rollback_consecutive_breaches):
        rolled = controller.poll_once() or rolled
    assert rolled is True
    assert controller.canary_weight == 0.0
    assert canary_rollback_total._value.get() == before + 1


# --------------------------------------------------------------------------- #
# DELIBERATE ASYMMETRY: drift stays an immediate trigger (not debounced)
# --------------------------------------------------------------------------- #
def test_drift_rollback_is_immediate_not_debounced():
    controller = make_controller(consecutive=3, drift=True)
    before = canary_rollback_total._value.get()

    # A single poll rolls back on drift — no consecutive-breach requirement.
    assert controller.poll_once() is True
    assert controller.canary_weight == 0.0
    assert canary_rollback_total._value.get() == before + 1


# --------------------------------------------------------------------------- #
# STATE HYGIENE: rollback leaves the controller clean for the next canary
# --------------------------------------------------------------------------- #
def test_rollback_clears_windows_and_debounce_counter():
    controller = make_controller(consecutive=3)
    _load_latencies(controller, 300, 1.0)
    for _ in range(3):  # 3 sustained breaching polls → rollback on the 3rd
        controller.poll_once()
    assert controller.canary_weight == 0.0  # rolled back

    # The rolling SLO windows and the debounce counter are cleared.
    assert len(controller._latencies) == 0
    assert len(controller._errors) == 0
    assert controller._consecutive_slo_breaches == 0


def test_freshly_promoted_canary_not_in_breach_from_stale_samples():
    controller = make_controller(canary_weight=0.5, consecutive=3)

    # Incident: a sustained breach drives a rollback, leaving 300 slow samples
    # in the window that (without hygiene) would carry into the next canary.
    _load_latencies(controller, 300, 1.0)
    for _ in range(3):
        controller.poll_once()
    assert controller.canary_weight == 0.0

    # Promote a fresh canary. Its window is empty, so it is healthy — the
    # previous incident's samples did NOT carry over and cannot force an
    # immediate rollback.
    before = canary_rollback_total._value.get()
    controller.advance_stage()  # weight → 0.01
    assert controller.canary_weight > 0
    assert controller.check_slo() is True
    assert controller.poll_once() is False
    assert controller.canary_weight > 0
    assert canary_rollback_total._value.get() == before


def test_rollback_resets_drift_monitor_state():
    dm = ResettableFakeDriftMonitor(should_roll=True)
    controller = CanaryController(
        registry=FakeRegistry(),
        drift_monitor=dm,
        canary_weight=0.5,
        slo_min_samples=200,
        rollback_consecutive_breaches=3,
    )
    # Drift breach → immediate rollback → controller resets the drift monitor.
    assert controller.poll_once() is True
    assert dm.reset_called is True
    assert controller._drift_monitor.should_rollback() is False
