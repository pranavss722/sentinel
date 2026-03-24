"""Tests for CanaryController."""
import pytest
import numpy as np


class FakeModel:
    def __init__(self, name: str, pred_value: float = 0.5):
        self.name = name
        self._pred_value = pred_value

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([
            np.full(n, 1 - self._pred_value),
            np.full(n, self._pred_value),
        ])


class FakeRegistry:
    def __init__(self, champion, challenger=None):
        self._champion = champion
        self._challenger = challenger

    def get_champion(self):
        return self._champion

    def get_challenger(self):
        return self._challenger


class FakeDriftMonitor:
    def __init__(self, should_roll: bool = False):
        self._should_roll = should_roll

    def should_rollback(self) -> bool:
        return self._should_roll


def make_controller(canary_weight=0.0, champion_pred=0.5, challenger_pred=0.9, drift_rollback=False):
    from app.canary import CanaryController
    champion = FakeModel("champion", champion_pred)
    challenger = FakeModel("challenger", challenger_pred)
    registry = FakeRegistry(champion, challenger)
    drift_monitor = FakeDriftMonitor(drift_rollback)
    controller = CanaryController(
        registry=registry,
        drift_monitor=drift_monitor,
        canary_weight=canary_weight,
        slo_p99_ms=200.0,
        slo_error_rate_pct=1.0,
    )
    return controller


def test_canary_weight_zero_always_routes_to_champion():
    controller = make_controller(canary_weight=0.0)
    payload = np.array([[1.0] * 10])
    results = set()
    for _ in range(100):
        result = controller.route_request(payload)
        results.add(result["model"])
    assert results == {"champion"}


def test_canary_weight_one_always_routes_to_challenger():
    controller = make_controller(canary_weight=1.0)
    payload = np.array([[1.0] * 10])
    results = set()
    for _ in range(100):
        result = controller.route_request(payload)
        results.add(result["model"])
    assert results == {"challenger"}


def test_canary_weight_half_routes_to_both():
    controller = make_controller(canary_weight=0.5)
    payload = np.array([[1.0] * 10])
    models_seen = set()
    for _ in range(200):
        result = controller.route_request(payload)
        models_seen.add(result["model"])
    assert models_seen == {"champion", "challenger"}


def test_canary_returns_prediction_score():
    controller = make_controller(canary_weight=0.0, champion_pred=0.7)
    payload = np.array([[1.0] * 10])
    result = controller.route_request(payload)
    assert result["score"] == pytest.approx(0.7)


def test_canary_no_challenger_always_champion():
    from app.canary import CanaryController
    champion = FakeModel("champion", 0.5)
    registry = FakeRegistry(champion, challenger=None)
    drift_monitor = FakeDriftMonitor(False)
    controller = CanaryController(
        registry=registry,
        drift_monitor=drift_monitor,
        canary_weight=0.5,  # would normally route to challenger
        slo_p99_ms=200.0,
        slo_error_rate_pct=1.0,
    )
    payload = np.array([[1.0] * 10])
    result = controller.route_request(payload)
    assert result["model"] == "champion"


def test_check_slo_passes_when_healthy():
    controller = make_controller()
    # Fresh controller with no recorded latencies — SLO should hold
    assert controller.check_slo() is True


def test_check_slo_fails_on_high_latency():
    controller = make_controller()
    # Simulate breached p99 latency
    for _ in range(100):
        controller.record_latency(0.300)  # 300ms > 200ms SLO
    assert controller.check_slo() is False


def test_check_slo_fails_on_high_error_rate():
    controller = make_controller()
    # Simulate 5% error rate
    for i in range(100):
        controller.record_request(error=(i < 5))
    assert controller.check_slo() is False


def test_rollback_resets_canary_weight():
    controller = make_controller(canary_weight=0.5)
    controller.rollback()
    assert controller.canary_weight == 0.0


def test_canary_progression_stages():
    from app.canary import CANARY_STAGES
    assert CANARY_STAGES == [0.01, 0.10, 0.50, 1.0]


def test_advance_stage_increments_weight():
    controller = make_controller(canary_weight=0.01)
    controller._stage_index = 0
    controller.advance_stage()
    assert controller.canary_weight == pytest.approx(0.10)
