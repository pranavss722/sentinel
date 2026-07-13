"""Tests for the synthetic prober — records latency/success; failures never crash."""
from __future__ import annotations

import pytest

from app.watchdog.config import Target
from app.watchdog.prober import Prober, ProbeResult


class FakeResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


class SequenceClock:
    """Returns successive values so start/end latency is deterministic."""

    def __init__(self, values):
        self._values = list(values)

    def __call__(self) -> float:
        return self._values.pop(0)


def make_target(**kw) -> Target:
    base = dict(
        name="svc",
        url="https://example.test/health",
        expected_status=200,
        latency_slo_ms=500,
        availability_slo=0.99,
    )
    base.update(kw)
    return Target(**base)


def test_successful_probe_records_latency_and_success():
    clock = SequenceClock([10.0, 10.25])  # 250 ms elapsed
    prober = Prober(request_fn=lambda *a, **k: FakeResponse(200, '{"status": "ok"}'), clock=clock)
    result = prober.probe(make_target(body_check="ok"))

    assert isinstance(result, ProbeResult)
    assert result.success is True
    assert result.status_code == 200
    assert result.latency_ms == pytest.approx(250.0)
    assert result.error is None


def test_unexpected_status_is_a_failed_probe():
    prober = Prober(request_fn=lambda *a, **k: FakeResponse(500, "boom"))
    result = prober.probe(make_target())
    assert result.success is False
    assert result.status_code == 500
    assert "unexpected status 500" in result.error


def test_body_check_mismatch_is_a_failed_probe():
    prober = Prober(request_fn=lambda *a, **k: FakeResponse(200, "<html>login</html>"))
    result = prober.probe(make_target(body_check="dashboard"))
    assert result.success is False
    assert result.body_ok is False
    assert "body check" in result.error


def test_timeout_counts_as_failed_probe_and_does_not_crash():
    class FakeTimeout(Exception):
        pass

    def boom(*a, **k):
        raise FakeTimeout("timed out")

    clock = SequenceClock([0.0, 2.0])
    prober = Prober(request_fn=boom, clock=clock)
    result = prober.probe(make_target())

    assert result.success is False
    assert result.status_code is None
    assert result.latency_ms == pytest.approx(2000.0)
    assert "FakeTimeout" in result.error


def test_connection_error_counts_as_failed_probe():
    def boom(*a, **k):
        raise OSError("connection refused")

    result = Prober(request_fn=boom).probe(make_target())
    assert result.success is False
    assert result.error is not None
