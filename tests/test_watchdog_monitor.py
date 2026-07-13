"""Tests for TargetMonitor (state machine + dedup + latency drift) and the engine.

Prove: an alert fires on a state TRANSITION only (no re-paging on repeated polls
in the same state), a single recovery notification is sent, a rising latency
distribution trips PSI before the hard SLO, and the one-shot engine cycle wires
prober -> SLO -> alert with a mocked sink and mocked probes (no network).
"""
from __future__ import annotations

from app.watchdog.config import BurnRateTier, Target, WatchdogConfig, WindowProfile
from app.watchdog.latency_drift import LatencyDriftMonitor
from app.watchdog.monitor import TargetMonitor, WatchdogEngine
from app.watchdog.prober import ProbeResult

PROFILE = WindowProfile(
    name="test",
    slo_window_seconds=500,
    latency_window_seconds=100,
    latency_min_samples=5,
    tiers=(
        BurnRateTier("fast", "PAGE", 100, 10, 14.4, 2.0, min_samples=5),
        BurnRateTier("slow", "TICKET", 200, 20, 1.0, 10.0, min_samples=5),
    ),
)


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float) -> float:
        self.t += dt
        return self.t


class RecordingSink:
    def __init__(self):
        self.alerts = []

    def send(self, alert) -> bool:
        self.alerts.append(alert)
        return True


def make_target(**kw) -> Target:
    base = dict(
        name="svc",
        url="https://example.test/health",
        latency_slo_ms=2000.0,  # high so latency-drift can trip before the hard SLO
        availability_slo=0.99,
    )
    base.update(kw)
    return Target(**base)


def _feed(monitor, clock, *, n, fail=0, latency_ms=50.0):
    """Record n probes at the current clock time; the first `fail` are failures."""
    for i in range(n):
        monitor.record(
            ProbeResult(
                target_name=monitor.target.name,
                success=i >= fail,
                status_code=200 if i >= fail else 500,
                latency_ms=latency_ms,
            ),
            timestamp=clock(),
        )


# --------------------------------------------------------------------------- #
# DEDUP: alert fires on transition only; recovery sent once
# --------------------------------------------------------------------------- #
def test_alert_fires_on_transition_only_and_recovers_once():
    clock = Clock(1000.0)
    sink = RecordingSink()
    monitor = TargetMonitor(make_target(), PROFILE, sink, clock=clock)

    # Healthy: no alert.
    _feed(monitor, clock, n=60, fail=0)
    assert monitor.evaluate().state == "OK"
    assert sink.alerts == []

    # Incident: 30% of the recent window fails -> PAGE. One alert.
    _feed(monitor, clock, n=60, fail=20)
    status = monitor.evaluate()
    assert status.state == "PAGE"
    assert status.transitioned is True
    assert len(sink.alerts) == 1
    assert sink.alerts[0].severity == "PAGE"

    # Still paging on the next polls -> NO new alert (dedup on state change).
    _feed(monitor, clock, n=10, fail=8)
    assert monitor.evaluate().transitioned is False
    assert monitor.evaluate().transitioned is False
    assert len(sink.alerts) == 1

    # Recover: advance past the burn windows and feed healthy probes -> ONE
    # recovery notification.
    clock.tick(300)
    _feed(monitor, clock, n=60, fail=0)
    status = monitor.evaluate()
    assert status.state == "OK"
    assert status.transitioned is True
    assert len(sink.alerts) == 2
    assert sink.alerts[1].severity == "RECOVERY"
    assert sink.alerts[1].is_recovery is True

    # Staying healthy does not re-notify.
    _feed(monitor, clock, n=10, fail=0)
    assert monitor.evaluate().transitioned is False
    assert len(sink.alerts) == 2


def test_single_transient_failure_does_not_page():
    clock = Clock(1000.0)
    sink = RecordingSink()
    monitor = TargetMonitor(make_target(), PROFILE, sink, clock=clock)

    _feed(monitor, clock, n=100, fail=0)
    # one lone failure
    monitor.record(
        ProbeResult(target_name="svc", success=False, status_code=500, latency_ms=50.0),
        timestamp=clock(),
    )
    status = monitor.evaluate()
    assert status.state == "OK"
    assert sink.alerts == []


# --------------------------------------------------------------------------- #
# LATENCY DRIFT: a rising latency distribution trips PSI before the hard SLO
# --------------------------------------------------------------------------- #
def test_latency_drift_trips_before_hard_slo():
    clock = Clock(1000.0)
    sink = RecordingSink()
    # min_samples small so a 30-probe window is enough to compute PSI.
    drift = LatencyDriftMonitor(psi_threshold=0.2, min_samples=30)
    monitor = TargetMonitor(make_target(latency_slo_ms=2000.0), PROFILE, sink, clock=clock,
                            latency_drift=drift)

    # Baseline: 50 healthy probes at ~90-139 ms pin the drift reference.
    for i in range(50):
        monitor.record(
            ProbeResult(target_name="svc", success=True, status_code=200,
                        latency_ms=90.0 + i),
            timestamp=clock(),
        )
    assert monitor.evaluate().state == "OK"

    # Later window: latency creeps to ~390-439 ms — still 200s< the 2000ms hard
    # SLO, still all 200 OK, but the DISTRIBUTION has shifted.
    clock.tick(200)
    for i in range(50):
        monitor.record(
            ProbeResult(target_name="svc", success=True, status_code=200,
                        latency_ms=390.0 + i),
            timestamp=clock(),
        )
    status = monitor.evaluate()

    assert status.latency_drift is True
    assert status.latency_psi > 0.2
    assert status.state == "TICKET"  # caught as a ticket-level regression
    # ... and this happened BEFORE the hard p99 SLO tripped:
    assert status.p99_latency_ms is not None
    assert status.p99_latency_ms < 2000.0
    assert len(sink.alerts) == 1
    assert "latency drift" in sink.alerts[0].reason


# --------------------------------------------------------------------------- #
# ENGINE: one-shot cycle wires prober -> evaluate -> alert (mocked probes/sink)
# --------------------------------------------------------------------------- #
class FakeProber:
    def __init__(self, results_by_target):
        self._results = results_by_target

    def probe(self, target) -> ProbeResult:
        return self._results[target.name]


def test_engine_probe_all_once_runs_every_target_without_network():
    targets = [
        make_target(name="up-svc"),
        make_target(name="down-svc"),
    ]
    config = WatchdogConfig(targets=targets, profile=PROFILE, alert_backend="ntfy")
    prober = FakeProber(
        {
            "up-svc": ProbeResult("up-svc", success=True, status_code=200, latency_ms=40.0),
            "down-svc": ProbeResult(
                "down-svc", success=False, status_code=None, latency_ms=10000.0,
                error="timeout"
            ),
        }
    )
    sink = RecordingSink()
    engine = WatchdogEngine(config, prober=prober, sink=sink, clock=Clock(1000.0))

    statuses = engine.probe_all_once()
    assert {s.target for s in statuses} == {"up-svc", "down-svc"}
    up = next(s for s in statuses if s.target == "up-svc")
    down = next(s for s in statuses if s.target == "down-svc")
    assert up.up is True
    assert down.up is False
