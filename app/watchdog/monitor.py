"""Per-target monitor + the watchdog engine.

``TargetMonitor`` owns one target's rolling probe history (``ErrorBudgetTracker``)
and latency-drift baseline (``LatencyDriftMonitor``), turns each probe into an
availability/latency/burn-rate verdict, drives a small OK→TICKET→PAGE state
machine, and emits a push alert ONLY on a state transition (dedup) — with a
single recovery notification on the return to OK.

``WatchdogEngine`` fans a :class:`~app.watchdog.prober.Prober` across all targets
for a one-shot ``--once`` cycle and runs the per-target daemon loop.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable

from app.watchdog.alerts import Alert, AlertSink
from app.watchdog.config import Target, WatchdogConfig, WindowProfile
from app.watchdog.latency_drift import LatencyDriftMonitor
from app.watchdog.metrics import (
    watchdog_alerts_total,
    watchdog_burn_rate,
    watchdog_error_budget_remaining,
    watchdog_latency_psi,
    watchdog_probe_latency_ms,
    watchdog_target_up,
)
from app.watchdog.prober import ProbeResult, Prober
from app.watchdog.slo import ErrorBudgetTracker, SLOEvaluation

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"OK": 0, "TICKET": 1, "PAGE": 2}

# Number of initial healthy probes used to pin the latency-drift baseline.
_REFERENCE_SAMPLE_TARGET = 50


@dataclass
class TargetStatus:
    """Flat summary of a single evaluation — handy for CLI/reporting/tests."""

    target: str
    up: bool
    state: str  # "OK" | "TICKET" | "PAGE"
    availability: float
    budget_remaining: float
    burn_rate: float
    p99_latency_ms: float | None
    latency_psi: float
    latency_drift: bool
    reason: str
    transitioned: bool


class TargetMonitor:
    def __init__(
        self,
        target: Target,
        profile: WindowProfile,
        sink: AlertSink,
        clock: Callable[[], float] = time.monotonic,
        latency_drift: LatencyDriftMonitor | None = None,
    ) -> None:
        self.target = target
        self._profile = profile
        self._sink = sink
        self._clock = clock
        self._tracker = ErrorBudgetTracker(
            availability_slo=target.availability_slo,
            latency_slo_ms=target.latency_slo_ms,
            profile=profile,
            clock=clock,
        )
        self._drift = latency_drift or LatencyDriftMonitor()
        self._state = "OK"
        self._ref_buf: list[float] = []
        self._last_up = True

    @property
    def state(self) -> str:
        return self._state

    def record(self, result: ProbeResult, timestamp: float | None = None) -> None:
        self._tracker.record(result.success, result.latency_ms, timestamp)
        self._last_up = result.success
        watchdog_target_up.labels(target=self.target.name).set(1 if result.success else 0)
        watchdog_probe_latency_ms.labels(target=self.target.name).set(result.latency_ms)

        # Pin the healthy latency baseline from the first N successful probes.
        if result.success and not self._drift.has_reference():
            self._ref_buf.append(result.latency_ms)
            if len(self._ref_buf) >= _REFERENCE_SAMPLE_TARGET:
                self._drift.set_reference(self._ref_buf)

    def evaluate(self, now: float | None = None) -> TargetStatus:
        ev = self._tracker.evaluate(now)

        # Latency drift (early warning): PSI of the current window vs baseline.
        cur_lat = self._tracker.recent_latencies(self._profile.latency_window_seconds, now)
        self._drift.update(cur_lat)
        drift_regressed = self._drift.is_regressed()

        desired, reason = self._decide(ev, drift_regressed)

        name = self.target.name
        watchdog_error_budget_remaining.labels(target=name).set(ev.error_budget_remaining)
        watchdog_latency_psi.labels(target=name).set(self._drift.psi())
        for tier in ev.tiers:
            window = f"{tier.name}-long"
            watchdog_burn_rate.labels(target=name, window=window).set(tier.long_burn_rate)

        transitioned = self._transition(desired, ev, reason)

        return TargetStatus(
            target=name,
            up=self._last_up,
            state=self._state,
            availability=ev.availability,
            budget_remaining=ev.error_budget_remaining,
            burn_rate=ev.burn_rate,
            p99_latency_ms=ev.p99_latency_ms,
            latency_psi=self._drift.psi(),
            latency_drift=drift_regressed,
            reason=reason,
            transitioned=transitioned,
        )

    def _decide(self, ev: SLOEvaluation, drift_regressed: bool) -> tuple[str, str]:
        """Map an evaluation to a desired state and a human-readable reason."""
        if ev.page:
            reasons = []
            for tier in ev.tiers:
                if tier.breaching and tier.severity == "PAGE":
                    reasons.append(
                        f"fast-burn {tier.name} (burn {tier.long_burn_rate:.1f}x)"
                    )
            if ev.latency_slo_breached and ev.p99_latency_ms is not None:
                reasons.append(
                    f"p99 latency {ev.p99_latency_ms:.0f}ms > {self.target.latency_slo_ms:.0f}ms SLO"
                )
            return "PAGE", "; ".join(reasons) or "PAGE condition"

        if ev.ticket or drift_regressed:
            reasons = []
            for tier in ev.tiers:
                if tier.breaching and tier.severity == "TICKET":
                    reasons.append(
                        f"slow-burn {tier.name} (burn {tier.long_burn_rate:.2f}x)"
                    )
            if drift_regressed:
                reasons.append(f"latency drift (PSI {self._drift.psi():.2f})")
            return "TICKET", "; ".join(reasons) or "TICKET condition"

        return "OK", "healthy"

    def _transition(self, desired: str, ev: SLOEvaluation, reason: str) -> bool:
        """Fire an alert only when the state actually changes. Returns True on a
        transition."""
        if desired == self._state:
            return False

        previous = self._state
        self._state = desired
        name = self.target.name

        if desired == "OK":
            alert = Alert(
                target=name,
                severity="RECOVERY",
                reason=f"recovered from {previous}",
                availability=ev.availability,
                budget_remaining=ev.error_budget_remaining,
                burn_rate=ev.burn_rate,
                p99_latency_ms=ev.p99_latency_ms,
                is_recovery=True,
            )
            watchdog_alerts_total.labels(target=name, severity="RECOVERY").inc()
            logger.warning("watchdog RECOVERY: %s recovered from %s", name, previous)
        else:
            alert = Alert(
                target=name,
                severity=desired,
                reason=reason,
                availability=ev.availability,
                budget_remaining=ev.error_budget_remaining,
                burn_rate=ev.burn_rate,
                p99_latency_ms=ev.p99_latency_ms,
            )
            watchdog_alerts_total.labels(target=name, severity=desired).inc()
            logger.warning("watchdog %s: %s - %s", desired, name, reason)

        self._sink.send(alert)
        return True


class WatchdogEngine:
    """Fans a prober across all configured targets."""

    def __init__(
        self,
        config: WatchdogConfig,
        prober: Prober | None = None,
        sink: AlertSink | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._prober = prober or Prober()
        self._sink = sink or AlertSink(backend=config.alert_backend)
        self._clock = clock
        self.monitors: dict[str, TargetMonitor] = {
            t.name: TargetMonitor(t, config.profile, self._sink, clock=clock)
            for t in config.targets
        }

    def probe_target(self, target: Target) -> TargetStatus:
        result = self._prober.probe(target)
        monitor = self.monitors[target.name]
        monitor.record(result)
        status = monitor.evaluate()
        logger.info(
            "probe %s up=%s state=%s avail=%.3f%% budget=%.1f%% burn=%.2fx",
            target.name,
            status.up,
            status.state,
            status.availability * 100,
            status.budget_remaining * 100,
            status.burn_rate,
        )
        return status

    def probe_all_once(self) -> list[TargetStatus]:
        """One probe→evaluate→alert cycle across every target (``--once``)."""
        return [self.probe_target(t) for t in self.config.targets]

    async def _target_loop(self, target: Target) -> None:
        while True:
            try:
                self.probe_target(target)
            except Exception:  # noqa: BLE001 - never let one target kill the loop
                logger.exception("watchdog loop error for %s", target.name)
            await asyncio.sleep(target.interval_seconds)

    async def run_forever(self) -> None:
        """Run every target on its own interval until cancelled."""
        tasks = [asyncio.create_task(self._target_loop(t)) for t in self.config.targets]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            for task in tasks:
                task.cancel()
            raise
