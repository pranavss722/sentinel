"""Error-budget + multi-window multi-burn-rate SLO engine.

Implements the Google SRE Workbook "Alerting on SLOs" multi-window
multi-burn-rate approach (https://sre.google/workbook/alerting-on-slos/):

  * The **error budget** for an availability SLO ``a`` is ``1 - a`` (the
    fraction of probes you are allowed to lose over the SLO window).
  * The **burn rate** over a window is ``observed_error_rate / (1 - a)`` — a
    burn rate of 1 exactly exhausts the budget over the full SLO window; 14.4
    exhausts 2% of it in one hour of a 30-day window.
  * A tier fires only when BOTH its long and short windows breach the threshold
    (see :class:`~app.watchdog.config.BurnRateTier`). The short window (1/12 of
    the long) keeps the alert live only while the burn is still happening, which
    auto-resolves stale alerts.

Also evaluates the hard p99 latency SLO with a ``CanaryController``-style
min-sample floor so a thin window of slow probes cannot spuriously breach.

A ``clock`` is injectable so tests are fully deterministic.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.watchdog.config import WindowProfile


@dataclass
class ProbeRecord:
    timestamp: float
    success: bool
    latency_ms: float


@dataclass
class TierState:
    name: str
    severity: str
    long_burn_rate: float
    short_burn_rate: float
    long_samples: int
    short_samples: int
    breaching: bool


@dataclass
class SLOEvaluation:
    availability: float
    error_budget_remaining: float  # clamped to [0, 1] for gauges
    error_budget_remaining_raw: float  # may be negative when over budget
    burn_rate: float  # headline: burn over the fastest tier's long window
    p99_latency_ms: float | None
    latency_slo_breached: bool
    page: bool
    ticket: bool
    worst_severity: str  # "OK" | "TICKET" | "PAGE"
    tiers: list[TierState] = field(default_factory=list)


class ErrorBudgetTracker:
    """Rolling probe history → availability, error budget and burn rates."""

    def __init__(
        self,
        availability_slo: float,
        latency_slo_ms: float,
        profile: WindowProfile,
        clock: Callable[[], float] = time.monotonic,
        maxlen: int = 100_000,
    ) -> None:
        self._availability_slo = availability_slo
        self._budget = 1.0 - availability_slo
        self._latency_slo_ms = latency_slo_ms
        self._profile = profile
        self._clock = clock
        self._records: deque[ProbeRecord] = deque(maxlen=maxlen)

    # -- ingestion --------------------------------------------------------- #
    def record(self, success: bool, latency_ms: float, timestamp: float | None = None) -> None:
        ts = self._clock() if timestamp is None else timestamp
        self._records.append(ProbeRecord(ts, success, latency_ms))

    # -- window helpers ---------------------------------------------------- #
    def _in_window(self, window_seconds: float, now: float) -> list[ProbeRecord]:
        cutoff = now - window_seconds
        # records are appended in time order; scan from the newest backwards.
        out: list[ProbeRecord] = []
        for rec in reversed(self._records):
            if rec.timestamp < cutoff:
                break
            out.append(rec)
        return out

    def _burn_rate(self, window_seconds: float, now: float) -> tuple[float, int]:
        recs = self._in_window(window_seconds, now)
        n = len(recs)
        if n == 0 or self._budget <= 0:
            return 0.0, n
        failures = sum(1 for r in recs if not r.success)
        error_rate = failures / n
        return error_rate / self._budget, n

    def recent_latencies(self, window_seconds: float, now: float | None = None) -> list[float]:
        now = self._clock() if now is None else now
        return [r.latency_ms for r in self._in_window(window_seconds, now)]

    # -- evaluation -------------------------------------------------------- #
    def evaluate(self, now: float | None = None) -> SLOEvaluation:
        now = self._clock() if now is None else now

        tiers: list[TierState] = []
        page = False
        ticket = False
        for tier in self._profile.tiers:
            long_br, long_n = self._burn_rate(tier.long_window_seconds, now)
            short_br, short_n = self._burn_rate(tier.short_window_seconds, now)
            breaching = (
                long_n >= tier.min_samples
                and short_n >= tier.min_samples
                and long_br >= tier.burn_rate_threshold
                and short_br >= tier.burn_rate_threshold
            )
            if breaching:
                if tier.severity == "PAGE":
                    page = True
                elif tier.severity == "TICKET":
                    ticket = True
            tiers.append(
                TierState(
                    name=tier.name,
                    severity=tier.severity,
                    long_burn_rate=long_br,
                    short_burn_rate=short_br,
                    long_samples=long_n,
                    short_samples=short_n,
                    breaching=breaching,
                )
            )

        # Headline burn rate = burn over the fastest PAGE tier's long window
        # (falls back to the first tier). Most actionable single number.
        fast = self._profile.fast_tiers()
        headline_window = (
            fast[0].long_window_seconds
            if fast
            else self._profile.tiers[0].long_window_seconds
        )
        headline_burn, _ = self._burn_rate(headline_window, now)

        # Availability + error budget over the full SLO window.
        slo_recs = self._in_window(self._profile.slo_window_seconds, now)
        if slo_recs:
            failures = sum(1 for r in slo_recs if not r.success)
            availability = 1.0 - failures / len(slo_recs)
            consumed = (failures / len(slo_recs)) / self._budget if self._budget > 0 else 0.0
        else:
            availability = 1.0
            consumed = 0.0
        remaining_raw = 1.0 - consumed
        remaining = max(0.0, min(1.0, remaining_raw))

        # Hard p99 latency SLO, min-sample gated (CanaryController philosophy).
        lat = self.recent_latencies(self._profile.latency_window_seconds, now)
        p99: float | None = None
        latency_breached = False
        if len(lat) >= self._profile.latency_min_samples:
            p99 = float(np.percentile(lat, 99))
            latency_breached = p99 > self._latency_slo_ms

        page = page or latency_breached

        if page:
            worst = "PAGE"
        elif ticket:
            worst = "TICKET"
        else:
            worst = "OK"

        return SLOEvaluation(
            availability=availability,
            error_budget_remaining=remaining,
            error_budget_remaining_raw=remaining_raw,
            burn_rate=headline_burn,
            p99_latency_ms=p99,
            latency_slo_breached=latency_breached,
            page=page,
            ticket=ticket,
            worst_severity=worst,
            tiers=tiers,
        )
