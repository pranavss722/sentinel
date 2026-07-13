"""Canary controller — Bernoulli traffic splitting and SLO-based rollback."""
from __future__ import annotations

import asyncio
import logging
import random
from collections import deque
from typing import Any

import numpy as np

from app.metrics import canary_rollback_total, route_decisions_total

logger = logging.getLogger(__name__)

CANARY_STAGES = [0.01, 0.10, 0.50, 1.0]

# --- SLO rollback hardening defaults ------------------------------------- #
# Minimum number of latency/error observations before the SLO is eligible to
# trigger a rollback. Below this, a thin window (a handful of slow requests)
# can produce a wild p99 and must NOT be trusted.
SLO_MIN_SAMPLES = 200
# Number of CONSECUTIVE monitoring polls that must observe an SLO breach before
# the controller rolls back. A single transient latency spike must not tear
# down a canary; any healthy poll resets the counter to 0.
ROLLBACK_CONSECUTIVE_BREACHES = 3


class CanaryController:
    """Routes prediction requests between champion and challenger models."""

    def __init__(
        self,
        registry: Any,
        drift_monitor: Any,
        canary_weight: float = 0.0,
        slo_p99_ms: float = 200.0,
        slo_error_rate_pct: float = 1.0,
        slo_min_samples: int = SLO_MIN_SAMPLES,
        rollback_consecutive_breaches: int = ROLLBACK_CONSECUTIVE_BREACHES,
    ) -> None:
        self._registry = registry
        self._drift_monitor = drift_monitor
        self.canary_weight = canary_weight
        self._slo_p99_ms = slo_p99_ms
        self._slo_error_rate_pct = slo_error_rate_pct
        self._slo_min_samples = slo_min_samples
        self._rollback_consecutive_breaches = rollback_consecutive_breaches
        # Consecutive-poll debounce counter for the SLO signal.
        self._consecutive_slo_breaches = 0
        self._latencies: deque[float] = deque(maxlen=10_000)
        self._errors: deque[bool] = deque(maxlen=10_000)
        self._stage_index = self._current_stage_index()
        self._poll_task: asyncio.Task | None = None

    def _current_stage_index(self) -> int:
        for i, stage in enumerate(CANARY_STAGES):
            if abs(self.canary_weight - stage) < 1e-9:
                return i
        return -1

    def route_request(self, payload: np.ndarray) -> dict:
        """Route a single request via Bernoulli draw on canary_weight."""
        challenger = self._registry.get_challenger()
        use_challenger = (
            challenger is not None
            and random.random() < self.canary_weight
        )

        if use_challenger:
            model = challenger
            model_label = "challenger"
        else:
            model = self._registry.get_champion()
            model_label = "champion"

        route_decisions_total.labels(model=model_label).inc()
        proba = model.predict_proba(payload)
        score = float(proba[0, 1])

        return {"model": model_label, "score": score}

    def record_latency(self, latency_s: float) -> None:
        self._latencies.append(latency_s)

    def record_request(self, error: bool = False) -> None:
        self._errors.append(error)

    def check_slo(self) -> bool:
        """Return True if the SLO holds over the recorded window.

        Each signal (p99 latency, error rate) is only evaluated once its window
        holds at least ``slo_min_samples`` observations. A thinner window is
        treated as "not enough data — healthy", so a handful of slow requests
        can never produce a spurious breach.
        """
        if len(self._latencies) >= self._slo_min_samples:
            p99_s = float(np.percentile(list(self._latencies), 99))
            if p99_s * 1000 > self._slo_p99_ms:
                return False

        if len(self._errors) >= self._slo_min_samples:
            error_rate = sum(self._errors) / len(self._errors) * 100
            if error_rate > self._slo_error_rate_pct:
                return False

        return True

    def rollback(self) -> None:
        """Reset canary weight to 0 — all traffic to champion.

        Also leaves the controller CLEAN for the next canary: the rolling SLO
        windows and the debounce counter are cleared, and the drift monitor's
        computed state is reset (its reference baseline is preserved). Without
        this, a subsequently-promoted canary would inherit the previous
        incident's latency/error samples and drift verdict in its rolling
        window — a freshly-promoted canary could then be judged "in breach"
        purely from stale pre-rollback state.
        """
        self.canary_weight = 0.0
        self._stage_index = -1
        self._latencies.clear()
        self._errors.clear()
        self._consecutive_slo_breaches = 0
        # DriftMonitor holds computed PSI/KL state that would otherwise carry a
        # prior incident forward; reset it if it supports it (test doubles may
        # not).
        reset = getattr(self._drift_monitor, "reset", None)
        if callable(reset):
            reset()
        canary_rollback_total.inc()

    def advance_stage(self) -> None:
        """Move to the next canary stage if one exists."""
        next_index = self._stage_index + 1
        if next_index < len(CANARY_STAGES):
            self._stage_index = next_index
            self.canary_weight = CANARY_STAGES[self._stage_index]

    def _rollback_reason(self) -> str | None:
        """Instantaneous rollback reason for the CURRENT window, ignoring the
        consecutive-poll debounce. Used for logging / reporting the live state.
        """
        if self._drift_monitor.should_rollback():
            return "drift threshold exceeded (PSI / KL divergence)"
        if not self.check_slo():
            return "SLO breach (p99 latency / error rate)"
        return None

    def poll_once(self) -> bool:
        """Run one monitoring evaluation and roll back if warranted.

        This is the per-cycle logic ``_drift_poll_loop`` invokes every poll
        interval (and which tests / the demo drive directly). Returns True iff
        it rolled back on this call.

        Debounce policy — deliberately asymmetric:
          * DRIFT (PSI / KL) is already debounced inside ``DriftMonitor``: its
            metrics are computed over a full window of observations, so a breach
            already reflects a sustained distributional shift, not point noise.
            We therefore act on it IMMEDIATELY — delaying a well-founded drift
            rollback by several poll cycles would only prolong bad serving.
          * SLO (rolling p99 / error rate) is spike-prone, so we require the
            breach to persist for ``rollback_consecutive_breaches`` CONSECUTIVE
            polls (each with >= ``slo_min_samples`` observations) before rolling
            back. Any healthy poll resets the counter to 0.
        """
        # Nothing to roll back (already at champion) — keep the debounce clean
        # and preserve fire-exactly-once behaviour.
        if self.canary_weight <= 0:
            self._consecutive_slo_breaches = 0
            return False

        if self._drift_monitor.should_rollback():
            self._consecutive_slo_breaches = 0
            self.rollback()
            logger.warning(
                "Auto-rollback triggered: drift threshold exceeded (PSI / KL divergence)"
            )
            return True

        if not self.check_slo():
            self._consecutive_slo_breaches += 1
            if self._consecutive_slo_breaches >= self._rollback_consecutive_breaches:
                self.rollback()
                logger.warning(
                    "Auto-rollback triggered: SLO breach sustained over %d consecutive polls",
                    self._consecutive_slo_breaches,
                )
                return True
            logger.info(
                "SLO breach observed (%d/%d consecutive polls) — rollback withheld",
                self._consecutive_slo_breaches,
                self._rollback_consecutive_breaches,
            )
        else:
            self._consecutive_slo_breaches = 0
        return False

    async def _drift_poll_loop(self, interval_seconds: int) -> None:
        while True:
            self.poll_once()
            await asyncio.sleep(interval_seconds)

    def stop_drift_polling(self) -> None:
        """Cancel the drift polling task cleanly."""
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
