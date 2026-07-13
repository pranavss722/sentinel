"""Latency-drift detection — reuses the PSI engine from :mod:`app.drift`.

This is the differentiator versus a plain uptime monitor: a service can stay
"up" (200 OK) while its latency distribution slowly creeps upward. Feeding the
rolling latency distribution through the SAME Population Stability Index engine
that guards the model's feature drift lets the watchdog flag a latency
REGRESSION *before* the hard p99 SLO trips.

We reuse :func:`app.drift._compute_psi` directly rather than driving a full
``DriftMonitor`` so we do not clobber the shared serving drift gauges
(``data_drift_psi`` / ``prediction_drift_kl_divergence``) — the watchdog exposes
its own ``watchdog_latency_psi`` gauge instead. The PSI math is identical.
"""
from __future__ import annotations

import numpy as np

from app.drift import _compute_psi


class LatencyDriftMonitor:
    """Tracks PSI of a target's latency distribution vs a healthy baseline."""

    def __init__(self, psi_threshold: float = 0.2, min_samples: int = 30) -> None:
        self._psi_threshold = psi_threshold
        self._min_samples = min_samples
        self._reference: np.ndarray | None = None
        self._psi: float = 0.0
        self._has_update = False

    def set_reference(self, latencies_ms: list[float] | np.ndarray) -> None:
        """Pin the healthy baseline latency distribution."""
        arr = np.asarray(latencies_ms, dtype=float)
        if arr.size >= self._min_samples:
            self._reference = arr

    def has_reference(self) -> bool:
        return self._reference is not None

    def update(self, latencies_ms: list[float] | np.ndarray) -> float | None:
        """Compute PSI of the current window vs the baseline.

        Returns the PSI value, or ``None`` if there is not yet enough data
        (no reference set, or fewer than ``min_samples`` current observations) —
        in which case the drift verdict stays quiet.
        """
        cur = np.asarray(latencies_ms, dtype=float)
        if self._reference is None or cur.size < self._min_samples:
            return None
        self._psi = _compute_psi(self._reference, cur)
        self._has_update = True
        return self._psi

    def psi(self) -> float:
        return self._psi

    def is_regressed(self) -> bool:
        """True once a computed PSI exceeds the drift threshold."""
        return self._has_update and self._psi > self._psi_threshold

    def reset(self) -> None:
        self._psi = 0.0
        self._has_update = False
