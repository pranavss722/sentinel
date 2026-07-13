"""Drift-triggered retrain controller.

:class:`~app.drift.DriftMonitor` tells us *whether* the live distribution has
drifted from the champion's training baseline at one point in time. Turning that
raw signal into an actual retrain is a policy decision, and a naive
``if drifted: retrain()`` trigger is dangerous:

  * a single noisy window would kick off an expensive retrain;
  * a freshly retrained model would immediately retrain again while the drift
    window still holds pre-retrain data;
  * a verdict computed on a handful of samples is statistically meaningless.

This controller applies the same discipline the SLO-rollback path already uses,
adapted to retraining:

  * ``min_samples``       — ignore verdicts computed on too little current data;
  * ``min_consecutive``   — require drift on N consecutive evaluations before
                            acting (debounce transient spikes);
  * ``cooldown``          — after a retrain fires, suppress further triggers for a
                            number of evaluations so the new model can settle and
                            the drift window can refill with post-retrain data.

It is pure and deterministic: no wall-clock, no I/O, and the actual retraining is
an injected callable, so this module never imports xgboost / mlflow and stays
trivially unit-testable. The one side effect is incrementing process-global
Prometheus counters, which is safe and observable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.metrics import model_retrains_total, retrain_consecutive_drift


@dataclass(frozen=True)
class DriftSignal:
    """A single drift evaluation handed to the controller.

    ``drifted`` is the monitor's verdict; ``sample_count`` is how many current-data
    points it was computed from; ``psi``/``kl`` are carried through for the reason
    string and observability only.
    """

    drifted: bool
    sample_count: int
    psi: float | None = None
    kl: float | None = None


@dataclass(frozen=True)
class RetrainDecision:
    """Outcome of feeding one signal to the controller."""

    triggered: bool
    reason: str
    consecutive_drift: int
    cooldown_remaining: int


@dataclass
class RetrainController:
    """Debounced, cooldown-gated retrain trigger.

    ``retrain_fn`` is invoked with the triggering :class:`DriftSignal` exactly once
    per decision to retrain; its return value is ignored (fire-and-forget). Any
    exception it raises propagates to the caller — a retrain that cannot start is a
    real failure, not something to swallow.
    """

    retrain_fn: Callable[[DriftSignal], object]
    min_consecutive: int = 3
    cooldown: int = 5
    min_samples: int = 200

    _streak: int = field(default=0, init=False)
    _cooldown_remaining: int = field(default=0, init=False)
    _retrain_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.min_consecutive < 1:
            raise ValueError("min_consecutive must be >= 1")
        if self.cooldown < 0:
            raise ValueError("cooldown must be >= 0")
        if self.min_samples < 0:
            raise ValueError("min_samples must be >= 0")

    @property
    def retrain_count(self) -> int:
        return self._retrain_count

    def observe(self, signal: DriftSignal) -> RetrainDecision:
        """Feed one drift evaluation; return whether a retrain was triggered."""
        # Cooldown wins over everything: hold off while the last retrain settles.
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._streak = 0
            retrain_consecutive_drift.set(0)
            return self._decision(False, "cooldown")

        # Too little data to trust the verdict — do not let it advance the streak.
        if signal.sample_count < self.min_samples:
            self._streak = 0
            retrain_consecutive_drift.set(0)
            return self._decision(
                False, f"insufficient samples ({signal.sample_count} < {self.min_samples})"
            )

        if not signal.drifted:
            self._streak = 0
            retrain_consecutive_drift.set(0)
            return self._decision(False, "no drift")

        self._streak += 1
        retrain_consecutive_drift.set(self._streak)

        if self._streak < self.min_consecutive:
            return self._decision(
                False, f"drift {self._streak}/{self.min_consecutive} (debouncing)"
            )

        # Sustained drift: fire.
        reason = self._trigger_reason(signal)
        self.retrain_fn(signal)
        self._retrain_count += 1
        model_retrains_total.labels(reason=self._reason_label(signal)).inc()
        self._streak = 0
        self._cooldown_remaining = self.cooldown
        retrain_consecutive_drift.set(0)
        return self._decision(True, reason)

    def _trigger_reason(self, signal: DriftSignal) -> str:
        bits = [f"sustained drift over {self.min_consecutive} evaluations"]
        if signal.psi is not None:
            bits.append(f"PSI {signal.psi:.3f}")
        if signal.kl is not None:
            bits.append(f"KL {signal.kl:.3f}")
        return "; ".join(bits)

    @staticmethod
    def _reason_label(signal: DriftSignal) -> str:
        # Coarse label for the Prometheus counter (bounded cardinality).
        if signal.psi is not None and signal.kl is not None:
            return "psi+kl"
        if signal.psi is not None:
            return "psi"
        if signal.kl is not None:
            return "kl"
        return "drift"

    def _decision(self, triggered: bool, reason: str) -> RetrainDecision:
        return RetrainDecision(
            triggered=triggered,
            reason=reason,
            consecutive_drift=self._streak,
            cooldown_remaining=self._cooldown_remaining,
        )


def signal_from_monitor(monitor, sample_count: int) -> DriftSignal:
    """Build a :class:`DriftSignal` from a :class:`~app.drift.DriftMonitor`.

    Reuses the monitor's own verdict (``should_rollback``) and carries the worst
    per-feature PSI plus the prediction-drift KL through for the reason string, so
    the retrain policy sits directly on top of the existing drift detector rather
    than re-deriving thresholds.
    """
    psi_values = monitor.get_psi_values()
    worst_psi = max(psi_values.values(), default=0.0)
    return DriftSignal(
        drifted=monitor.should_rollback(),
        sample_count=sample_count,
        psi=worst_psi,
        kl=monitor.get_kl_divergence(),
    )
