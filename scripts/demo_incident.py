"""Reproducible autonomous-rollback incident demo — BOTH rollback paths.

Stages TWO realistic model-serving incidents WITHOUT the docker stack, each with
its OWN fresh controller/registry (no cross-contamination), while exercising the
*real* production code paths:

  * app.model_registry.ModelRegistry  (champion / challenger resolution)
  * app.drift.DriftMonitor            (PSI + KL divergence)
  * app.canary.CanaryController       (Bernoulli routing, the hardened
                                        ``poll_once`` autonomous rollback with a
                                        min-sample floor + sustained-breach
                                        debounce, and window/state hygiene on
                                        rollback)
  * app.metrics.*                     (Prometheus counters / gauges)

The two scenarios demonstrate the controller's deliberately *asymmetric* policy:

  Scenario A — DRIFT incident: a real +6σ covariate shift makes PSI/KL breach.
    Drift is already batch-debounced inside ``DriftMonitor``, so the controller
    rolls back IMMEDIATELY — on the first poll, zero polls withheld.

  Scenario B — SLO incident: a degraded challenger saturates the serving path so
    p99 latency / error rate breach, while the input distribution stays stable
    (drift quiet). The rolling SLO signal is spike-prone, so the controller
    WITHHOLDS rollback until the breach is confirmed on N consecutive polls.

All breach evidence (PSI/KL, p99, error rate, sample count, consecutive-breach
count) is captured at the moment of detection/confirmation — BEFORE rollback
clears the controller's windows — so the report shows the values the controller
actually acted on.

Run standalone::

    python scripts/demo_incident.py

Artifacts written to ``reports/``: ``incident_report.md`` (both timelines +
comparison + postmortem + metric evidence) and, if matplotlib is available,
``incident_timeline.png`` (a two-panel figure, one per scenario).

Fully deterministic (seeded RNG); both scenarios fire identically on repeat
runs. ``tests/test_incident_demo.py`` runs it headlessly and asserts both
rollback paths and the debounce.
"""
from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from app.canary import CANARY_STAGES, ROLLBACK_CONSECUTIVE_BREACHES, CanaryController
from app.drift import DriftMonitor
from app.metrics import (
    canary_rollback_total,
    data_drift_psi,
    prediction_drift_kl_divergence,
    route_decisions_total,
)
from app.model_registry import ModelRegistry

SEED = 1337
MODEL_NAME = "ad-click-baseline"
FEATURE_NAMES = [f"f{i}" for i in range(10)]
DRIFTED_FEATURE = "f3"
DRIFT_SHIFT_SIGMA = 6.0  # covariate shift applied in the drift scenario

# SLO thresholds used for the demo controllers.
SLO_P99_MS = 200.0
SLO_ERROR_RATE_PCT = 1.0

# Batch sizes.
HEALTHY_CYCLES = 2
HEALTHY_BATCH = 400
INCIDENT_BATCH = 400
RECOVERY_BATCH = 500

# Latency profile (seconds). Healthy traffic is comfortably inside the SLO.
# In the SLO scenario the degraded challenger saturates the shared inference
# path, inflating latency for champion traffic too, and is itself
# catastrophically slow and erroring.
HEALTHY_LATENCY_S = 0.020
INCIDENT_CHAMPION_LATENCY_S = 0.240  # 240ms > 200ms p99 SLO
DEGRADED_LATENCY_S = 1.500
INCIDENT_ERROR_EVERY = 20  # ~5% of incident champion traffic errors under load

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


# --------------------------------------------------------------------------- #
# Timeline plumbing
# --------------------------------------------------------------------------- #
@dataclass
class TimelineEvent:
    t: float  # wall-clock seconds since scenario start
    ts: str  # ISO-8601 timestamp
    kind: str
    message: str


@dataclass
class TickSample:
    t: float
    label: str
    canary_weight: float
    p99_ms: float
    max_psi: float
    consecutive_breaches: int


@dataclass
class ScenarioResult:
    key: str  # "drift" | "slo"
    title: str
    trigger_label: str
    events: list[TimelineEvent] = field(default_factory=list)
    samples: list[TickSample] = field(default_factory=list)

    # Hardening configuration observed on the controller.
    consecutive_breaches_required: int = 0
    min_samples_required: int = 0

    # Detection / debounce evidence
    polls_until_rollback: int = 0  # drift = 1 (immediate), slo = N (confirmed)
    polls_withheld: int = 0        # drift = 0, slo = N - 1
    rolled_back: bool = False

    # Rollback invariants
    canary_weight_final: float = 1.0
    rollback_count_delta: float = 0.0

    # Traffic evidence (post-rollback recovery window)
    champion_decisions_after: float = 0.0
    challenger_decisions_after: float = 0.0

    # Evidence captured AT detection, BEFORE any state clearing on rollback.
    p99_latency_ms_at_detection: float = 0.0
    error_rate_pct_at_detection: float = 0.0
    slo_sample_count_at_detection: int = 0
    slo_breached_at_detection: bool = False
    drift_psi_max_at_detection: float = 0.0
    drift_kl_at_detection: float = 0.0
    drift_breached_at_detection: bool = False
    gauge_psi_drifted: float = 0.0
    gauge_kl: float = 0.0
    rollback_reason: str = ""

    # State-hygiene evidence AFTER rollback (windows/debounce should be clean).
    latencies_after_rollback: int = -1
    errors_after_rollback: int = -1
    consecutive_after_rollback: int = -1
    drift_should_rollback_after: bool = True

    @property
    def champion_share_after(self) -> float:
        total = self.champion_decisions_after + self.challenger_decisions_after
        return 100.0 if total == 0 else 100.0 * self.champion_decisions_after / total

    @property
    def hygiene_clean(self) -> bool:
        return (
            self.latencies_after_rollback == 0
            and self.errors_after_rollback == 0
            and self.consecutive_after_rollback == 0
            and self.drift_should_rollback_after is False
        )

    @property
    def time_to_rollback_desc(self) -> str:
        if self.polls_withheld == 0:
            return f"immediate ({self.polls_until_rollback} poll, 0 withheld)"
        return (
            f"{self.polls_until_rollback} polls "
            f"(~{self.polls_until_rollback} min at 60s/poll, {self.polls_withheld} withheld)"
        )


@dataclass
class DemoResult:
    drift: ScenarioResult
    slo: ScenarioResult
    plot_path: Path | None = None

    @property
    def scenarios(self) -> list[ScenarioResult]:
        return [self.drift, self.slo]


# --------------------------------------------------------------------------- #
# Real-model / real-registry construction
# --------------------------------------------------------------------------- #
class _StubVersion:
    def __init__(self, source: str) -> None:
        self.source = source
        self.version = "1"


class _InMemoryMlflowClient:
    """Minimal MLflow-client stand-in so the *real* ModelRegistry code runs.

    No tracking server, no network — the loader returns pre-trained in-process
    models keyed by the alias source string.
    """

    def __init__(self) -> None:
        self.challenger_present = True

    def get_model_version_by_alias(self, name: str, alias: str) -> _StubVersion:
        if alias == "champion":
            return _StubVersion("champion")
        if alias == "challenger" and self.challenger_present:
            return _StubVersion("challenger")
        raise ValueError(f"no model registered for alias {alias!r}")


def _train_models(rng: np.random.RandomState) -> tuple[Any, Any, pd.DataFrame, np.ndarray]:
    """Train a healthy champion and a deliberately degraded challenger.

    Returns (champion, challenger, reference_features, reference_scores).
    """
    n = 4_000
    X = rng.randn(n, len(FEATURE_NAMES))
    # A clean, learnable signal for the champion.
    logits = 1.4 * X[:, 0] - 1.1 * X[:, 1] + 0.9 * X[:, 3]
    y = (logits + 0.3 * rng.randn(n) > 0).astype(int)

    champion = xgb.XGBClassifier(
        n_estimators=60, max_depth=4, learning_rate=0.2,
        random_state=SEED, eval_metric="logloss",
    )
    champion.fit(X, y)

    # Degraded challenger: trained on label-corrupted, feature-scrambled data so
    # its score distribution is skewed and its predictions are poor.
    flip = rng.rand(n) < 0.45
    y_bad = np.where(flip, 1 - y, y)
    X_bad = X.copy()
    X_bad[:, 3] = rng.randn(n)  # destroy the most informative feature
    challenger = xgb.XGBClassifier(
        n_estimators=25, max_depth=2, learning_rate=0.05,
        random_state=SEED, eval_metric="logloss",
    )
    challenger.fit(X_bad, y_bad)

    reference_features = pd.DataFrame(X, columns=FEATURE_NAMES)
    reference_scores = champion.predict_proba(X)[:, 1]
    return champion, challenger, reference_features, reference_scores


def build_stack(
    rng: np.random.RandomState,
) -> tuple[CanaryController, DriftMonitor, _InMemoryMlflowClient, pd.DataFrame, np.ndarray]:
    champion, challenger, ref_features, ref_scores = _train_models(rng)

    client = _InMemoryMlflowClient()
    models = {"champion": champion, "challenger": challenger}
    registry = ModelRegistry(
        client=client,
        model_name=MODEL_NAME,
        loader_fn=lambda source: models[source],
    )

    drift_monitor = DriftMonitor(
        feature_names=FEATURE_NAMES, psi_threshold=0.2, kl_threshold=0.1
    )
    drift_monitor.set_reference(ref_features, ref_scores)

    controller = CanaryController(
        registry=registry,
        drift_monitor=drift_monitor,
        canary_weight=0.0,
        slo_p99_ms=SLO_P99_MS,
        slo_error_rate_pct=SLO_ERROR_RATE_PCT,
    )
    return controller, drift_monitor, client, ref_features, ref_scores


# --------------------------------------------------------------------------- #
# Traffic generation
# --------------------------------------------------------------------------- #
def _sample_batch(
    rng: np.random.RandomState, size: int, drift_shift: float = 0.0
) -> pd.DataFrame:
    """Sample a batch of incoming feature rows from the reference distribution.

    ``drift_shift`` applies a covariate shift (in sigmas) to one feature — used
    by the drift scenario; left at 0 for the SLO scenario so drift stays quiet.
    """
    X = rng.randn(size, len(FEATURE_NAMES))
    data = pd.DataFrame(X, columns=FEATURE_NAMES)
    if drift_shift:
        data[DRIFTED_FEATURE] = data[DRIFTED_FEATURE] + drift_shift
    return data


def drive_traffic(
    controller: CanaryController,
    drift_monitor: DriftMonitor,
    rng: np.random.RandomState,
    batch: pd.DataFrame,
    *,
    degraded: bool,
) -> None:
    """Route a batch through the *real* controller and record real signals.

    When ``degraded`` is True, the serving path is saturated: challenger-served
    requests get catastrophic latency + errors, and champion requests are slowed
    past the p99 budget with sporadic errors.
    """
    scores: list[float] = []
    for i, (_, row) in enumerate(batch.iterrows()):
        payload = row.to_numpy(dtype=float).reshape(1, -1)
        result = controller.route_request(payload)  # real Bernoulli + predict_proba
        scores.append(result["score"])

        is_challenger = result["model"] == "challenger"
        if degraded and is_challenger:
            controller.record_latency(DEGRADED_LATENCY_S)
            controller.record_request(error=True)
        elif degraded:
            controller.record_latency(INCIDENT_CHAMPION_LATENCY_S)
            controller.record_request(error=(i % INCIDENT_ERROR_EVERY == 0))
        else:
            controller.record_latency(HEALTHY_LATENCY_S)
            controller.record_request(error=False)

    # Feed the real DriftMonitor with the current window (features + scores).
    drift_monitor.update(batch, np.asarray(scores))


def _counter_value(counter: Any, **labels: str) -> float:
    metric = counter.labels(**labels) if labels else counter
    return float(metric._value.get())


def _p99_ms(controller: CanaryController) -> float:
    if not controller._latencies:
        return 0.0
    return float(np.percentile(list(controller._latencies), 99)) * 1000


def _error_rate_pct(controller: CanaryController) -> float:
    if not controller._errors:
        return 0.0
    return 100.0 * sum(controller._errors) / len(controller._errors)


def _now(start: float) -> tuple[float, str]:
    return time.perf_counter() - start, datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def run_autonomous_monitor_poll(controller: CanaryController) -> bool:
    """Run exactly ONE production monitoring poll and return whether it rolled back.

    ``poll_once`` is the per-cycle logic the production ``_drift_poll_loop``
    invokes every poll interval — driving it directly gives us one deterministic
    poll per call, which is exactly what the consecutive-breach debounce reasons
    about.
    """
    return controller.poll_once()


def _snapshot(controller: CanaryController, drift_monitor: DriftMonitor) -> dict:
    """Evidence snapshot taken BEFORE a poll (so it survives rollback clearing)."""
    psi = drift_monitor.get_psi_values()
    return {
        "p99": _p99_ms(controller),
        "err": _error_rate_pct(controller),
        "samples": len(controller._latencies),
        "psi_max": max(psi.values()) if psi else 0.0,
        "kl": drift_monitor.get_kl_divergence(),
        "gauge_psi": float(data_drift_psi.labels(feature=DRIFTED_FEATURE)._value.get()),
        "gauge_kl": float(prediction_drift_kl_divergence._value.get()),
        "consecutive": controller._consecutive_slo_breaches,
        "slo_breached": not controller.check_slo(),
        "drift_breached": drift_monitor.should_rollback(),
    }


# --------------------------------------------------------------------------- #
# Generic scenario runner (drives ONE incident on a fresh stack)
# --------------------------------------------------------------------------- #
def _run_scenario(
    seed: int,
    *,
    key: str,
    title: str,
    trigger_label: str,
    incident_event_msg: str,
    incident_drift_shift: float,
    incident_degraded: bool,
) -> ScenarioResult:
    random.seed(seed)
    rng = np.random.RandomState(seed)

    controller, drift_monitor, _client, _ref_features, _ref_scores = build_stack(rng)
    sr = ScenarioResult(key=key, title=title, trigger_label=trigger_label)
    sr.consecutive_breaches_required = controller._rollback_consecutive_breaches
    sr.min_samples_required = controller._slo_min_samples
    n_required = sr.consecutive_breaches_required
    start = time.perf_counter()

    def log(kind: str, message: str) -> None:
        t, ts = _now(start)
        sr.events.append(TimelineEvent(t=t, ts=ts, kind=kind, message=message))

    def sample(label: str, *, p99: float | None = None, psi: float | None = None) -> None:
        t = time.perf_counter() - start
        live_psi = drift_monitor.get_psi_values()
        sr.samples.append(
            TickSample(
                t=t,
                label=label,
                canary_weight=controller.canary_weight,
                p99_ms=_p99_ms(controller) if p99 is None else p99,
                max_psi=(max(live_psi.values()) if live_psi else 0.0) if psi is None else psi,
                consecutive_breaches=controller._consecutive_slo_breaches,
            )
        )

    log("start", "Sentinel serving healthy — champion at 100% traffic")

    # 1) Promote challenger to canary stage 1 (1%).
    controller.advance_stage()
    assert abs(controller.canary_weight - CANARY_STAGES[0]) < 1e-9
    log("promotion", f"Challenger promoted to canary at {controller.canary_weight:.0%}")

    # 2) Healthy canary phase — clean traffic, monitor polls, stays quiet.
    for i in range(HEALTHY_CYCLES):
        batch = _sample_batch(rng, size=HEALTHY_BATCH)
        drive_traffic(controller, drift_monitor, rng, batch, degraded=False)
        rolled = run_autonomous_monitor_poll(controller)
        assert not rolled and controller.canary_weight == CANARY_STAGES[0]
        sample(f"healthy-{i}")
    log(
        "steady",
        f"Canary healthy over {HEALTHY_CYCLES * HEALTHY_BATCH} requests — "
        f"p99={sr.samples[-1].p99_ms:.0f}ms within {SLO_P99_MS:.0f}ms SLO, "
        "drift within threshold",
    )

    rollback_before = _counter_value(canary_rollback_total)

    # 3) Incident onset.
    log("incident", incident_event_msg)

    # 4) Poll loop — one incident batch + one real monitor poll per cycle.
    for cycle in range(1, n_required + 5):  # headroom; must fire by n_required
        batch = _sample_batch(rng, size=INCIDENT_BATCH, drift_shift=incident_drift_shift)
        drive_traffic(controller, drift_monitor, rng, batch, degraded=incident_degraded)

        snap = _snapshot(controller, drift_monitor)  # BEFORE the poll clears state
        rolled = run_autonomous_monitor_poll(controller)
        sample(f"incident-poll-{cycle}", p99=snap["p99"], psi=snap["psi_max"])

        if rolled:
            sr.rolled_back = True
            sr.polls_until_rollback = cycle
            sr.polls_withheld = cycle - 1
            # Evidence the controller acted on (pre-rollback snapshot).
            sr.p99_latency_ms_at_detection = snap["p99"]
            sr.error_rate_pct_at_detection = snap["err"]
            sr.slo_sample_count_at_detection = snap["samples"]
            sr.slo_breached_at_detection = snap["slo_breached"]
            sr.drift_psi_max_at_detection = snap["psi_max"]
            sr.drift_kl_at_detection = snap["kl"]
            sr.drift_breached_at_detection = snap["drift_breached"]
            sr.gauge_psi_drifted = snap["gauge_psi"]
            sr.gauge_kl = snap["gauge_kl"]
            reason = "drift" if snap["drift_breached"] else "SLO"
            sr.rollback_reason = (
                "drift threshold exceeded (PSI / KL divergence)"
                if snap["drift_breached"]
                else "SLO breach (p99 latency / error rate)"
            )
            if reason == "drift":
                log(
                    "rollback",
                    f"AUTONOMOUS ROLLBACK fired IMMEDIATELY — drift breach "
                    f"(PSI[{DRIFTED_FEATURE}]={snap['gauge_psi']:.2f} > 0.20, "
                    f"KL={snap['kl']:.3f} > 0.10). 0 polls withheld. "
                    f"canary_weight {CANARY_STAGES[0]:.2f} → {controller.canary_weight:.2f}",
                )
            else:
                log(
                    "rollback",
                    f"AUTONOMOUS ROLLBACK fired — SLO breach sustained over {cycle} "
                    f"consecutive polls (p99={snap['p99']:.0f}ms, "
                    f"err={snap['err']:.2f}%). canary_weight "
                    f"{CANARY_STAGES[0]:.2f} → {controller.canary_weight:.2f}",
                )
            break

        log(
            "breach_withheld",
            f"SLO breach observed on poll {cycle}/{n_required} "
            f"(p99={snap['p99']:.0f}ms > {SLO_P99_MS:.0f}ms) — "
            "rollback WITHHELD pending confirmation",
        )

    sr.canary_weight_final = controller.canary_weight
    sr.rollback_count_delta = _counter_value(canary_rollback_total) - rollback_before

    # 4b) State-hygiene evidence — captured AFTER rollback cleared the windows.
    sr.latencies_after_rollback = len(controller._latencies)
    sr.errors_after_rollback = len(controller._errors)
    sr.consecutive_after_rollback = controller._consecutive_slo_breaches
    sr.drift_should_rollback_after = drift_monitor.should_rollback()

    # 5) Verify traffic is fully restored to champion.
    champ_pre = _counter_value(route_decisions_total, model="champion")
    chall_pre = _counter_value(route_decisions_total, model="challenger")
    recovery_batch = _sample_batch(rng, size=RECOVERY_BATCH)
    drive_traffic(controller, drift_monitor, rng, recovery_batch, degraded=False)
    sr.champion_decisions_after = (
        _counter_value(route_decisions_total, model="champion") - champ_pre
    )
    sr.challenger_decisions_after = (
        _counter_value(route_decisions_total, model="challenger") - chall_pre
    )
    sample("recovered")
    log(
        "recovered",
        f"Traffic restored: {sr.champion_decisions_after:.0f}/"
        f"{sr.champion_decisions_after + sr.challenger_decisions_after:.0f} "
        f"post-rollback requests served by champion ({sr.champion_share_after:.1f}%). "
        f"Windows cleared (lat={sr.latencies_after_rollback}, err={sr.errors_after_rollback}), "
        f"debounce reset ({sr.consecutive_after_rollback}), "
        f"drift verdict cleared ({not sr.drift_should_rollback_after}).",
    )
    return sr


# --------------------------------------------------------------------------- #
# Top-level: run BOTH scenarios on independent fresh stacks
# --------------------------------------------------------------------------- #
def run_incident(seed: int = SEED) -> DemoResult:
    """Run the DRIFT and SLO scenarios, each on its own fresh controller."""
    drift = _run_scenario(
        seed,
        key="drift",
        title="Scenario A — Data-drift incident (immediate rollback)",
        trigger_label="Data drift (PSI / KL divergence)",
        incident_event_msg=(
            f"Incident begins: +{DRIFT_SHIFT_SIGMA:.0f} sigma covariate shift on "
            f"{DRIFTED_FEATURE} — PSI/KL breach. Drift is batch-debounced, so the "
            "controller acts on the FIRST poll (no consecutive-breach wait)."
        ),
        incident_drift_shift=DRIFT_SHIFT_SIGMA,
        incident_degraded=False,
    )
    slo = _run_scenario(
        seed,
        key="slo",
        title="Scenario B — SLO incident (debounced rollback)",
        trigger_label="SLO breach (p99 latency / error rate)",
        incident_event_msg=(
            "Incident begins: degraded challenger saturates the serving path — "
            "p99 latency + error rate breach SLO, input distribution stable "
            f"(drift quiet). Controller requires {ROLLBACK_CONSECUTIVE_BREACHES} "
            "consecutive breaching polls before acting."
        ),
        incident_drift_shift=0.0,
        incident_degraded=True,
    )
    return DemoResult(drift=drift, slo=slo)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _render_plot(demo: DemoResult, path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.4))

    def _panel(ax, sr: ScenarioResult, signal: str) -> None:
        idx = list(range(len(sr.samples)))
        labels = [s.label for s in sr.samples]
        weights = [s.canary_weight * 100 for s in sr.samples]
        color_w = "#2563eb"
        ax.plot(idx, weights, "-o", color=color_w, label="Canary traffic %")
        ax.set_ylabel("Canary traffic (%)", color=color_w)
        ax.tick_params(axis="y", labelcolor=color_w)
        ax.set_ylim(-0.5, max(weights + [1]) * 1.6 + 0.5)
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)

        ax2 = ax.twinx()
        color_p = "#dc2626"
        if signal == "p99":
            series = [s.p99_ms for s in sr.samples]
            ax2.plot(idx, series, "-s", color=color_p, label="p99 latency (ms)")
            ax2.axhline(SLO_P99_MS, color=color_p, linestyle="--", alpha=0.6)
            ax2.set_ylabel("p99 latency (ms)", color=color_p)
        else:
            series = [s.max_psi for s in sr.samples]
            ax2.plot(idx, series, "-s", color=color_p, label="max PSI (drift)")
            ax2.axhline(0.20, color=color_p, linestyle="--", alpha=0.6)
            ax2.set_ylabel("max PSI", color=color_p)
        ax2.tick_params(axis="y", labelcolor=color_p)

        # Shade withheld polls (SLO scenario only).
        for i, s in enumerate(sr.samples):
            if s.label.startswith("incident-poll") and s.canary_weight > 0:
                ax.axvspan(i - 0.4, i + 0.4, color="#f59e0b", alpha=0.12)
        # Mark the rollback poll.
        for i, s in enumerate(sr.samples):
            if s.label.startswith("incident-poll") and s.canary_weight == 0:
                ax.axvline(i, color="#16a34a", linestyle=":", linewidth=2)
                tag = "immediate\nrollback" if sr.polls_withheld == 0 else (
                    f"rollback\n(confirmed {sr.polls_until_rollback}x)"
                )
                ax.annotate(
                    tag, xy=(i, 0), xytext=(i, max(weights + [1]) * 1.0 + 0.4),
                    color="#16a34a", ha="center", fontsize=8, fontweight="bold",
                )
                break
        ax.set_title(
            f"{sr.title}  —  {sr.polls_withheld} polls withheld", fontsize=10
        )

    _panel(axes[0], demo.drift, "psi")
    _panel(axes[1], demo.slo, "p99")
    axes[1].set_xlabel("Monitoring poll")
    fig.suptitle(
        "Sentinel — asymmetric autonomous rollback: drift = immediate, SLO = debounced",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _verdict(breached: bool, quiet_label: str = "ok (quiet)") -> str:
    return "🔴 BREACH" if breached else quiet_label


def _scenario_section(sr: ScenarioResult) -> str:
    timeline_rows = "\n".join(
        f"| +{ev.t:6.3f}s | `{ev.kind}` | {ev.message} |" for ev in sr.events
    )
    return f"""### {sr.title}

**Trigger the controller acted on:** {sr.rollback_reason}
**Detection:** {sr.time_to_rollback_desc}

| Signal | Value at detection | Threshold | Verdict |
|--------|-------------------|-----------|---------|
| Data drift — PSI[`{DRIFTED_FEATURE}`] | {sr.gauge_psi_drifted:.3f} | 0.20 | {_verdict(sr.gauge_psi_drifted > 0.20)} |
| Prediction drift — KL divergence | {sr.drift_kl_at_detection:.3f} | 0.10 | {_verdict(sr.drift_kl_at_detection > 0.10)} |
| Latency SLO — p99 | {sr.p99_latency_ms_at_detection:.0f} ms | {SLO_P99_MS:.0f} ms | {_verdict(sr.p99_latency_ms_at_detection > SLO_P99_MS)} |
| Error-rate SLO | {sr.error_rate_pct_at_detection:.2f} % | {SLO_ERROR_RATE_PCT:.2f} % | {_verdict(sr.error_rate_pct_at_detection > SLO_ERROR_RATE_PCT)} |

| Invariant | Observed |
|-----------|----------|
| `canary_weight` after rollback | **{sr.canary_weight_final:.2f}** (expected 0.00) |
| `canary_rollback_total` delta | **+{sr.rollback_count_delta:.0f}** |
| Polls until rollback | **{sr.polls_until_rollback}** |
| Polls withheld | **{sr.polls_withheld}** |
| SLO window at detection | **{sr.slo_sample_count_at_detection:,}** obs (floor {sr.min_samples_required}) |
| Post-rollback traffic to champion | **{sr.champion_decisions_after:.0f} / {sr.champion_decisions_after + sr.challenger_decisions_after:.0f}** ({sr.champion_share_after:.1f}%) |
| **State hygiene** — windows cleared | lat={sr.latencies_after_rollback}, err={sr.errors_after_rollback}, debounce={sr.consecutive_after_rollback}, drift-verdict-cleared={not sr.drift_should_rollback_after} |

**Timeline**

| Elapsed | Event | Detail |
|---------|-------|--------|
{timeline_rows}
"""


def write_report(demo: DemoResult, reports_dir: Path = REPORTS_DIR) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    plot_path = _render_plot(demo, reports_dir / "incident_timeline.png")
    demo.plot_path = plot_path

    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    d, s = demo.drift, demo.slo

    plot_section = ""
    if plot_path is not None:
        plot_section = (
            "\n## Traffic split & breach signal over both incidents\n\n"
            f"![Both scenarios — drift immediate vs SLO debounced]({plot_path.name})\n"
        )

    report = f"""# Incident Report — Autonomous Canary Rollback (both paths)

*Generated: {generated} • Scenario: `scripts/demo_incident.py` (deterministic, seed={SEED})*

## Summary

Two independent incidents, each on its own fresh `CanaryController`, demonstrate
Sentinel's **deliberately asymmetric** autonomous-rollback policy:

- **Drift (PSI/KL) is batch-debounced inside `DriftMonitor`**, so a breach
  already reflects a sustained distributional shift → the controller rolls back
  **immediately**, on the first poll.
- **The rolling SLO signal (p99 / error rate) is spike-prone**, so the
  controller **withholds rollback until the breach is confirmed on
  {s.consecutive_breaches_required} consecutive polls** — a single transient spike never tears
  down a canary.

Both incidents ended with `canary_weight` back to 0, `canary_rollback_total`
incremented once, 100% of traffic restored to the champion, and the controller
left **clean** for the next canary (windows + debounce + drift verdict reset).

## Asymmetric policy — side by side

| Scenario | Trigger signal | Debounced? | Polls withheld | Time-to-rollback |
|----------|----------------|-----------|----------------|------------------|
| **A — Drift** | PSI[`{DRIFTED_FEATURE}`]={d.gauge_psi_drifted:.2f}, KL={d.drift_kl_at_detection:.3f} | No (already batch-debounced) | **{d.polls_withheld}** | {d.time_to_rollback_desc} |
| **B — SLO** | p99={s.p99_latency_ms_at_detection:.0f}ms, err={s.error_rate_pct_at_detection:.2f}% | Yes ({s.consecutive_breaches_required} consecutive polls) | **{s.polls_withheld}** | {s.time_to_rollback_desc} |

*Both scenarios are deterministic (seed={SEED}) and fire identically on repeat runs.*
{plot_section}
## Scenario detail

{_scenario_section(d)}
{_scenario_section(s)}
## State hygiene (window/debounce reset on rollback)

`rollback()` clears the controller's rolling `_latencies`/`_errors` windows,
resets the SLO debounce counter, and resets the `DriftMonitor`'s computed
verdict (its reference baseline is preserved). Without this, a subsequently
promoted canary would inherit the previous incident's samples and could be
judged "in breach" purely from stale state. Observed after rollback:

| Scenario | latencies | errors | debounce counter | drift `should_rollback()` |
|----------|-----------|--------|------------------|---------------------------|
| A — Drift | {d.latencies_after_rollback} | {d.errors_after_rollback} | {d.consecutive_after_rollback} | {d.drift_should_rollback_after} |
| B — SLO | {s.latencies_after_rollback} | {s.errors_after_rollback} | {s.consecutive_after_rollback} | {s.drift_should_rollback_after} |

## Postmortem

- **Two failure modes, one controller.** A model can go bad by *drifting* (its
  inputs/scores move away from the training distribution) or by *degrading
  service* (latency/errors). Sentinel guards both, autonomously.
- **Why the asymmetry is correct.** PSI/KL are computed over a whole window of
  observations, so a drift breach is already a sustained signal — waiting extra
  poll cycles would only prolong bad serving. A rolling p99, by contrast, can
  spike from a single slow batch; rolling back on one poll would make the system
  trigger-happy. Confirming the SLO breach over {s.consecutive_breaches_required} polls (each above the
  {s.min_samples_required}-sample floor) trades a few minutes of confirmation for never tearing
  down a healthy canary on noise.
- **State hygiene.** Because `rollback()` clears the windows and the drift
  verdict, the *next* canary starts from a clean slate — the two scenarios in
  this report each began healthy despite the global process having served a
  prior incident.
- **What a human would have done manually.** Notice the alert, confirm the
  regression against a dashboard, decide to roll back, run the promotion tooling
  to reset the alias, and verify traffic drained off the canary — realistically
  **5–15+ minutes** of on-call time per incident. Sentinel does it autonomously:
  instantly for drift, and after a short confirmation window for SLO.

---
*Regression-protected by `tests/test_incident_demo.py` (asserts both paths) and
`tests/test_slo_debounce.py` (debounce + state hygiene).*
"""
    report_path = reports_dir / "incident_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_scenario(sr: ScenarioResult) -> None:
    print("\n" + "=" * 78)
    print(f"  {sr.title}")
    print("=" * 78)
    for ev in sr.events:
        print(f"  +{ev.t:6.3f}s  [{ev.kind:>16}]  {ev.message}")
    print("-" * 78)
    print(f"  canary_weight (final)   : {sr.canary_weight_final:.2f}  (expected 0.00)")
    print(f"  rollback_total (delta)  : +{sr.rollback_count_delta:.0f}")
    print(
        f"  polls until rollback    : {sr.polls_until_rollback}  "
        f"(withheld {sr.polls_withheld})"
    )
    print(
        f"  state hygiene           : lat={sr.latencies_after_rollback}, "
        f"err={sr.errors_after_rollback}, debounce={sr.consecutive_after_rollback}, "
        f"drift_cleared={not sr.drift_should_rollback_after}  "
        f"[{'clean' if sr.hygiene_clean else 'DIRTY'}]"
    )
    print(
        f"  champion post-rollback  : {sr.champion_share_after:.1f}%  "
        f"({sr.champion_decisions_after:.0f} champ / {sr.challenger_decisions_after:.0f} chall)"
    )


def _scenario_ok(sr: ScenarioResult, *, expect_immediate: bool) -> bool:
    base = (
        sr.rolled_back
        and sr.canary_weight_final == 0.0
        and sr.rollback_count_delta == 1
        and sr.challenger_decisions_after == 0
        and sr.hygiene_clean
    )
    if expect_immediate:
        return base and sr.polls_withheld == 0 and sr.polls_until_rollback == 1 \
            and sr.drift_breached_at_detection
    return base and sr.polls_withheld == sr.consecutive_breaches_required - 1 \
        and sr.polls_until_rollback == sr.consecutive_breaches_required \
        and sr.slo_breached_at_detection \
        and sr.slo_sample_count_at_detection >= sr.min_samples_required


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    demo = run_incident()
    for sr in demo.scenarios:
        _print_scenario(sr)

    report_path = write_report(demo)
    print(f"\n  Incident report : {report_path}")
    if demo.plot_path is not None:
        print(f"  Timeline plot   : {demo.plot_path}")
    else:
        print("  Timeline plot   : skipped (matplotlib unavailable)")

    ok = _scenario_ok(demo.drift, expect_immediate=True) and _scenario_ok(
        demo.slo, expect_immediate=False
    )
    if not ok:
        print("\n  ERROR: rollback/debounce/hygiene invariants NOT met.")
        return 1
    print("\n  Both scenarios fired correctly (drift immediate, SLO debounced); "
          "hygiene clean. [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
