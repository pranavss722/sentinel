"""Prometheus metrics for the watchdog.

Kept separate from :mod:`app.metrics` so the watchdog is strictly additive — the
existing serving/canary metric names are untouched. All metrics are labelled by
``target`` so they slot into the existing Grafana/Prometheus story.
"""
from prometheus_client import Counter, Gauge

watchdog_target_up = Gauge(
    "watchdog_target_up",
    "Whether the last probe of the target succeeded (1) or failed (0)",
    ["target"],
)

watchdog_probe_latency_ms = Gauge(
    "watchdog_probe_latency_ms",
    "Latency of the most recent probe in milliseconds",
    ["target"],
)

watchdog_error_budget_remaining = Gauge(
    "watchdog_error_budget_remaining",
    "Fraction (0..1) of the availability error budget remaining over the SLO window",
    ["target"],
)

watchdog_burn_rate = Gauge(
    "watchdog_burn_rate",
    "Error-budget burn rate for the target over a given window",
    ["target", "window"],
)

watchdog_latency_psi = Gauge(
    "watchdog_latency_psi",
    "PSI of the current latency distribution vs the healthy baseline (latency drift)",
    ["target"],
)

watchdog_alerts_total = Counter(
    "watchdog_alerts_total",
    "Total watchdog alerts emitted, by severity (PAGE/TICKET/RECOVERY)",
    ["target", "severity"],
)
