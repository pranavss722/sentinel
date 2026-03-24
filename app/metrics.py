"""Prometheus metrics instrumentation."""
from prometheus_client import Counter, Histogram, Gauge

prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total prediction requests",
)

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
)

model_version_info = Gauge(
    "model_version_info",
    "Currently loaded model version",
)

canary_rollback_total = Counter(
    "canary_rollback_total",
    "Total canary rollbacks triggered",
)

route_decisions_total = Counter(
    "route_decisions_total",
    "Routing decisions by model role",
    ["model"],
)

data_drift_psi = Gauge(
    "data_drift_psi",
    "PSI drift score per feature",
    ["feature"],
)

prediction_drift_kl_divergence = Gauge(
    "prediction_drift_kl_divergence",
    "KL divergence on prediction score distribution",
)
