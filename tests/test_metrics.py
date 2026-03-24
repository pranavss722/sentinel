"""Tests for Prometheus metrics definitions."""
from prometheus_client import Counter, Histogram, Gauge


def test_prediction_requests_total_exists():
    from app.metrics import prediction_requests_total
    assert isinstance(prediction_requests_total, Counter)


def test_prediction_latency_seconds_exists():
    from app.metrics import prediction_latency_seconds
    assert isinstance(prediction_latency_seconds, Histogram)


def test_model_version_info_exists():
    from app.metrics import model_version_info
    assert isinstance(model_version_info, Gauge)


def test_canary_rollback_total_exists():
    from app.metrics import canary_rollback_total
    assert isinstance(canary_rollback_total, Counter)


def test_route_decisions_total_exists():
    from app.metrics import route_decisions_total
    assert isinstance(route_decisions_total, Counter)


def test_route_decisions_total_has_model_label():
    from app.metrics import route_decisions_total
    assert "model" in route_decisions_total._labelnames


def test_data_drift_psi_exists():
    from app.metrics import data_drift_psi
    assert isinstance(data_drift_psi, Gauge)


def test_data_drift_psi_has_feature_label():
    from app.metrics import data_drift_psi
    assert "feature" in data_drift_psi._labelnames


def test_prediction_drift_kl_divergence_exists():
    from app.metrics import prediction_drift_kl_divergence
    assert isinstance(prediction_drift_kl_divergence, Gauge)
