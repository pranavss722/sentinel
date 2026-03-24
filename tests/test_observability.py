"""Tests for observability config: Prometheus alerts, Grafana dashboard."""
import json

import yaml


def test_alerts_yml_is_valid_yaml():
    with open("prometheus/alerts.yml") as f:
        data = yaml.safe_load(f)
    assert "groups" in data


def test_alerts_yml_has_all_5_rules():
    with open("prometheus/alerts.yml") as f:
        data = yaml.safe_load(f)
    rule_names = set()
    for group in data["groups"]:
        for rule in group["rules"]:
            rule_names.add(rule["alert"])
    assert rule_names == {
        "HighP99Latency",
        "HighErrorRate",
        "DataDriftDetected",
        "PredictionDriftDetected",
        "CanaryRollbackFired",
    }


def test_grafana_dashboard_is_valid_json():
    with open("grafana/dashboards/sentinel.json") as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_grafana_dashboard_has_6_panels():
    with open("grafana/dashboards/sentinel.json") as f:
        data = json.load(f)
    assert len(data["panels"]) == 6


def test_grafana_dashboard_panel_titles():
    with open("grafana/dashboards/sentinel.json") as f:
        data = json.load(f)
    titles = {p["title"] for p in data["panels"]}
    assert titles == {
        "Request Rate",
        "p99 Latency (ms)",
        "Error Rate %",
        "Canary Weight",
        "Data Drift PSI (max across features)",
        "Prediction Drift KL Divergence",
    }


def test_prometheus_datasource_is_valid_yaml():
    with open("grafana/provisioning/datasources/prometheus.yml") as f:
        data = yaml.safe_load(f)
    assert data["datasources"][0]["url"] == "http://prometheus:9090"
