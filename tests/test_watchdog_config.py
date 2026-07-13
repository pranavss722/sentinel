"""Tests for watchdog config loading and the burn-rate window profiles."""
from __future__ import annotations

import pytest

from app.watchdog.config import (
    DEMO_PROFILE,
    PRODUCTION_PROFILE,
    config_from_dict,
    get_profile,
    load_config,
)

VALID = {
    "profile": "production",
    "alerting": {"backend": "ntfy"},
    "targets": [
        {
            "name": "svc",
            "url": "https://example.test/health",
            "body_check": "ok",
            "latency_slo_ms": 800,
            "availability_slo": 0.995,
            "interval_seconds": 30,
        }
    ],
}


def test_config_from_dict_parses_targets_and_profile():
    cfg = config_from_dict(VALID)
    assert cfg.profile.name == "production"
    assert cfg.alert_backend == "ntfy"
    assert len(cfg.targets) == 1
    assert cfg.targets[0].url.endswith("/health")


def test_profile_override_wins():
    cfg = config_from_dict(VALID, profile_override="demo")
    assert cfg.profile.name == "demo"


def test_duplicate_target_names_rejected():
    raw = {"targets": [VALID["targets"][0], VALID["targets"][0]]}
    with pytest.raises(ValueError, match="duplicate target names"):
        config_from_dict(raw)


def test_unknown_target_key_rejected():
    raw = {"targets": [{"name": "x", "url": "https://x.test", "frequency": 5}]}
    with pytest.raises(ValueError, match="unknown keys"):
        config_from_dict(raw)


def test_no_targets_rejected():
    with pytest.raises(ValueError, match="no targets"):
        config_from_dict({"targets": []})


def test_invalid_availability_slo_rejected():
    raw = {"targets": [{"name": "x", "url": "https://x.test", "availability_slo": 1.5}]}
    with pytest.raises(ValueError, match="availability_slo"):
        config_from_dict(raw)


def test_unknown_profile_rejected():
    with pytest.raises(ValueError, match="unknown window profile"):
        get_profile("nope")


def test_production_profile_matches_canonical_sre_table():
    # fast-burn 2% in 1h -> 14.4 (1h/5m PAGE); 5% in 6h -> 6 (6h/30m PAGE);
    # slow-burn 10% in 3d -> 1 (3d/6h TICKET). Short window == 1/12 of long.
    by_name = {t.name: t for t in PRODUCTION_PROFILE.tiers}
    assert by_name["fast-1h"].burn_rate_threshold == 14.4
    assert by_name["fast-1h"].long_window_seconds == 3600
    assert by_name["fast-1h"].short_window_seconds == 300  # 3600 / 12
    assert by_name["fast-1h"].severity == "PAGE"

    assert by_name["fast-6h"].burn_rate_threshold == 6.0
    assert by_name["fast-6h"].long_window_seconds == 21600
    assert by_name["fast-6h"].short_window_seconds == 1800  # 21600 / 12

    assert by_name["slow-3d"].burn_rate_threshold == 1.0
    assert by_name["slow-3d"].severity == "TICKET"
    assert by_name["slow-3d"].short_window_seconds == 21600  # 259200 / 12

    # Every pair keeps the 1/12 short:long ratio.
    for tier in PRODUCTION_PROFILE.tiers:
        assert tier.short_window_seconds == pytest.approx(tier.long_window_seconds / 12)


def test_demo_profile_preserves_thresholds_but_compresses_windows():
    prod_thresholds = sorted(t.burn_rate_threshold for t in PRODUCTION_PROFILE.tiers)
    demo_thresholds = sorted(t.burn_rate_threshold for t in DEMO_PROFILE.tiers)
    assert prod_thresholds == demo_thresholds  # same 14.4 / 6 / 1
    # ... but the windows are seconds-scale, not hours/days.
    assert max(t.long_window_seconds for t in DEMO_PROFILE.tiers) <= 60


def test_load_config_from_repo_yaml_file():
    cfg = load_config("config/watch_targets.yaml")
    assert len(cfg.targets) >= 1
    names = {t.name for t in cfg.targets}
    assert "rag-ops-health" in names
