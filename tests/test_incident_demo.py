"""Regression test for the dual-scenario autonomous-rollback showpiece.

Runs the real incident scenarios from ``scripts/demo_incident.py`` headlessly
and asserts BOTH rollback paths and the deliberate asymmetric policy:

  * Scenario A (drift): rolls back IMMEDIATELY — 0 polls withheld.
  * Scenario B (SLO):   WITHHOLDS rollback until the breach is confirmed on
    N consecutive polls — N-1 polls withheld.

It also asserts the state-hygiene invariant (windows + debounce + drift verdict
reset on rollback) and determinism across repeat runs.
"""
from __future__ import annotations

import pytest

from scripts.demo_incident import DemoResult, run_incident


@pytest.fixture(scope="module")
def demo() -> DemoResult:
    return run_incident()


# --------------------------------------------------------------------------- #
# Scenario A — DRIFT: immediate rollback
# --------------------------------------------------------------------------- #
def test_drift_scenario_rolls_back_immediately(demo: DemoResult):
    d = demo.drift
    assert d.rolled_back is True
    assert d.canary_weight_final == 0.0
    assert d.rollback_count_delta == 1
    assert d.polls_until_rollback == 1
    assert d.polls_withheld == 0  # drift is NOT debounced


def test_drift_scenario_breach_evidence_captured_pre_rollback(demo: DemoResult):
    d = demo.drift
    assert d.drift_breached_at_detection is True
    assert d.gauge_psi_drifted > 0.20  # captured before rollback cleared state
    assert d.drift_kl_at_detection > 0.10


def test_drift_scenario_slo_stayed_quiet(demo: DemoResult):
    d = demo.drift
    assert d.slo_breached_at_detection is False
    assert d.p99_latency_ms_at_detection <= 200.0


# --------------------------------------------------------------------------- #
# Scenario B — SLO: debounced rollback
# --------------------------------------------------------------------------- #
def test_slo_scenario_withholds_then_rolls_back(demo: DemoResult):
    s = demo.slo
    assert s.rolled_back is True
    assert s.canary_weight_final == 0.0
    assert s.rollback_count_delta == 1
    assert s.consecutive_breaches_required >= 2  # a real debounce, not 1
    assert s.polls_until_rollback == s.consecutive_breaches_required
    assert s.polls_withheld == s.consecutive_breaches_required - 1


def test_slo_scenario_breach_evidence_captured_pre_rollback(demo: DemoResult):
    s = demo.slo
    assert s.slo_breached_at_detection is True
    assert s.p99_latency_ms_at_detection > 200.0
    # Judged over a window above the min-sample floor — captured pre-clearing.
    assert s.slo_sample_count_at_detection >= s.min_samples_required
    assert s.min_samples_required >= 200


def test_slo_scenario_drift_stayed_quiet(demo: DemoResult):
    s = demo.slo
    assert s.drift_breached_at_detection is False
    assert s.gauge_psi_drifted <= 0.20


# --------------------------------------------------------------------------- #
# The headline: asymmetric policy — drift = 0 withheld, SLO = N-1 withheld
# --------------------------------------------------------------------------- #
def test_asymmetric_policy_drift_immediate_slo_debounced(demo: DemoResult):
    assert demo.drift.polls_withheld == 0
    assert demo.slo.polls_withheld == demo.slo.consecutive_breaches_required - 1
    # Drift acts strictly sooner than SLO.
    assert demo.drift.polls_until_rollback < demo.slo.polls_until_rollback


# --------------------------------------------------------------------------- #
# Shared invariants across both scenarios
# --------------------------------------------------------------------------- #
def test_both_scenarios_restore_champion(demo: DemoResult):
    for sr in demo.scenarios:
        assert sr.challenger_decisions_after == 0
        assert sr.champion_decisions_after > 0
        assert sr.champion_share_after == 100.0


def test_both_scenarios_clean_state_after_rollback(demo: DemoResult):
    for sr in demo.scenarios:
        assert sr.latencies_after_rollback == 0
        assert sr.errors_after_rollback == 0
        assert sr.consecutive_after_rollback == 0
        assert sr.drift_should_rollback_after is False
        assert sr.hygiene_clean is True


def test_timelines_have_expected_events(demo: DemoResult):
    dkinds = [e.kind for e in demo.drift.events]
    assert "rollback" in dkinds
    assert "breach_withheld" not in dkinds  # drift never withholds

    skinds = [e.kind for e in demo.slo.events]
    assert skinds.count("breach_withheld") == demo.slo.consecutive_breaches_required - 1
    assert skinds.index("breach_withheld") < skinds.index("rollback") < skinds.index("recovered")


def test_incident_is_deterministic():
    """Both scenarios must produce identical evidence on repeat runs."""
    a = run_incident()
    b = run_incident()
    for x, y in zip(a.scenarios, b.scenarios):
        assert x.key == y.key
        assert x.polls_until_rollback == y.polls_until_rollback
        assert x.polls_withheld == y.polls_withheld
        assert x.p99_latency_ms_at_detection == pytest.approx(y.p99_latency_ms_at_detection)
        assert x.gauge_psi_drifted == pytest.approx(y.gauge_psi_drifted)
        assert x.champion_share_after == y.champion_share_after == 100.0
