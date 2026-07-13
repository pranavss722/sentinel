"""Tests for the error-budget + multi-window multi-burn-rate SLO engine.

These lock in the SRE Workbook math: burn rate = error_rate / (1 - SLO), a tier
breaches only when BOTH its long and short windows exceed the threshold, a
single transient failure stays below the fast threshold (no page), and the hard
p99 latency SLO is min-sample gated.
"""
from __future__ import annotations

import pytest

from app.watchdog.config import BurnRateTier, WindowProfile
from app.watchdog.slo import ErrorBudgetTracker

# A compact, deterministic profile: fast PAGE (long 100s / short 10s, thr 14.4)
# and slow TICKET (long 200s / short 20s, thr 1). SLO window 500s; p99 latency
# evaluated over 100s with a 5-sample floor.
PROFILE = WindowProfile(
    name="test",
    slo_window_seconds=500,
    latency_window_seconds=100,
    latency_min_samples=5,
    tiers=(
        BurnRateTier("fast", "PAGE", 100, 10, 14.4, 2.0, min_samples=5),
        BurnRateTier("slow", "TICKET", 200, 20, 1.0, 10.0, min_samples=5),
    ),
)


def make_tracker(availability_slo=0.99, latency_slo_ms=200.0):
    # clock is unused: every record/evaluate gets an explicit timestamp.
    return ErrorBudgetTracker(
        availability_slo=availability_slo,
        latency_slo_ms=latency_slo_ms,
        profile=PROFILE,
        clock=lambda: 0.0,
    )


def _fill(tracker, *, start, count, fail_every=None, latency_ms=50.0):
    """Add ``count`` probes at 1s spacing ending at ``start``+count-1."""
    for i in range(count):
        fail = fail_every is not None and (i % fail_every == 0)
        tracker.record(success=not fail, latency_ms=latency_ms, timestamp=start + i)


def test_healthy_service_has_full_budget_and_no_alert():
    t = make_tracker()
    _fill(t, start=900, count=100, fail_every=None)
    ev = t.evaluate(now=1000)
    assert ev.availability == pytest.approx(1.0)
    assert ev.error_budget_remaining == pytest.approx(1.0)
    assert ev.burn_rate == pytest.approx(0.0)
    assert ev.page is False and ev.ticket is False
    assert ev.worst_severity == "OK"


def test_burn_rate_math_matches_error_rate_over_budget():
    # 20% of probes failing, SLO 0.99 -> budget 0.01 -> burn rate 20.
    t = make_tracker(availability_slo=0.99)
    _fill(t, start=901, count=100, fail_every=5)  # every 5th fails -> 20/100
    ev = t.evaluate(now=1000)
    fast = next(x for x in ev.tiers if x.name == "fast")
    assert fast.long_burn_rate == pytest.approx(20.0)
    assert ev.burn_rate == pytest.approx(20.0)


def test_fast_burn_pages_when_both_windows_breach():
    t = make_tracker(availability_slo=0.99)  # budget 0.01
    # 30% failing across the whole 100s long window (and thus the last 10s too).
    _fill(t, start=901, count=100, fail_every=3)  # ~34/100
    ev = t.evaluate(now=1000)
    fast = next(x for x in ev.tiers if x.name == "fast")
    assert fast.long_burn_rate >= 14.4
    assert fast.short_burn_rate >= 14.4
    assert fast.breaching is True
    assert ev.page is True
    assert ev.worst_severity == "PAGE"


def test_single_transient_failure_below_threshold_does_not_page():
    t = make_tracker(availability_slo=0.99)
    _fill(t, start=901, count=100, fail_every=None)  # all healthy
    # one lone failure at the newest timestamp
    t.record(success=False, latency_ms=50.0, timestamp=1000)
    ev = t.evaluate(now=1000)
    fast = next(x for x in ev.tiers if x.name == "fast")
    # long window ~1/101 fail -> burn ~1; short window 1/11 -> burn ~9; neither
    # clears the 14.4 fast threshold.
    assert fast.long_burn_rate < 14.4
    assert fast.breaching is False
    assert ev.page is False


def test_slow_burn_raises_ticket_not_page():
    t = make_tracker(availability_slo=0.99)  # budget 0.01
    # 5% failing, evenly spread so BOTH slow windows breach -> burn 5: clears the
    # slow threshold (1) but not the fast one (14.4).
    _fill(t, start=801, count=200, fail_every=20)  # 10/200 = 5%
    ev = t.evaluate(now=1000)
    slow = next(x for x in ev.tiers if x.name == "slow")
    fast = next(x for x in ev.tiers if x.name == "fast")
    assert slow.breaching is True
    assert fast.breaching is False
    assert ev.ticket is True
    assert ev.page is False
    assert ev.worst_severity == "TICKET"


def test_min_sample_floor_blocks_breach_from_thin_window():
    t = make_tracker(availability_slo=0.99)
    # Only 3 probes (< min_samples 5), all failing — a wild burn rate that must
    # NOT breach because the window is too thin to trust.
    _fill(t, start=998, count=3, fail_every=1)
    ev = t.evaluate(now=1000)
    fast = next(x for x in ev.tiers if x.name == "fast")
    assert fast.breaching is False
    assert ev.page is False


def test_hard_p99_latency_slo_breach_pages_with_min_samples():
    t = make_tracker(latency_slo_ms=200.0)
    # 10 successful-but-slow probes (500ms) — availability fine, latency breached.
    _fill(t, start=990, count=10, fail_every=None, latency_ms=500.0)
    ev = t.evaluate(now=1000)
    assert ev.p99_latency_ms == pytest.approx(500.0)
    assert ev.latency_slo_breached is True
    assert ev.page is True


def test_p99_latency_not_evaluated_below_min_samples():
    t = make_tracker(latency_slo_ms=200.0)
    _fill(t, start=997, count=3, fail_every=None, latency_ms=999.0)  # < 5 samples
    ev = t.evaluate(now=1000)
    assert ev.p99_latency_ms is None
    assert ev.latency_slo_breached is False
    assert ev.page is False


def test_error_budget_drains_with_failures():
    t = make_tracker(availability_slo=0.99)  # budget 0.01
    # 1 failure in 100 probes over the SLO window -> consumes exactly 100%.
    _fill(t, start=901, count=100, fail_every=100)  # index 0 fails -> 1/100
    ev = t.evaluate(now=1000)
    assert ev.availability == pytest.approx(0.99)
    assert ev.error_budget_remaining == pytest.approx(0.0)


def test_stale_failures_fall_out_of_burn_window():
    t = make_tracker(availability_slo=0.99)
    # Heavy failures long ago, only healthy probes recently.
    _fill(t, start=100, count=50, fail_every=1)  # all fail, ~t=100..149
    _fill(t, start=960, count=50, fail_every=None)  # healthy, recent
    ev = t.evaluate(now=1000)
    fast = next(x for x in ev.tiers if x.name == "fast")
    assert fast.long_burn_rate == pytest.approx(0.0)  # old failures aged out
    assert ev.page is False
