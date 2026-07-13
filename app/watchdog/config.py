"""Watchdog configuration: targets + burn-rate window profiles.

Targets and the alerting backend selector are read from a YAML file (see
``config/watch_targets.yaml``) — URLs are NEVER hardcoded in code. Endpoints,
topics and tokens for the push sink are read from the environment at send time
(see :mod:`app.watchdog.alerts`).

The burn-rate window profiles implement the Google SRE Workbook chapter
"Alerting on SLOs" — the multi-window, multi-burn-rate approach
(https://sre.google/workbook/alerting-on-slos/). See :data:`PRODUCTION_PROFILE`
for the canonical 30-day table and :data:`DEMO_PROFILE` for a compressed,
seconds-scale profile so the capability is demonstrable/testable quickly.
"""
from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - trivial import shim
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Burn-rate windows (SRE "Alerting on SLOs", multi-window multi-burn-rate)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BurnRateTier:
    """One multi-window burn-rate alerting tier.

    An alert for this tier fires only when BOTH the ``long_window`` and the
    ``short_window`` burn rate meet/exceed ``burn_rate_threshold``. The long
    window gives a strong, low-noise signal; the short window (canonically 1/12
    of the long window) ensures the condition is STILL active, which
    auto-resolves stale alerts once the burn subsides.

    ``budget_consumed_pct`` documents how much of a 30-day error budget the
    threshold corresponds to over the long window (production semantics):
    ``threshold * long_window / slo_window``.
    """

    name: str
    severity: str  # "PAGE" | "TICKET"
    long_window_seconds: float
    short_window_seconds: float
    burn_rate_threshold: float
    budget_consumed_pct: float
    # Minimum probe count required in EACH window before the tier is eligible to
    # breach — mirrors ``CanaryController``'s min-sample floor so a lone failed
    # probe (error_rate 1.0 → huge burn rate) can never page.
    min_samples: int = 1


@dataclass(frozen=True)
class WindowProfile:
    """A named set of burn-rate tiers plus the SLO (error-budget) window."""

    name: str
    slo_window_seconds: float
    # Window over which p99 latency is evaluated against the hard latency SLO.
    latency_window_seconds: float
    # Min samples before the hard p99 latency SLO is eligible to breach.
    latency_min_samples: int
    tiers: tuple[BurnRateTier, ...]

    def fast_tiers(self) -> tuple[BurnRateTier, ...]:
        return tuple(t for t in self.tiers if t.severity == "PAGE")

    def slow_tiers(self) -> tuple[BurnRateTier, ...]:
        return tuple(t for t in self.tiers if t.severity == "TICKET")


_MONTH_SECONDS = 30 * 24 * 3600  # 2,592,000 s (30-day SLO window)

# Canonical Google SRE Workbook table for a 30-day window:
#   fast-burn: 2% budget in 1h  -> burn rate 14.4, windows 1h / 5m   -> PAGE
#   fast-burn: 5% budget in 6h  -> burn rate 6,    windows 6h / 30m  -> PAGE
#   slow-burn: 10% budget in 3d -> burn rate 1,    windows 3d / 6h   -> TICKET
# In every pair the short window is 1/12 of the long window.
PRODUCTION_PROFILE = WindowProfile(
    name="production",
    slo_window_seconds=_MONTH_SECONDS,
    latency_window_seconds=3600,
    latency_min_samples=60,
    tiers=(
        BurnRateTier("fast-1h", "PAGE", 3600, 300, 14.4, 2.0, min_samples=5),
        BurnRateTier("fast-6h", "PAGE", 21600, 1800, 6.0, 5.0, min_samples=5),
        BurnRateTier("slow-3d", "TICKET", 259200, 21600, 1.0, 10.0, min_samples=20),
    ),
)

# Accelerated profile: SAME burn-rate thresholds (14.4 / 6 / 1) and the same 1/12
# short:long ratio, but compressed to seconds so a fast-burn PAGE can be provoked
# in ~12s of sustained failure and a slow-burn TICKET in ~36s. The slo_window is
# compressed too so the error-budget-remaining gauge visibly drains during a
# demo; the budget-percent labels are therefore illustrative, not physical — only
# the thresholds and the two-window logic carry production meaning.
DEMO_PROFILE = WindowProfile(
    name="demo",
    slo_window_seconds=120,
    latency_window_seconds=12,
    latency_min_samples=5,
    tiers=(
        BurnRateTier("fast-1", "PAGE", 12, 1, 14.4, 2.0, min_samples=3),
        BurnRateTier("fast-2", "PAGE", 24, 2, 6.0, 5.0, min_samples=3),
        BurnRateTier("slow", "TICKET", 36, 3, 1.0, 10.0, min_samples=5),
    ),
)

PROFILES: dict[str, WindowProfile] = {
    PRODUCTION_PROFILE.name: PRODUCTION_PROFILE,
    DEMO_PROFILE.name: DEMO_PROFILE,
}


def get_profile(name: str) -> WindowProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown window profile {name!r}; choose one of {sorted(PROFILES)}"
        ) from None


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
@dataclass
class Target:
    """A single service to probe on its own interval."""

    name: str
    url: str
    method: str = "GET"
    expected_status: int = 200
    body_check: str | None = None  # optional substring the response body must contain
    latency_slo_ms: float = 1000.0
    availability_slo: float = 0.995
    interval_seconds: float = 30.0
    timeout_seconds: float = 10.0
    # TLS certificate verification. Default True (secure). Set False only for
    # endpoints with self-signed certs or when the local CA store cannot verify
    # the chain — an escape hatch, not a default.
    verify_tls: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.availability_slo < 1.0:
            raise ValueError(
                f"target {self.name!r}: availability_slo must be in (0, 1), "
                f"got {self.availability_slo}"
            )
        if self.latency_slo_ms <= 0:
            raise ValueError(f"target {self.name!r}: latency_slo_ms must be > 0")


@dataclass
class WatchdogConfig:
    targets: list[Target]
    profile: WindowProfile
    alert_backend: str = "ntfy"


def _target_from_dict(raw: dict) -> Target:
    known = {
        "name",
        "url",
        "method",
        "expected_status",
        "body_check",
        "latency_slo_ms",
        "availability_slo",
        "interval_seconds",
        "timeout_seconds",
        "verify_tls",
    }
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"target {raw.get('name')!r}: unknown keys {sorted(unknown)}")
    return Target(**raw)


def load_config(path: str, profile_override: str | None = None) -> WatchdogConfig:
    """Load a :class:`WatchdogConfig` from a YAML file.

    ``profile_override`` (e.g. from ``--profile demo``) wins over the file's
    ``profile`` key so the accelerated windows can be selected at the CLI.
    """
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load watchdog config; pip install pyyaml")

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return config_from_dict(raw, profile_override=profile_override)


def config_from_dict(raw: dict, profile_override: str | None = None) -> WatchdogConfig:
    profile_name = profile_override or raw.get("profile", "production")
    profile = get_profile(profile_name)

    alerting = raw.get("alerting") or {}
    backend = alerting.get("backend", "ntfy")

    targets_raw = raw.get("targets") or []
    if not targets_raw:
        raise ValueError("watchdog config lists no targets")
    targets = [_target_from_dict(t) for t in targets_raw]

    names = [t.name for t in targets]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate target names in config: {sorted(dupes)}")

    return WatchdogConfig(targets=targets, profile=profile, alert_backend=backend)
