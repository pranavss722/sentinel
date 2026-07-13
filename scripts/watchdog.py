"""Sentinel Watchdog runner — synthetic monitoring of real deployed services.

Runs the probe -> evaluate (SLO/error-budget/burn-rate + latency-drift) -> alert
loop against the targets in ``config/watch_targets.yaml``.

Usage::

    # one probe cycle across every target (CI / cron / smoke) — exits non-zero
    # if any target is paging:
    python scripts/watchdog.py --once

    # accelerated demo windows:
    python scripts/watchdog.py --once --profile demo

    # long-running daemon (probes each target on its own interval):
    python scripts/watchdog.py

Push alerting is configured via env (see app/watchdog/alerts.py); if unset, the
watchdog logs alerts instead of pushing — it never crashes on a down target.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure the repo root is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.watchdog.config import load_config  # noqa: E402
from app.watchdog.monitor import WatchdogEngine  # noqa: E402

DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "watch_targets.yaml")


def _print_status_table(statuses) -> None:
    header = f"{'TARGET':22} {'UP':>4} {'STATE':>7} {'AVAIL%':>9} {'BUDGET%':>9} {'BURN':>7}  REASON"
    print(header)
    print("-" * len(header))
    for s in statuses:
        p99 = f"{s.p99_latency_ms:.0f}ms" if s.p99_latency_ms is not None else "n/a"
        print(
            f"{s.target:22} {('UP' if s.up else 'DOWN'):>4} {s.state:>7} "
            f"{s.availability * 100:>8.3f} {s.budget_remaining * 100:>8.1f} "
            f"{s.burn_rate:>6.2f}x  {s.reason} [p99 {p99}]"
        )


def run_once(config_path: str, profile: str | None) -> int:
    config = load_config(config_path, profile_override=profile)
    engine = WatchdogEngine(config)
    logging.info(
        "watchdog --once: %d targets, profile=%s, backend=%s",
        len(config.targets),
        config.profile.name,
        config.alert_backend,
    )
    statuses = engine.probe_all_once()
    _print_status_table(statuses)
    # Non-zero exit if anything is paging, so cron/CI can react.
    return 1 if any(s.state == "PAGE" for s in statuses) else 0


def run_daemon(config_path: str, profile: str | None) -> int:
    config = load_config(config_path, profile_override=profile)
    engine = WatchdogEngine(config)
    logging.info(
        "watchdog daemon: %d targets, profile=%s, backend=%s",
        len(config.targets),
        config.profile.name,
        config.alert_backend,
    )
    try:
        asyncio.run(engine.run_forever())
    except KeyboardInterrupt:
        logging.info("watchdog stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sentinel synthetic-monitoring watchdog")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to watch_targets.yaml")
    parser.add_argument(
        "--profile",
        default=None,
        choices=["production", "demo"],
        help="burn-rate window profile override",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run a single probe cycle and exit (for CI/cron)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.once:
        return run_once(args.config, args.profile)
    return run_daemon(args.config, args.profile)


if __name__ == "__main__":
    raise SystemExit(main())
