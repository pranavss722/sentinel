"""Push-notification alert sink.

Sends a push notification on a state TRANSITION into a PAGE/TICKET condition and
a single recovery notification on return to OK (dedup is the caller's job — see
:class:`~app.watchdog.monitor.TargetMonitor`). Two backends:

  * **ntfy** (default): HTTP POST to ``{NTFY_SERVER}/{NTFY_TOPIC}``. Server
    defaults to https://ntfy.sh; topic comes from ``NTFY_TOPIC``.
  * **Pushover** (alternative, env-gated): POST to the Pushover messages API
    using ``PUSHOVER_TOKEN`` + ``PUSHOVER_USER``.

Endpoints/topics/tokens are read from the environment — never hardcoded. If the
selected backend is unconfigured the alert is LOGGED instead of pushed, and the
sink NEVER raises: a failing push must not take down the watchdog.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# http_post(url, *, data, headers) -> None. Injectable so tests never hit network.
HttpPost = Callable[..., object]

NTFY_PRIORITY = {"PAGE": "urgent", "TICKET": "default", "RECOVERY": "low"}
NTFY_TAGS = {"PAGE": "rotating_light", "TICKET": "ticket", "RECOVERY": "white_check_mark"}
PUSHOVER_PRIORITY = {"PAGE": 1, "TICKET": 0, "RECOVERY": -1}


@dataclass
class Alert:
    target: str
    severity: str  # "PAGE" | "TICKET" | "RECOVERY"
    reason: str
    availability: float
    budget_remaining: float
    burn_rate: float
    p99_latency_ms: float | None = None
    is_recovery: bool = False


def _default_post(url: str, *, data: str, headers: dict[str, str]) -> None:  # pragma: no cover
    if httpx is None:
        raise RuntimeError("httpx is required to push alerts; pip install httpx")
    httpx.post(url, content=data.encode("utf-8"), headers=headers, timeout=10.0)


def format_message(alert: Alert) -> tuple[str, str]:
    """Return (title, body) describing what breached and the budget/burn state."""
    if alert.is_recovery:
        title = f"[RECOVERY] {alert.target} healthy again"
    else:
        title = f"[{alert.severity}] {alert.target}: {alert.reason}"

    lines = [
        f"Service : {alert.target}",
        f"Status  : {'RECOVERED' if alert.is_recovery else alert.severity}",
        f"What    : {alert.reason}",
        f"Avail   : {alert.availability * 100:.3f}%",
        f"Budget  : {alert.budget_remaining * 100:.1f}% remaining",
        f"BurnRate: {alert.burn_rate:.2f}x",
    ]
    if alert.p99_latency_ms is not None:
        lines.append(f"p99     : {alert.p99_latency_ms:.0f} ms")
    return title, "\n".join(lines)


class AlertSink:
    """Formats and dispatches alerts to a push backend (or logs if unconfigured)."""

    def __init__(
        self,
        backend: str = "ntfy",
        env: dict[str, str] | None = None,
        http_post: HttpPost | None = None,
    ) -> None:
        self._backend = backend
        self._env = os.environ if env is None else env
        self._http_post = http_post or _default_post

    # -- configuration ----------------------------------------------------- #
    def _ntfy_topic(self) -> str | None:
        return self._env.get("NTFY_TOPIC") or None

    def _ntfy_server(self) -> str:
        return self._env.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")

    def _pushover_creds(self) -> tuple[str | None, str | None]:
        return self._env.get("PUSHOVER_TOKEN") or None, self._env.get("PUSHOVER_USER") or None

    def is_configured(self) -> bool:
        if self._backend == "ntfy":
            return self._ntfy_topic() is not None
        if self._backend == "pushover":
            token, user = self._pushover_creds()
            return bool(token and user)
        return False

    # -- dispatch ---------------------------------------------------------- #
    def send(self, alert: Alert) -> bool:
        """Dispatch an alert. Returns True if a push was attempted, False if it
        was only logged. NEVER raises."""
        title, body = format_message(alert)
        try:
            if not self.is_configured():
                logger.warning(
                    "watchdog alert (%s backend unconfigured - logging only)\n%s\n%s",
                    self._backend,
                    title,
                    body,
                )
                return False

            if self._backend == "ntfy":
                self._send_ntfy(alert, title, body)
            elif self._backend == "pushover":
                self._send_pushover(alert, title, body)
            else:  # pragma: no cover - is_configured already gates unknown backends
                logger.warning("unknown alert backend %r - logging only\n%s", self._backend, body)
                return False

            logger.info("watchdog alert pushed via %s: %s", self._backend, title)
            return True
        except Exception as exc:  # noqa: BLE001 - a failing push must never crash the daemon
            logger.error("watchdog alert push FAILED (%s): %s\n%s", self._backend, exc, body)
            return False

    def _send_ntfy(self, alert: Alert, title: str, body: str) -> None:
        topic = self._ntfy_topic()
        url = f"{self._ntfy_server()}/{topic}"
        headers = {
            "Title": title,
            "Priority": NTFY_PRIORITY.get(alert.severity, "default"),
            "Tags": NTFY_TAGS.get(alert.severity, "warning"),
        }
        self._http_post(url, data=body, headers=headers)

    def _send_pushover(self, alert: Alert, title: str, body: str) -> None:
        token, user = self._pushover_creds()
        form = (
            f"token={token}&user={user}"
            f"&priority={PUSHOVER_PRIORITY.get(alert.severity, 0)}"
            f"&title={title}&message={body}"
        )
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        self._http_post("https://api.pushover.net/1/messages.json", data=form, headers=headers)
