"""Synthetic HTTP prober.

Probes a :class:`~app.watchdog.config.Target` once, measuring up/down (status
code + optional body assertion) and latency. Timeouts, connection errors and any
other transport failure are recorded as a FAILED probe — the prober never raises
on a down target, so a dead service degrades the error budget instead of
crashing the daemon.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

try:  # httpx is already a project dependency; import lazily for hermetic tests.
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

from app.watchdog.config import Target


@dataclass
class ProbeResult:
    """Outcome of a single probe."""

    target_name: str
    success: bool
    status_code: int | None
    latency_ms: float
    error: str | None = None
    body_ok: bool = True


class _Response(Protocol):
    status_code: int

    # Read-only property (not a settable attribute) so httpx.Response — whose
    # ``text`` is a read-only property — structurally satisfies this Protocol.
    @property
    def text(self) -> str: ...


# A request function takes (method, url, timeout) and returns a response-like
# object exposing ``status_code`` and ``text``. Injectable so tests never touch
# the network.
RequestFn = Callable[..., _Response]


def _httpx_request(  # pragma: no cover
    method: str, url: str, timeout: float, verify: bool = True
) -> _Response:
    if httpx is None:
        raise RuntimeError("httpx is required for live probing; pip install httpx")
    return httpx.request(method, url, timeout=timeout, follow_redirects=True, verify=verify)


class Prober:
    """Probes targets over HTTP. ``request_fn`` is injectable for tests."""

    def __init__(
        self,
        request_fn: RequestFn | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._request_fn = request_fn or _httpx_request
        self._clock = clock

    def probe(self, target: Target) -> ProbeResult:
        start = self._clock()
        try:
            resp = self._request_fn(
                target.method,
                target.url,
                timeout=target.timeout_seconds,
                verify=target.verify_tls,
            )
        except Exception as exc:  # noqa: BLE001 - any transport error is a failed probe
            latency_ms = (self._clock() - start) * 1000.0
            return ProbeResult(
                target_name=target.name,
                success=False,
                status_code=None,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
                body_ok=False,
            )

        latency_ms = (self._clock() - start) * 1000.0
        status_ok = resp.status_code == target.expected_status
        body_ok = target.body_check is None or target.body_check in getattr(resp, "text", "")

        error: str | None = None
        if not status_ok:
            error = f"unexpected status {resp.status_code} (want {target.expected_status})"
        elif not body_ok:
            error = f"body check {target.body_check!r} not found in response"

        return ProbeResult(
            target_name=target.name,
            success=status_ok and body_ok,
            status_code=resp.status_code,
            latency_ms=latency_ms,
            error=error,
            body_ok=body_ok,
        )
