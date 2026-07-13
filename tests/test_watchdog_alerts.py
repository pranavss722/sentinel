"""Tests for the push alert sink — formatting, ntfy/pushover, graceful no-op."""
from __future__ import annotations

from app.watchdog.alerts import Alert, AlertSink, format_message


def make_alert(**kw) -> Alert:
    base = dict(
        target="rag-ops-health",
        severity="PAGE",
        reason="fast-burn fast-1h (burn 30.0x)",
        availability=0.97,
        budget_remaining=0.0,
        burn_rate=30.0,
        p99_latency_ms=850.0,
    )
    base.update(kw)
    return Alert(**base)


class Capture:
    def __init__(self):
        self.calls = []

    def __call__(self, url, *, data, headers):
        self.calls.append({"url": url, "data": data, "headers": headers})


def test_format_message_states_service_breach_budget_and_burn():
    title, body = format_message(make_alert())
    assert "PAGE" in title
    assert "rag-ops-health" in title
    assert "fast-burn" in body
    assert "0.0% remaining" in body
    assert "30.00x" in body
    assert "850 ms" in body


def test_recovery_message_is_labelled():
    title, body = format_message(make_alert(severity="RECOVERY", is_recovery=True))
    assert "RECOVERY" in title
    assert "RECOVERED" in body


def test_ntfy_backend_posts_to_server_topic_with_headers():
    cap = Capture()
    sink = AlertSink(backend="ntfy", env={"NTFY_TOPIC": "my-phone"}, http_post=cap)
    assert sink.is_configured() is True

    pushed = sink.send(make_alert())
    assert pushed is True
    assert len(cap.calls) == 1
    call = cap.calls[0]
    assert call["url"] == "https://ntfy.sh/my-phone"
    assert call["headers"]["Priority"] == "urgent"  # PAGE -> urgent
    assert "rag-ops-health" in call["headers"]["Title"]


def test_ntfy_respects_custom_server():
    cap = Capture()
    sink = AlertSink(
        backend="ntfy",
        env={"NTFY_TOPIC": "t", "NTFY_SERVER": "https://ntfy.example.com/"},
        http_post=cap,
    )
    sink.send(make_alert(severity="TICKET"))
    assert cap.calls[0]["url"] == "https://ntfy.example.com/t"
    assert cap.calls[0]["headers"]["Priority"] == "default"  # TICKET


def test_pushover_backend_posts_when_creds_present():
    cap = Capture()
    sink = AlertSink(
        backend="pushover",
        env={"PUSHOVER_TOKEN": "tok", "PUSHOVER_USER": "usr"},
        http_post=cap,
    )
    assert sink.is_configured() is True
    sink.send(make_alert())
    assert cap.calls[0]["url"] == "https://api.pushover.net/1/messages.json"
    assert "token=tok" in cap.calls[0]["data"]
    assert "user=usr" in cap.calls[0]["data"]


def test_unconfigured_sink_logs_and_does_not_push():
    cap = Capture()
    sink = AlertSink(backend="ntfy", env={}, http_post=cap)  # no NTFY_TOPIC
    assert sink.is_configured() is False
    pushed = sink.send(make_alert())
    assert pushed is False
    assert cap.calls == []  # nothing sent


def test_send_never_raises_when_transport_fails():
    def boom(url, *, data, headers):
        raise ConnectionError("network down")

    sink = AlertSink(backend="ntfy", env={"NTFY_TOPIC": "t"}, http_post=boom)
    # Must swallow the error and report a non-push rather than crashing.
    assert sink.send(make_alert()) is False
