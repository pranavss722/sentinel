"""Synthetic-monitoring watchdog.

Reuses Sentinel's monitoring engine (the PSI drift math from :mod:`app.drift`
and the SLO / min-sample concepts from :mod:`app.canary`) to probe *real*
deployed HTTP services, track availability + latency against SLOs with SRE-style
multi-window multi-burn-rate error-budget alerting, catch latency regressions via
PSI, and push alerts to the operator's phone.

This package is deliberately ADDITIVE: it does not modify or import-mutate the
canary/serving code. It only *reuses* the PSI engine and the SLO philosophy.
"""
