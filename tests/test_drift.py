"""Tests for DriftMonitor."""
import numpy as np
import pandas as pd


def make_monitor(feature_names=None):
    from app.drift import DriftMonitor
    names = feature_names or [f"f{i}" for i in range(3)]
    return DriftMonitor(feature_names=names, psi_threshold=0.2, kl_threshold=0.1)


def test_no_drift_on_identical_distributions():
    monitor = make_monitor()
    ref = pd.DataFrame(np.random.RandomState(42).randn(1000, 3), columns=["f0", "f1", "f2"])
    monitor.set_reference(ref, ref_scores=np.random.RandomState(42).rand(1000))

    # Current data drawn from same distribution
    cur = pd.DataFrame(np.random.RandomState(43).randn(500, 3), columns=["f0", "f1", "f2"])
    cur_scores = np.random.RandomState(43).rand(500)
    monitor.update(cur, cur_scores)
    assert monitor.should_rollback() is False


def test_data_drift_triggers_rollback_on_shifted_feature():
    monitor = make_monitor()
    rng = np.random.RandomState(42)
    ref = pd.DataFrame(rng.randn(1000, 3), columns=["f0", "f1", "f2"])
    monitor.set_reference(ref, ref_scores=rng.rand(1000))

    # Shift f1 by +10 — PSI should exceed 0.2
    cur = pd.DataFrame(rng.randn(500, 3), columns=["f0", "f1", "f2"])
    cur["f1"] = cur["f1"] + 10.0
    cur_scores = rng.rand(500)
    monitor.update(cur, cur_scores)
    assert monitor.should_rollback() is True


def test_prediction_drift_triggers_rollback():
    monitor = make_monitor()
    rng = np.random.RandomState(42)
    ref = pd.DataFrame(rng.randn(1000, 3), columns=["f0", "f1", "f2"])
    ref_scores = rng.uniform(0.4, 0.6, 1000)  # narrow distribution
    monitor.set_reference(ref, ref_scores=ref_scores)

    cur = pd.DataFrame(rng.randn(500, 3), columns=["f0", "f1", "f2"])
    # Wildly different prediction distribution
    cur_scores = rng.uniform(0.0, 0.1, 500)
    monitor.update(cur, cur_scores)
    assert monitor.should_rollback() is True


def test_psi_values_exposed_per_feature():
    monitor = make_monitor()
    rng = np.random.RandomState(42)
    ref = pd.DataFrame(rng.randn(1000, 3), columns=["f0", "f1", "f2"])
    monitor.set_reference(ref, ref_scores=rng.rand(1000))

    cur = pd.DataFrame(rng.randn(500, 3), columns=["f0", "f1", "f2"])
    monitor.update(cur, rng.rand(500))
    psi_values = monitor.get_psi_values()
    assert set(psi_values.keys()) == {"f0", "f1", "f2"}
    assert all(isinstance(v, float) for v in psi_values.values())


def test_kl_divergence_exposed():
    monitor = make_monitor()
    rng = np.random.RandomState(42)
    ref = pd.DataFrame(rng.randn(1000, 3), columns=["f0", "f1", "f2"])
    monitor.set_reference(ref, ref_scores=rng.rand(1000))

    cur = pd.DataFrame(rng.randn(500, 3), columns=["f0", "f1", "f2"])
    monitor.update(cur, rng.rand(500))
    kl = monitor.get_kl_divergence()
    assert isinstance(kl, float)
    assert kl >= 0.0


def test_should_rollback_false_before_any_data():
    monitor = make_monitor()
    assert monitor.should_rollback() is False
