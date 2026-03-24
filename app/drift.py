"""Data and prediction drift monitoring via PSI and KL divergence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.metrics import data_drift_psi, prediction_drift_kl_divergence


def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D arrays."""
    eps = 1e-6
    combined = np.concatenate([reference, current])
    breakpoints = np.histogram_bin_edges(combined, bins=bins)
    ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(current, bins=breakpoints)[0].astype(float)

    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def _compute_kl(ref_scores: np.ndarray, cur_scores: np.ndarray, bins: int = 10) -> float:
    """KL divergence between two score distributions."""
    eps = 1e-6
    combined = np.concatenate([ref_scores, cur_scores])
    breakpoints = np.histogram_bin_edges(combined, bins=bins)
    ref_counts = np.histogram(ref_scores, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(cur_scores, bins=breakpoints)[0].astype(float)

    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps

    kl = float(np.sum(ref_pct * np.log(ref_pct / cur_pct)))
    return max(kl, 0.0)


class DriftMonitor:
    """Monitors data drift (PSI per feature) and prediction drift (KL divergence)."""

    def __init__(
        self,
        feature_names: list[str],
        psi_threshold: float = 0.2,
        kl_threshold: float = 0.1,
    ) -> None:
        self._feature_names = feature_names
        self._psi_threshold = psi_threshold
        self._kl_threshold = kl_threshold
        self._ref_data: pd.DataFrame | None = None
        self._ref_scores: np.ndarray | None = None
        self._psi_values: dict[str, float] = {}
        self._kl_value: float = 0.0
        self._has_update = False

    def set_reference(self, data: pd.DataFrame, ref_scores: np.ndarray) -> None:
        self._ref_data = data
        self._ref_scores = ref_scores

    def update(self, current_data: pd.DataFrame, current_scores: np.ndarray) -> None:
        """Compute drift metrics against reference data."""
        if self._ref_data is None or self._ref_scores is None:
            return

        for feat in self._feature_names:
            psi = _compute_psi(
                self._ref_data[feat].values,
                current_data[feat].values,
            )
            self._psi_values[feat] = psi
            data_drift_psi.labels(feature=feat).set(psi)

        self._kl_value = _compute_kl(self._ref_scores, current_scores)
        prediction_drift_kl_divergence.set(self._kl_value)
        self._has_update = True

    def get_psi_values(self) -> dict[str, float]:
        return dict(self._psi_values)

    def get_kl_divergence(self) -> float:
        return self._kl_value

    def should_rollback(self) -> bool:
        """Return True if any drift signal exceeds its threshold."""
        if not self._has_update:
            return False

        for psi in self._psi_values.values():
            if psi > self._psi_threshold:
                return True

        if self._kl_value > self._kl_threshold:
            return True

        return False
