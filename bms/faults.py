"""
Fault simulation, injection, and hybrid (rule + ML) detection.

Failure modes
-------------
* OVERCHARGE       — cell driven above 4.25 V, e.g. by a runaway charger.
* SHORT_CIRCUIT    — internal short collapses voltage and dumps current.
* THERMAL_RUNAWAY  — exothermic side reactions trigger when T exceeds a
                     threshold; once entered, T grows super-linearly.
* SENSOR_DROPOUT   — voltage or temperature sensor outputs frozen/zero.
* SENSOR_BIAS      — slow voltage sensor drift.

Each fault has an `apply(state)` method that mutates the simulation state
in place (or annotates measurements) so it can be injected at any timestep.

The hybrid detector combines:
  • Hand-tuned rules over instantaneous and short-window features.
  • A RandomForestClassifier trained on labelled simulation data.
The two are fused with a logical OR, so the rules act as a safety net
covering the labels the ML model was never shown.

Rolling-window features
-----------------------
`RollingFeatureBuffer` maintains a sliding window of raw cell voltages and
temperatures.  Its `stats_vector()` method returns four rolling statistics
(max voltage std, max temperature std, max voltage drift, max temperature drift)
that are appended to the 10-element base feature vector before passing to the
RandomForest.  This gives the ML model visibility into slow-onset anomalies
(e.g. sensor bias drift) that are invisible in instantaneous features alone.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# ----------------------------------------------------------------------
# Fault catalogue
# ----------------------------------------------------------------------
class FaultMode(str, Enum):
    NONE = "none"
    OVERCHARGE = "overcharge"
    SHORT_CIRCUIT = "short_circuit"
    THERMAL_RUNAWAY = "thermal_runaway"
    SENSOR_DROPOUT = "sensor_dropout"
    SENSOR_BIAS = "sensor_bias"


@dataclass
class FaultSpec:
    mode: FaultMode
    start_step: int = 0
    end_step: int = 10**9
    cell_index: int = 0
    severity: float = 1.0

    def is_active(self, k: int) -> bool:
        return self.start_step <= k < self.end_step


class FaultInjector:
    """Applies a list of `FaultSpec`s to ECM state and measurements."""

    def __init__(self, specs: list[FaultSpec] | None = None,
                 chemistry: str = "nmc"):
        self.specs = specs or []
        from .chemistry import get_chemistry_props
        props = get_chemistry_props(chemistry)
        self.OVERCHARGE_V: float = props["v_overcharge"]
        self.THERMAL_RUNAWAY_T: float = props["T_runaway_C"]

    def add(self, spec: FaultSpec) -> None:
        self.specs.append(spec)

    # ------------------------------------------------------------------
    def apply_to_currents(self, currents: np.ndarray, k: int) -> np.ndarray:
        """Inject load-side faults into the per-cell current vector."""
        out = currents.copy()
        for s in self.specs:
            if not s.is_active(k):
                continue
            if s.mode == FaultMode.SHORT_CIRCUIT:
                out[s.cell_index] += 30.0 * s.severity
            elif s.mode == FaultMode.OVERCHARGE:
                out[s.cell_index] -= 5.0 * s.severity
        return out

    def apply_to_temperatures(self, T: np.ndarray, k: int, dt: float) -> np.ndarray:
        out = T.copy()
        for s in self.specs:
            if not s.is_active(k):
                continue
            if s.mode == FaultMode.THERMAL_RUNAWAY and out[s.cell_index] > self.THERMAL_RUNAWAY_T:
                out[s.cell_index] += 0.5 * s.severity * (out[s.cell_index] - self.THERMAL_RUNAWAY_T) * dt
        return out

    def apply_to_voltage_meas(self, v_meas: np.ndarray, k: int) -> np.ndarray:
        out = v_meas.copy()
        for s in self.specs:
            if not s.is_active(k):
                continue
            if s.mode == FaultMode.SENSOR_DROPOUT:
                out[s.cell_index] = 0.0
            elif s.mode == FaultMode.SENSOR_BIAS:
                tau = max(1, k - s.start_step + 1)
                out[s.cell_index] += 0.0005 * tau * s.severity
        return out

    def label(self, k: int) -> FaultMode:
        """Return the dominant active fault label at step k (or NONE)."""
        for s in self.specs:
            if s.is_active(k):
                return s.mode
        return FaultMode.NONE


# ----------------------------------------------------------------------
# Feature extraction (used by both rule and ML detectors)
# ----------------------------------------------------------------------
def extract_features(v_cells: np.ndarray, currents: np.ndarray,
                     T_cells: np.ndarray, dv_dt: np.ndarray,
                     dT_dt: np.ndarray) -> np.ndarray:
    """10-element per-step base feature vector for the detector."""
    return np.array([
        v_cells.max(), v_cells.min(), v_cells.max() - v_cells.min(),
        currents.max(), np.abs(currents).max(),
        T_cells.max(), T_cells.max() - T_cells.min(),
        dv_dt.max(), np.abs(dv_dt).max(),
        dT_dt.max(),
    ])


# ----------------------------------------------------------------------
# Rolling-window feature buffer
# ----------------------------------------------------------------------
class RollingFeatureBuffer:
    """Sliding window of raw observations for multi-step anomaly features.

    Maintains the last `window` steps of per-cell voltages and temperatures.
    ``stats_vector()`` returns 4 scalar statistics that capture slow-onset
    drift invisible in instantaneous features:

        [max_cell_V_std, max_cell_T_std, max_V_drift, max_T_drift]

    These are appended to the 10-element base features, giving the ML
    classifier a 14-element input vector.

    Parameters
    ----------
    window : int
        Number of past steps to retain (default 30 ≈ 30 s at dt=1 s).
    """

    N_EXTRA: int = 4  # number of features added by stats_vector()

    def __init__(self, window: int = 30):
        self.window = window
        self._v_buf: deque = deque(maxlen=window)
        self._T_buf: deque = deque(maxlen=window)

    def push(self, v_cells: np.ndarray, T_cells: np.ndarray) -> None:
        """Append the latest observation to the sliding window."""
        self._v_buf.append(v_cells.copy())
        self._T_buf.append(T_cells.copy())

    def stats_vector(self) -> np.ndarray:
        """Return 4 rolling statistics; zeros when fewer than 2 samples."""
        n = len(self._v_buf)
        if n < 2:
            return np.zeros(self.N_EXTRA)
        v_arr = np.array(self._v_buf)   # (n, n_cells)
        T_arr = np.array(self._T_buf)
        return np.array([
            v_arr.std(axis=0).max(),                # max cell-voltage std
            T_arr.std(axis=0).max(),                # max cell-temperature std
            (v_arr[-1] - v_arr[0]).max(),           # max voltage drift over window
            (T_arr[-1] - T_arr[0]).max(),           # max temperature drift over window
        ])

    def reset(self) -> None:
        self._v_buf.clear()
        self._T_buf.clear()

    @property
    def full_feature_size(self) -> int:
        """Total feature length: 10 base + 4 rolling = 14."""
        return 10 + self.N_EXTRA


# ----------------------------------------------------------------------
# Hybrid detector
# ----------------------------------------------------------------------
class HybridFaultDetector:
    """Rule-based + RandomForest fault detector with a unified interface.

    Workflow
    --------
    1. ``fit(X, y)`` trains the RandomForest on labelled feature rows.
       Each row should be the 14-element vector produced by concatenating
       ``extract_features(...)`` with ``buffer.stats_vector()``.
    2. ``predict_step(features, v_cells, T_cells)`` returns
       (predicted_mode, source) where ``source`` ∈ {"rule", "ml", "none"}.
       The detector maintains an internal `RollingFeatureBuffer`; only the
       rule check uses the raw v_cells/T_cells directly.

    Parameters
    ----------
    chemistry : str
        ``"nmc"`` (default) or ``"lfp"``.  Sets chemistry-specific rule
        thresholds (overcharge voltage and thermal-runaway temperature).
    """

    def __init__(self, n_trees: int = 80, random_state: int = 0,
                 ml_min_confidence: float = 0.65, buffer_window: int = 30,
                 chemistry: str = "nmc"):
        self.clf = RandomForestClassifier(n_estimators=n_trees,
                                          random_state=random_state,
                                          class_weight="balanced", n_jobs=1)
        self.fitted = False
        self.ml_min_confidence = float(ml_min_confidence)
        self._buffer = RollingFeatureBuffer(window=buffer_window)

        from .chemistry import get_chemistry_props
        props = get_chemistry_props(chemistry)
        self._v_overcharge: float = props["v_overcharge"]
        self._v_dropout: float = props["v_dropout"]
        self._T_runaway_C: float = props["T_runaway_C"]

    # ---- Rules ------------------------------------------------------
    def rule_check(self, v_cells: np.ndarray, T_cells: np.ndarray) -> FaultMode:
        if (v_cells >= self._v_overcharge).any():
            return FaultMode.OVERCHARGE
        if (v_cells <= self._v_dropout).any():
            return FaultMode.SENSOR_DROPOUT
        if (T_cells >= self._T_runaway_C).any():
            return FaultMode.THERMAL_RUNAWAY
        return FaultMode.NONE

    # ---- ML ---------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.clf.fit(X, y)
        self.fitted = True

    def reset_buffer(self) -> None:
        """Clear the rolling feature buffer (call between scenarios)."""
        self._buffer.reset()

    # ---- Combined ---------------------------------------------------
    def predict_step(self, features: np.ndarray, v_cells: np.ndarray,
                     T_cells: np.ndarray) -> tuple[str, str]:
        """Evaluate one step: update buffer, run rule check, then ML.

        Parameters
        ----------
        features : (10,) array
            Base feature vector from ``extract_features``.
        v_cells : (n,) array   Per-cell terminal voltages [V].
        T_cells : (n,) array   Per-cell temperatures [°C].

        Returns
        -------
        (label, source) : str, str
            ``source`` ∈ {"rule", "ml", "none"}.
        """
        # Update rolling buffer with latest observations.
        self._buffer.push(v_cells, T_cells)

        # Rule layer always takes precedence.
        rule = self.rule_check(v_cells, T_cells)
        if rule != FaultMode.NONE:
            return rule.value, "rule"

        if self.fitted:
            # Build 14-element enhanced feature vector.
            ml_feats = np.concatenate([features, self._buffer.stats_vector()])
            proba = self.clf.predict_proba(ml_feats.reshape(1, -1))[0]
            classes = list(self.clf.classes_)
            top_idx = int(np.argmax(proba))
            label = classes[top_idx]
            confidence = float(proba[top_idx])
            if label != FaultMode.NONE.value and confidence >= self.ml_min_confidence:
                return label, "ml"
        return FaultMode.NONE.value, "none"

    # ---- Convenience: confusion-style report ------------------------
    @staticmethod
    def report(y_true: list[str], y_pred: list[str]) -> pd.DataFrame:
        labels = sorted(set(y_true) | set(y_pred))
        cm = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
        for t, p in zip(y_true, y_pred):
            cm.loc[t, p] += 1
        cm.index.name = "true"
        cm.columns.name = "pred"
        return cm
