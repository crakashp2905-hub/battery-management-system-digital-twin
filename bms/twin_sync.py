"""Online data assimilation — the step from *simulator* to *digital twin*.

A simulator runs open-loop.  A **digital twin** continuously corrects its
internal state against the real device it mirrors.  :class:`TwinSync` runs the
ECM plant forward on the measured current, then nudges the plant's SoC toward
the measured terminal voltage with a Luenberger correction — so the twin tracks
the physical cell rather than drifting away from it.

Crucially it also watches the **model-vs-measurement residual**.  When the twin's
model matches reality the residual is zero-mean noise; when the physical cell
changes in a way the model does not capture (resistance growth from ageing, an
OCV shift from a fault), the residual grows and stays biased.  That rising
residual is a **drift / health signal** a plain forward simulation cannot give —
the twin knowing it is out of sync and needs recalibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .ecm import ECMParameters
from .ocv_soc import OCVSOC


@dataclass
class TwinSync:
    """Assimilate live (current, voltage, temperature) into an ECM twin.

    Parameters
    ----------
    params, ocv_curve : the twin's cell model.
    correction_gain : Luenberger gain mapping a voltage residual to a SoC
        correction (0 = open-loop simulation, larger = tighter tracking).
    drift_alpha : EWMA rate for the residual statistics.
    drift_threshold_V : residual RMS above which the twin is flagged as drifted.
    max_soc_correction : per-step clamp on the SoC nudge (robust to flat OCV).
    """

    params: ECMParameters
    ocv_curve: OCVSOC = field(default_factory=OCVSOC)
    correction_gain: float = 0.1
    drift_alpha: float = 0.02
    drift_threshold_V: float = 0.006
    max_soc_correction: float = 0.02
    # Model error (e.g. resistance growth) shows in the voltage residual only
    # *under load*; the corrector can hide a steady offset by biasing SoC, so the
    # drift statistic is accumulated only when |current| exceeds this.
    drift_min_current_A: float = 0.3

    def __post_init__(self) -> None:
        self.reset(1.0)

    def reset(self, soc0: float = 1.0) -> None:
        self.soc = float(np.clip(soc0, 0.0, 1.0))
        self.vrc1 = 0.0
        self.vrc2 = 0.0
        self._resid_ewma = 0.0
        self._resid_sq_ewma = 0.0
        self._n = 0

    def predicted_voltage(self, current: float, temperature_C: float) -> float:
        p = self.params.at_temperature(temperature_C)
        ocv = float(self.ocv_curve.ocv(self.soc, current, T_C=temperature_C))
        return ocv - self.vrc1 - self.vrc2 - p.R0 * current

    def assimilate(self, current: float, voltage_meas: float, dt: float,
                   temperature_C: float = 25.0) -> dict:
        """One assimilation step; ``current > 0`` = discharge.  Returns the
        corrected SoC, the predicted voltage, the residual, its EWMA RMS, and a
        ``drift`` flag."""
        p = self.params.at_temperature(temperature_C)
        # ---- Predict the plant forward on the measured current -------
        a1 = math.exp(-dt / max(p.tau1, 1e-9))
        a2 = math.exp(-dt / max(p.tau2, 1e-9))
        self.vrc1 = a1 * self.vrc1 + (1 - a1) * p.R1 * current
        self.vrc2 = a2 * self.vrc2 + (1 - a2) * p.R2 * current
        self.soc = float(np.clip(self.soc - current * dt / (p.Q_nom_Ah * 3600.0),
                                 0.0, 1.0))
        # ---- Residual vs the real measurement ------------------------
        v_pred = self.predicted_voltage(current, temperature_C)
        residual = float(voltage_meas) - v_pred
        # ---- Luenberger correction of the plant SoC ------------------
        docv = float(self.ocv_curve.docv_dsoc(self.soc))
        correction = self.correction_gain * residual / (docv if abs(docv) > 0.05 else 0.05)
        correction = float(np.clip(correction, -self.max_soc_correction, self.max_soc_correction))
        self.soc = float(np.clip(self.soc + correction, 0.0, 1.0))
        # ---- Drift / health statistics (EWMA of the under-load residual) ----
        if abs(current) >= self.drift_min_current_A:
            a = self.drift_alpha
            self._resid_ewma = (1 - a) * self._resid_ewma + a * residual
            self._resid_sq_ewma = (1 - a) * self._resid_sq_ewma + a * residual ** 2
            self._n += 1
        rms = math.sqrt(max(self._resid_sq_ewma, 0.0))
        return {
            "soc": self.soc,
            "predicted_voltage_V": v_pred,
            "residual_V": residual,
            "residual_rms_V": rms,
            "residual_bias_V": self._resid_ewma,
            "drift": bool(self._n > 20 and rms > self.drift_threshold_V),
        }

    def run(self, currents: np.ndarray, voltages: np.ndarray, dt: float,
            temperatures: np.ndarray | None = None) -> dict:
        """Assimilate a full trace; returns SoC and residual-RMS trajectories and
        whether the twin drifted by the end."""
        currents = np.asarray(currents, float)
        voltages = np.asarray(voltages, float)
        n = len(currents)
        T = np.full(n, 25.0) if temperatures is None else np.asarray(temperatures, float)
        soc, rms, resid = np.empty(n), np.empty(n), np.empty(n)
        drift = False
        for k in range(n):
            out = self.assimilate(float(currents[k]), float(voltages[k]), dt,
                                  temperature_C=float(T[k]))
            soc[k] = out["soc"]
            rms[k] = out["residual_rms_V"]
            resid[k] = out["residual_V"]
            drift = drift or out["drift"]
        return {"soc": soc, "residual_rms_V": rms, "residual_V": resid,
                "drift": drift, "final_residual_rms_V": float(rms[-1])}
