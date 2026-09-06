"""
Online state-of-health via a joint Extended Kalman Filter.

A plain SoC EKF assumes a fixed capacity ``Q``.  As a cell ages ``Q`` shrinks,
and a fixed-``Q`` filter silently mis-attributes the discrepancy to SoC.  This
**joint EKF** augments the state with capacity,

    x = [SoC, V_RC1, V_RC2, Q]

and tracks ``Q`` online as a slow random-walk parameter.  Capacity becomes
observable whenever SoC moves appreciably (a good charge/discharge), because the
*rate* of SoC change depends on ``Q``.  The result is a live
``soh = Q / Q_nominal`` alongside SoC, each with its own 1-σ uncertainty.

It implements the :class:`bms.estimation.RecursiveSocEstimator` interface, so it
drops straight into the estimator registry as ``"joint_ekf"`` and pairs with the
offline :class:`bms.aging.AgingModel` (which predicts fade; this estimates it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ecm import ECMParameters
from .ocv_soc import OCVSOC


@dataclass
class JointEKFSoH:
    """Joint SoC + capacity EKF.

    Parameters
    ----------
    params : ECMParameters
        Cell ECM (its ``Q_nom_Ah`` sets the default nominal capacity).
    ocv_curve : OCVSOC
        OCV–SOC characteristic (and its Jacobian) used by the measurement model.
    q_nominal_Ah : float, optional
        Beginning-of-life capacity used to normalise SoH.  Defaults to
        ``params.Q_nom_Ah``.
    capacity_rw_std_Ah : float
        Per-step random-walk std of capacity — how fast ``Q`` is allowed to
        drift.  Small (aging is slow); larger tracks faster but noisier.
    """

    params: ECMParameters
    ocv_curve: OCVSOC = field(default_factory=OCVSOC)
    q_nominal_Ah: float | None = None
    capacity_rw_std_Ah: float = 1e-4
    # Initial capacity uncertainty (fraction of nominal).  Aged capacity is
    # genuinely uncertain; a wider prior lets the filter converge to the true
    # capacity within a cycle instead of crawling.
    initial_capacity_std_frac: float = 0.20
    R_cov: float = 1e-4
    name: str = "joint_ekf"

    def __post_init__(self) -> None:
        self.q_nominal_Ah = float(self.q_nominal_Ah
                                  if self.q_nominal_Ah is not None
                                  else self.params.Q_nom_Ah)
        self.reset(1.0)
        self.Q_cov = np.diag([1e-7, 1e-6, 1e-6, self.capacity_rw_std_Ah ** 2])

    # ------------------------------------------------------------------
    def reset(self, soc0: float = 1.0, q0: float | None = None) -> None:
        q0 = float(q0 if q0 is not None else self.q_nominal_Ah)
        self.x = np.array([float(np.clip(soc0, 0.0, 1.0)), 0.0, 0.0, q0])
        cap_var = (self.initial_capacity_std_frac * self.q_nominal_Ah) ** 2
        self.P = np.diag([1e-2, 1e-3, 1e-3, cap_var])

    # ---- read-outs -----------------------------------------------------
    @property
    def soc(self) -> float:
        return float(np.clip(self.x[0], 0.0, 1.0))

    @property
    def capacity_Ah(self) -> float:
        return float(self.x[3])

    @property
    def soh(self) -> float:
        """State-of-health as capacity retention Q / Q_nominal."""
        return float(self.x[3] / max(self.q_nominal_Ah, 1e-9))

    @property
    def soc_uncertainty_1sigma(self) -> float:
        return float(np.sqrt(max(self.P[0, 0], 0.0)))

    @property
    def soh_uncertainty_1sigma(self) -> float:
        return float(np.sqrt(max(self.P[3, 3], 0.0)) / max(self.q_nominal_Ah, 1e-9))

    # ------------------------------------------------------------------
    def update(self, current: float, voltage: float, dt: float,
               temperature_C: float = 25.0) -> float:
        """One joint predict/update step (``current > 0`` = discharge)."""
        p = self.params.at_temperature(temperature_C)
        a1 = float(np.exp(-dt / max(p.tau1, 1e-9)))
        a2 = float(np.exp(-dt / max(p.tau2, 1e-9)))
        soc, vrc1, vrc2, q = self.x
        q = max(q, 1e-3)

        # ---- Predict ------------------------------------------------
        x_pred = np.array([
            soc - current * dt / (q * 3600.0),
            a1 * vrc1 + (1.0 - a1) * p.R1 * current,
            a2 * vrc2 + (1.0 - a2) * p.R2 * current,
            q,                                    # capacity: random walk
        ])
        dsoc_dq = current * dt / (q ** 2 * 3600.0)
        F = np.array([
            [1.0, 0.0, 0.0, dsoc_dq],
            [0.0, a1, 0.0, 0.0],
            [0.0, 0.0, a2, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        P_pred = F @ self.P @ F.T + self.Q_cov

        # ---- Update -------------------------------------------------
        ocv = float(self.ocv_curve.ocv(x_pred[0], current, T_C=temperature_C))
        h = ocv - x_pred[1] - x_pred[2] - p.R0 * current
        docv = float(self.ocv_curve.docv_dsoc(x_pred[0]))
        H = np.array([[docv, -1.0, -1.0, 0.0]])
        innovation = voltage - h
        S = float((H @ P_pred @ H.T).item() + self.R_cov)
        K = (P_pred @ H.T / S).flatten()
        self.x = x_pred + K * innovation
        self.P = (np.eye(4) - np.outer(K, H)) @ P_pred
        self.x[3] = max(self.x[3], 1e-3)          # capacity stays positive
        return self.soc

    def run(self, currents: np.ndarray, voltages: np.ndarray, dt: float,
            temperatures: np.ndarray | None = None) -> np.ndarray:
        """Run over a full trace; returns the SoC series."""
        n = len(currents)
        T = np.full(n, 25.0) if temperatures is None else np.asarray(temperatures, float)
        out = np.empty(n)
        for k in range(n):
            out[k] = self.update(float(currents[k]), float(voltages[k]), dt,
                                 temperature_C=float(T[k]))
        return out
