"""
SOC estimation algorithms and a benchmarking harness.

Algorithms
----------
* CoulombCounter  – open-loop integrator, baseline.
* EKFEstimator    – Extended Kalman filter with the 2-RC ECM as the
                    process model and OCV(SOC) as the measurement function.
* UKFEstimator    – Unscented Kalman filter via filterpy.
* LSTMEstimator   – Single-layer LSTM implemented in pure NumPy and trained
                    with Adam + BPTT on synthetic ECM trajectories.

`benchmark_estimators` runs every estimator on the same noisy current/voltage
trace and returns RMSE, MAE, max error and runtime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

from .ecm import ECMParameters
from .ocv_soc import OCVSOC


# ======================================================================
# 1.  Coulomb counter — open-loop integrator
# ======================================================================
class CoulombCounter:
    """Pure ampere-second integrator. Drifts unbounded with sensor bias."""

    name = "coulomb_counting"

    def __init__(self, capacity_Ah: float, soc0: float = 1.0):
        self.capacity_C = capacity_Ah * 3600.0
        self._soc0 = float(np.clip(soc0, 0.0, 1.0))
        self.soc = self._soc0

    def reset(self, soc0: float | None = None) -> None:
        if soc0 is not None:
            self._soc0 = float(np.clip(soc0, 0.0, 1.0))
        self.soc = self._soc0

    def update(self, current: float, voltage: float, dt: float) -> float:
        # voltage unused — present in signature for interface uniformity
        self.soc = float(np.clip(self.soc - current * dt / self.capacity_C, 0.0, 1.0))
        return self.soc

    def run(self, currents: np.ndarray, voltages: np.ndarray, dt: float) -> np.ndarray:
        out = np.empty(len(currents))
        for k in range(len(currents)):
            out[k] = self.update(currents[k], voltages[k], dt)
        return out


# ======================================================================
# 2.  Extended Kalman filter
# ======================================================================
@dataclass
class EKFEstimator:
    """EKF over state x = [SOC, V_RC1, V_RC2].

    Process model (discretised second-order ECM):
        SOC      ← SOC - i·dt/Q
        V_RC{1,2}← a_{1,2}·V_RC{1,2} + (1-a_{1,2})·R_{1,2}·i
    Measurement:
        V_t      = OCV(SOC) - V_RC1 - V_RC2 - R0·i + v
    """

    params: ECMParameters
    ocv_curve: OCVSOC = field(default_factory=OCVSOC)
    Q_cov: np.ndarray | None = None      # process noise
    R_cov: float = 1e-4                  # voltage measurement variance (V²)
    x: np.ndarray = field(init=False)
    P: np.ndarray = field(init=False)
    name: str = "ekf"

    def __post_init__(self):
        self.x = np.array([1.0, 0.0, 0.0])         # initial state
        self.P = np.diag([1e-2, 1e-3, 1e-3])       # initial covariance
        if self.Q_cov is None:
            self.Q_cov = np.diag([1e-7, 1e-6, 1e-6])

    # ------------------------------------------------------------------
    def reset(self, soc0: float = 1.0):
        self.x = np.array([float(np.clip(soc0, 0.0, 1.0)), 0.0, 0.0])
        self.P = np.diag([1e-2, 1e-3, 1e-3])

    @property
    def soc(self) -> float:
        return float(self.x[0])

    @property
    def soc_uncertainty_1sigma(self) -> float:
        """1-σ SOC uncertainty from EKF covariance [SOC fraction]."""
        return float(np.sqrt(max(self.P[0, 0], 0.0)))

    # ------------------------------------------------------------------
    def update(self, current: float, voltage: float, dt: float,
               temperature_C: float = 25.0) -> float:
        """One EKF predict-update step.

        Parameters
        ----------
        temperature_C : float
            Cell temperature [°C].  Used to Arrhenius-scale ECM parameters
            and apply OCV temperature correction, matching the physical model.
        """
        p = self.params.at_temperature(temperature_C)
        a1 = float(np.exp(-dt / max(p.tau1, 1e-9)))
        a2 = float(np.exp(-dt / max(p.tau2, 1e-9)))

        # ---- Predict ------------------------------------------------
        F = np.array([[1.0, 0.0, 0.0],
                      [0.0, a1,  0.0],
                      [0.0, 0.0, a2]])
        b = np.array([-dt / (p.Q_nom_Ah * 3600.0),
                      (1 - a1) * p.R1,
                      (1 - a2) * p.R2])
        self.x = F @ self.x + b * current
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
        self.P = F @ self.P @ F.T + self.Q_cov

        # ---- Update -------------------------------------------------
        ocv = float(self.ocv_curve.ocv(self.x[0], T_C=temperature_C))
        h = ocv - self.x[1] - self.x[2] - p.R0 * current
        H = np.array([[float(self.ocv_curve.docv_dsoc(self.x[0])), -1.0, -1.0]])
        y_innov = voltage - h
        S = float((H @ self.P @ H.T).item() + self.R_cov)
        K = (self.P @ H.T / S).flatten()
        self.x = self.x + K * y_innov
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
        self.P = (np.eye(3) - np.outer(K, H)) @ self.P
        return self.soc

    def run(self, currents: np.ndarray, voltages: np.ndarray, dt: float,
            temperatures: np.ndarray | None = None) -> np.ndarray:
        """Run the EKF over a full trace.

        Parameters
        ----------
        temperatures : (N,) array, optional
            Per-step cell temperature [°C].  Defaults to 25 °C.
        """
        T_arr = np.full(len(currents), 25.0) if temperatures is None else np.asarray(temperatures, float)
        out = np.empty(len(currents))
        for k in range(len(currents)):
            out[k] = self.update(currents[k], voltages[k], dt, temperature_C=float(T_arr[k]))
        return out


# ======================================================================
# 3.  Unscented Kalman filter (filterpy)
# ======================================================================
class UKFEstimator:
    """UKF wrapper around filterpy.kalman.UnscentedKalmanFilter."""

    name = "ukf"

    def __init__(self, params: ECMParameters, ocv_curve: OCVSOC | None = None,
                 Q_cov: Sequence[float] | None = None, R_cov: float = 1e-4,
                 dt: float = 1.0):
        self.params = params
        self.ocv = ocv_curve or OCVSOC()
        self.dt = dt
        self.current_input = 0.0
        self._temperature_C = 25.0
        sigma = MerweScaledSigmaPoints(n=3, alpha=1e-3, beta=2.0, kappa=0.0)
        self.filter = UnscentedKalmanFilter(
            dim_x=3, dim_z=1, dt=dt, fx=self._fx, hx=self._hx, points=sigma,
        )
        self.filter.x = np.array([1.0, 0.0, 0.0])
        self.filter.P = np.diag([1e-2, 1e-3, 1e-3])
        self.filter.Q = np.diag(Q_cov if Q_cov is not None else [1e-7, 1e-6, 1e-6])
        self.filter.R = np.array([[float(R_cov)]])

    def _fx(self, x: np.ndarray, dt: float) -> np.ndarray:
        p = self.params.at_temperature(self._temperature_C)
        a1 = np.exp(-dt / max(p.tau1, 1e-9))
        a2 = np.exp(-dt / max(p.tau2, 1e-9))
        i = self.current_input
        return np.array([
            np.clip(x[0] - i * dt / (p.Q_nom_Ah * 3600.0), 0.0, 1.0),
            a1 * x[1] + (1 - a1) * p.R1 * i,
            a2 * x[2] + (1 - a2) * p.R2 * i,
        ])

    def _hx(self, x: np.ndarray) -> np.ndarray:
        p = self.params.at_temperature(self._temperature_C)
        ocv_val = float(self.ocv.ocv(np.clip(x[0], 0.0, 1.0), T_C=self._temperature_C))
        return np.array([ocv_val - x[1] - x[2] - p.R0 * self.current_input])

    def reset(self, soc0: float = 1.0):
        self.filter.x = np.array([float(np.clip(soc0, 0.0, 1.0)), 0.0, 0.0])
        self.filter.P = np.diag([1e-2, 1e-3, 1e-3])

    @property
    def soc(self) -> float:
        return float(np.clip(self.filter.x[0], 0.0, 1.0))

    @property
    def soc_uncertainty_1sigma(self) -> float:
        """1-σ SOC uncertainty from UKF covariance [SOC fraction]."""
        return float(np.sqrt(max(self.filter.P[0, 0], 0.0)))

    def update(self, current: float, voltage: float, dt: float,
               temperature_C: float = 25.0) -> float:
        """One UKF predict-update step.

        Parameters
        ----------
        temperature_C : float
            Cell temperature [°C].  Passed to the process and measurement
            functions for Arrhenius-scaled ECM parameters and OCV correction.
        """
        self.current_input = float(current)
        self._temperature_C = float(temperature_C)
        self.filter.predict(dt=dt)
        self.filter.update(np.array([float(voltage)]))
        return self.soc

    def run(self, currents: np.ndarray, voltages: np.ndarray, dt: float,
            temperatures: np.ndarray | None = None) -> np.ndarray:
        """Run the UKF over a full trace.

        Parameters
        ----------
        temperatures : (N,) array, optional
            Per-step cell temperature [°C].  Defaults to 25 °C.
        """
        T_arr = np.full(len(currents), 25.0) if temperatures is None else np.asarray(temperatures, float)
        out = np.empty(len(currents))
        for k in range(len(currents)):
            out[k] = self.update(currents[k], voltages[k], dt, temperature_C=float(T_arr[k]))
        return out


# ======================================================================
# 4.  Pure-NumPy LSTM SOC estimator
# ======================================================================
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -50, 50))


class LSTMEstimator:
    """Single-layer LSTM with Adam + BPTT, written in NumPy.

    Inputs (per timestep):  [voltage, current, dt-normalised SOC anchor]
    Output:                 SOC ∈ [0, 1] via sigmoid head.
    """

    name = "lstm"

    def __init__(self, input_size: int = 3, hidden_size: int = 16, seed: int = 0):
        self.D, self.H = input_size, hidden_size
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(hidden_size)

        # Concatenated weight: [W_x | W_h] for each gate (i, f, o, g).
        self.Wf = rng.uniform(-s, s, (input_size + hidden_size, hidden_size))
        self.Wi = rng.uniform(-s, s, (input_size + hidden_size, hidden_size))
        self.Wo = rng.uniform(-s, s, (input_size + hidden_size, hidden_size))
        self.Wg = rng.uniform(-s, s, (input_size + hidden_size, hidden_size))
        self.bf = np.ones(hidden_size)        # bias forget gate to 1 → remember by default
        self.bi = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
        self.bg = np.zeros(hidden_size)
        # Output head SOC = σ(h · Wy + by)
        self.Wy = rng.uniform(-s, s, (hidden_size, 1))
        self.by = np.zeros(1)

        # Adam moments
        self._m: dict = {}
        self._v: dict = {}
        self._t = 0
        self._normalisers: dict | None = None

    # ------------------------------------------------------------------
    def _normalise(self, X: np.ndarray) -> np.ndarray:
        n = self._normalisers
        if n is None:
            mu = X.reshape(-1, X.shape[-1]).mean(axis=0)
            sd = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
            self._normalisers = {"mu": mu, "sd": sd}
            n = self._normalisers
        return (X - n["mu"]) / n["sd"]

    # ------------------------------------------------------------------
    def _forward_seq(self, x_seq: np.ndarray) -> tuple[np.ndarray, list]:
        """One sequence forward.  x_seq: (T, D)  →  y_seq: (T,) plus cache."""
        T = x_seq.shape[0]
        h = np.zeros(self.H)
        c = np.zeros(self.H)
        ys = np.empty(T)
        cache = []
        for t in range(T):
            z = np.concatenate([x_seq[t], h])
            f = _sigmoid(z @ self.Wf + self.bf)
            i = _sigmoid(z @ self.Wi + self.bi)
            o = _sigmoid(z @ self.Wo + self.bo)
            g = _tanh(z @ self.Wg + self.bg)
            c = f * c + i * g
            h = o * _tanh(c)
            y_pre = h @ self.Wy + self.by
            y = _sigmoid(y_pre)
            ys[t] = float(y[0])
            cache.append((z, f, i, o, g, c.copy(), h.copy(), float(y_pre[0]), float(y[0])))
        return ys, cache

    # ------------------------------------------------------------------
    def _backward_seq(self, x_seq: np.ndarray, y_true: np.ndarray, cache: list, ys: np.ndarray) -> dict:
        T = x_seq.shape[0]
        grads = {k: np.zeros_like(getattr(self, k)) for k in
                 ("Wf", "Wi", "Wo", "Wg", "bf", "bi", "bo", "bg", "Wy", "by")}

        dh_next = np.zeros(self.H)
        dc_next = np.zeros(self.H)

        for t in reversed(range(T)):
            z, f, i_, o, g, c_t, h_t, y_pre, y_t = cache[t]
            c_prev = cache[t - 1][5] if t > 0 else np.zeros(self.H)

            # Loss gradient (MSE on sigmoid output)
            dy = 2.0 * (y_t - y_true[t]) / T              # dL/dy_t
            dy_pre = dy * y_t * (1 - y_t)                 # through sigmoid
            grads["Wy"] += np.outer(h_t, dy_pre)
            grads["by"] += np.array([dy_pre])
            dh = self.Wy.flatten() * dy_pre + dh_next

            do = dh * _tanh(c_t)
            dc = dh * o * (1 - _tanh(c_t) ** 2) + dc_next
            df = dc * c_prev
            di = dc * g
            dg = dc * i_
            dc_prev = dc * f

            # Through gate non-linearities
            df_z = df * f * (1 - f)
            di_z = di * i_ * (1 - i_)
            do_z = do * o * (1 - o)
            dg_z = dg * (1 - g ** 2)

            grads["Wf"] += np.outer(z, df_z)
            grads["Wi"] += np.outer(z, di_z)
            grads["Wo"] += np.outer(z, do_z)
            grads["Wg"] += np.outer(z, dg_z)
            grads["bf"] += df_z
            grads["bi"] += di_z
            grads["bo"] += do_z
            grads["bg"] += dg_z

            dz = (df_z @ self.Wf.T + di_z @ self.Wi.T +
                  do_z @ self.Wo.T + dg_z @ self.Wg.T)
            dh_next = dz[self.D:]            # gradient w.r.t. previous h
            dc_next = dc_prev

        # Clip exploding gradients
        for k, g in grads.items():
            np.clip(g, -1.0, 1.0, out=g)
        return grads

    # ------------------------------------------------------------------
    def _adam_step(self, grads: dict, lr: float = 1e-2,
                   b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self._t += 1
        for k, g in grads.items():
            if k not in self._m:
                self._m[k] = np.zeros_like(g)
                self._v[k] = np.zeros_like(g)
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * (g * g)
            m_hat = self._m[k] / (1 - b1 ** self._t)
            v_hat = self._v[k] / (1 - b2 ** self._t)
            param = getattr(self, k)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, k, param)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray,
            epochs: int = 30, lr: float = 1e-2, verbose: bool = False) -> list[float]:
        """X: (n_seq, T, D). Y: (n_seq, T)."""
        X = self._normalise(np.asarray(X, float))
        Y = np.asarray(Y, float)
        n = X.shape[0]
        history = []
        for ep in range(epochs):
            order = np.random.permutation(n)
            losses = []
            for k in order:
                ys, cache = self._forward_seq(X[k])
                losses.append(float(np.mean((ys - Y[k]) ** 2)))
                grads = self._backward_seq(X[k], Y[k], cache, ys)
                self._adam_step(grads, lr=lr)
            history.append(float(np.mean(losses)))
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"  epoch {ep:3d}  loss={history[-1]:.5f}")
        return history

    def reset(self, *_, **__):
        self._h = np.zeros(self.H)
        self._c = np.zeros(self.H)

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        if self._normalisers is None:
            raise RuntimeError("LSTM has not been fitted")
        x = (np.asarray(x_seq, float) - self._normalisers["mu"]) / self._normalisers["sd"]
        ys, _ = self._forward_seq(x)
        return ys

    def run(self, currents: np.ndarray, voltages: np.ndarray, dt: float,
            soc0: float = 1.0, temperatures: np.ndarray | None = None) -> np.ndarray:
        """Run inference over a trace.

        If the LSTM was initialised with ``input_size=4`` and ``temperatures``
        is provided, temperature is appended as the 4th input feature.
        Otherwise the standard 3-feature vector [V, I, SOC_anchor] is used.
        """
        anchor = np.full(len(currents), soc0)
        if self.D == 4 and temperatures is not None:
            x_seq = np.column_stack([voltages, currents, anchor,
                                     np.asarray(temperatures, float)])
        else:
            x_seq = np.column_stack([voltages, currents, anchor])
        return self.predict(x_seq)


# ======================================================================
# Benchmark harness
# ======================================================================
def benchmark_estimators(
    estimators: dict,
    currents: np.ndarray,
    voltages: np.ndarray,
    soc_truth: np.ndarray,
    dt: float,
    soc0: float = 1.0,
    noise_levels_v: Sequence[float] = (0.0, 0.005, 0.02),
    seed: int = 1,
) -> pd.DataFrame:
    """Run each estimator over the truth trace at multiple noise levels."""
    rng = np.random.default_rng(seed)
    rows = []
    for sigma in noise_levels_v:
        v_meas = voltages + rng.normal(0, sigma, len(voltages)) if sigma > 0 else voltages.copy()
        for name, est in estimators.items():
            t0 = time.perf_counter()
            try:
                est.reset(soc0)
            except TypeError:
                pass
            if hasattr(est, "run") and not isinstance(est, LSTMEstimator):
                soc_hat = est.run(currents, v_meas, dt)
            else:
                soc_hat = est.run(currents, v_meas, dt, soc0=soc0)
            wall = time.perf_counter() - t0
            err = soc_hat - soc_truth
            rows.append({
                "estimator": name,
                "noise_v_sigma": sigma,
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mae": float(np.mean(np.abs(err))),
                "max_err": float(np.max(np.abs(err))),
                "runtime_s": wall,
            })
    return pd.DataFrame(rows)
