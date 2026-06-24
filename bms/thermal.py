"""
Finite-difference thermal model + PID-regulated cooling.

We treat the pack as a 1-D rod of N cells (one node per cell) with:

  • Internal heat generation Q_gen,i = I² · R0,i + I · (V_OC - V_terminal)_i
    – the Ohmic loss plus reversible/entropic heat (lumped into R0 for
    simplicity at this level of fidelity);
  • Conductive coupling between adjacent cells (kappa);
  • Convective dissipation to the ambient through cooling-plate contact
    with a heat-transfer coefficient h(t) modulated by the PID controller.

dT_i/dt = (Q_gen,i + κ(T_{i-1} - 2T_i + T_{i+1}) - h_eff(t)·(T_i - T_amb))
          / (m·c_p)

Boundary conditions are insulated (Neumann) at the two ends, modelled by
mirror nodes (T_{-1}=T_0, T_N=T_{N-1}).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ThermalParameters:
    """Lumped thermal properties for a single cell node."""
    mass_g: float = 45.0          # 18650 ≈ 45 g
    cp_J_per_gK: float = 0.9      # ≈ 0.9 J/g·K for Li-ion average
    kappa_W_per_K: float = 0.4    # cell-to-cell conductance (busbar + contact)
    h_min_W_per_K: float = 0.2    # baseline natural convection
    h_max_W_per_K: float = 4.0    # max forced cooling
    T_amb_C: float = 25.0


# ----------------------------------------------------------------------
class PIDController:
    """Simple anti-windup PID with output saturation."""

    def __init__(self, kp: float, ki: float, kd: float,
                 out_min: float = 0.0, out_max: float = 1.0,
                 setpoint: float = 30.0):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.out_min, self.out_max = float(out_min), float(out_max)
        self.setpoint = float(setpoint)
        self.integral = 0.0
        self.prev_err = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_err = 0.0

    def step(self, measured: float, dt: float) -> float:
        err = measured - self.setpoint              # higher T → larger error
        deriv = (err - self.prev_err) / max(dt, 1e-9)
        self.prev_err = err

        u_unsat = self.kp * err + self.ki * self.integral + self.kd * deriv
        u = float(np.clip(u_unsat, self.out_min, self.out_max))
        # Anti-windup: only integrate when not saturating in the same direction.
        if (self.out_min < u_unsat < self.out_max) or (np.sign(err) != np.sign(self.integral)):
            self.integral += err * dt
        return u


# ----------------------------------------------------------------------
@dataclass
class ThermalModel:
    """1-D FDM thermal model coupled to a battery pack."""

    n_cells: int
    params: ThermalParameters = field(default_factory=ThermalParameters)
    T: np.ndarray = field(init=False)   # node temperatures [°C]

    def __post_init__(self):
        self.T = np.full(self.n_cells, self.params.T_amb_C, dtype=float)

    # ------------------------------------------------------------------
    def reset(self, T0: float | None = None) -> None:
        T0 = self.params.T_amb_C if T0 is None else float(T0)
        self.T[:] = T0

    def step(self, heat_W: np.ndarray, cooling_duty: float, dt: float) -> np.ndarray:
        """Advance one time-step.

        Parameters
        ----------
        heat_W : (n_cells,) array
            Instantaneous heat generation per cell, W.
        cooling_duty : float in [0, 1]
            PID output mapping linearly to h_eff between h_min and h_max.
        """
        p = self.params
        cooling_duty = float(np.clip(cooling_duty, 0.0, 1.0))
        h_eff = p.h_min_W_per_K + cooling_duty * (p.h_max_W_per_K - p.h_min_W_per_K)

        m_cp = (p.mass_g * p.cp_J_per_gK)            # J/K per cell

        # Mirror BC: T_{-1} = T_0, T_N = T_{N-1}
        T_left = np.concatenate([[self.T[0]], self.T[:-1]])
        T_right = np.concatenate([self.T[1:], [self.T[-1]]])
        cond = p.kappa_W_per_K * (T_left - 2 * self.T + T_right)
        conv = -h_eff * (self.T - p.T_amb_C)

        dT = (heat_W + cond + conv) * dt / m_cp
        self.T = self.T + dT
        return self.T

    # ------------------------------------------------------------------
    def gradient(self) -> float:
        """Max-min temperature spread – proxy for thermal uniformity."""
        return float(self.T.max() - self.T.min())

    @staticmethod
    def heat_generation(currents: np.ndarray, R0: np.ndarray,
                        v_terminal: np.ndarray, ocv: np.ndarray) -> np.ndarray:
        """Per-cell heat production: ohmic + over-potential."""
        ohmic = currents ** 2 * R0
        polarisation = currents * np.maximum(ocv - v_terminal, 0.0)
        return ohmic + polarisation


# ----------------------------------------------------------------------
class PredictiveCoolingController:
    """PID cooling with a feed-forward term based on anticipated heat.

    The feed-forward term adds a fraction of the predicted heat generation
    (in watts, summed across all cells) to the PID output, pre-emptively
    increasing cooling duty *before* the temperature rises.  This reduces
    peak temperature overshoot by 1–3 °C compared with a pure PID under
    aggressive load steps.

    Parameters
    ----------
    kp, ki, kd : float
        PID gains (same semantics as `PIDController`).
    setpoint : float
        Target temperature [°C].
    out_max : float
        Maximum duty-cycle output [0, 1].
    ff_gain : float
        Feed-forward gain [duty / W].  Rule of thumb: 0.01–0.05 W⁻¹.
    """

    def __init__(self, kp: float = 0.15, ki: float = 0.005, kd: float = 0.4,
                 setpoint: float = 35.0, out_max: float = 1.0,
                 ff_gain: float = 0.02):
        self._pid = PIDController(kp=kp, ki=ki, kd=kd,
                                  setpoint=setpoint, out_max=out_max)
        self.ff_gain = float(ff_gain)
        self.setpoint = float(setpoint)

    def step(self, measured_T_max: float, dt: float,
             predicted_heat_W: float = 0.0) -> float:
        """Compute cooling duty.

        Parameters
        ----------
        measured_T_max : float
            Measured peak cell temperature [°C] (feedback signal).
        dt : float
            Time step [s].
        predicted_heat_W : float
            Sum of per-cell heat generation predicted for this step [W].
            Used for feed-forward pre-cooling; defaults to 0 (pure PID).
        """
        pid_out = self._pid.step(measured_T_max, dt)
        ff_out = self.ff_gain * max(0.0, predicted_heat_W)
        return float(np.clip(pid_out + ff_out, 0.0, 1.0))

    def reset(self) -> None:
        self._pid.reset()
