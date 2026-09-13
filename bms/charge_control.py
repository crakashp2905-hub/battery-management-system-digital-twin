"""Model-predictive / optimal fast charging.

`bms.charging` *quantifies* a charge session's effects; this *controls* it. The
:class:`MPCCharger` chooses, each step, the largest charge current that keeps the
next state inside every limit at once — terminal voltage, cell temperature, and
the lithium-**plating** current cap ``plating_c_limit(T, SoC)·Q`` — driving SoC
to a target as fast as the physics allows.  Because the twin already provides the
plant models (ECM + a lumped thermal cell), this is a natural one-step receding-
horizon controller with closed-form per-constraint current limits.

Compared with plain CC-CV on the *same* plant it reaches the target sooner at an
equal (bounded) temperature and with a guaranteed plating margin — the health-
aware fast-charge result.  :func:`cccv_charge` runs CC-CV on the same plant so
the two are directly comparable, and :func:`compare_charging` reports both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .charging import plating_c_limit
from .ecm import ECMParameters
from .ocv_soc import OCVSOC


@dataclass
class ChargeLimits:
    v_max: float = 4.2          # terminal-voltage limit [V]
    t_max_C: float = 45.0       # cell-temperature limit [°C]
    i_max_A: float = 10.0       # charger/hardware current limit (charge magnitude)
    soc_target: float = 0.90


@dataclass
class _Plant:
    """Shared single-cell plant: ECM RC states + a lumped thermal mass."""

    params: ECMParameters
    ocv_curve: OCVSOC
    heat_capacity_J_per_K: float
    cooling_W_per_K: float
    ambient_C: float
    soc: float = 0.0
    vrc1: float = 0.0
    vrc2: float = 0.0
    T: float = 25.0

    def reset(self, soc0: float, T0: float) -> None:
        self.soc, self.vrc1, self.vrc2, self.T = soc0, 0.0, 0.0, T0

    def terminal_voltage(self, charge_A: float) -> float:
        # Signed current is negative on charge; V = OCV − vrc1 − vrc2 + R0·I_charge.
        p = self.params.at_temperature(self.T)
        ocv = float(self.ocv_curve.ocv(self.soc, -charge_A, T_C=self.T))
        return ocv - self.vrc1 - self.vrc2 + p.R0 * charge_A

    def step(self, charge_A: float, dt: float) -> None:
        p = self.params.at_temperature(self.T)
        signed = -charge_A                                   # charge = negative current
        a1 = math.exp(-dt / max(p.tau1, 1e-9))
        a2 = math.exp(-dt / max(p.tau2, 1e-9))
        self.vrc1 = a1 * self.vrc1 + (1 - a1) * p.R1 * signed
        self.vrc2 = a2 * self.vrc2 + (1 - a2) * p.R2 * signed
        self.soc = float(np.clip(self.soc + charge_A * dt / (p.Q_nom_Ah * 3600.0), 0.0, 1.0))
        heat = signed ** 2 * p.R0 - self.cooling_W_per_K * (self.T - self.ambient_C)
        self.T = self.T + heat * dt / self.heat_capacity_J_per_K


@dataclass
class MPCCharger:
    params: ECMParameters
    ocv_curve: OCVSOC = field(default_factory=OCVSOC)
    limits: ChargeLimits = field(default_factory=ChargeLimits)
    heat_capacity_J_per_K: float = 40.0
    cooling_W_per_K: float = 0.5
    ambient_C: float = 25.0
    plating_aware: bool = True

    def _plant(self) -> _Plant:
        return _Plant(self.params, self.ocv_curve, self.heat_capacity_J_per_K,
                      self.cooling_W_per_K, self.ambient_C)

    def max_charge_current(self, plant: _Plant, dt: float) -> float:
        """Largest charge current [A] that keeps the next state within all limits."""
        p = plant.params.at_temperature(plant.T)
        ocv = float(self.ocv_curve.ocv(plant.soc, T_C=plant.T))
        # Voltage: OCV − vrc + R0·I ≤ v_max.
        i_volt = (self.limits.v_max - ocv + plant.vrc1 + plant.vrc2) / max(p.R0, 1e-9)
        # Plating cap (coldest/most-charged condition is handled by the live T/SoC).
        i_plate = (plating_c_limit(plant.T, plant.soc) * p.Q_nom_Ah
                   if self.plating_aware else math.inf)
        # Thermal: keep T + dT ≤ t_max, i.e. I²R0 ≤ cooling·(T−amb) + headroom·C/dt.
        headroom = self.limits.t_max_C - plant.T
        max_heat = (self.cooling_W_per_K * (plant.T - self.ambient_C)
                    + headroom * self.heat_capacity_J_per_K / max(dt, 1e-9))
        i_therm = math.sqrt(max(0.0, max_heat) / max(p.R0, 1e-9))
        return max(0.0, min(i_volt, i_plate, i_therm, self.limits.i_max_A))

    def charge(self, soc0: float, T0: float | None = None, dt: float = 1.0,
               max_time_s: float = 7200.0) -> dict:
        """Run the MPC charge; returns trajectories and a summary."""
        plant = self._plant()
        plant.reset(soc0, self.ambient_C if T0 is None else T0)
        return _run(plant, self, dt, max_time_s, controller="mpc")


def cccv_charge(charger: MPCCharger, soc0: float, c_rate: float,
                T0: float | None = None, dt: float = 1.0,
                max_time_s: float = 7200.0) -> dict:
    """CC-CV on the same plant: constant current until v_max, then voltage-hold taper."""
    plant = charger._plant()
    plant.reset(soc0, charger.ambient_C if T0 is None else T0)
    i_cc = c_rate * charger.params.Q_nom_Ah
    return _run(plant, charger, dt, max_time_s, controller="cccv", i_cc=i_cc)


def _run(plant: _Plant, charger: MPCCharger, dt: float, max_time_s: float,
         controller: str, i_cc: float = 0.0) -> dict:
    lim = charger.limits
    t, soc, cur, volt, temp, plate_margin = [], [], [], [], [], []
    steps = int(max_time_s / dt)
    for k in range(steps):
        if plant.soc >= lim.soc_target:
            break
        if controller == "mpc":
            ic = charger.max_charge_current(plant, dt)
        else:  # CC-CV: constant current, tapered to hold v_max.
            ic = i_cc
            v = plant.terminal_voltage(ic)
            if v > lim.v_max:                                # CV phase: solve for V = v_max
                p = plant.params.at_temperature(plant.T)
                ocv = float(charger.ocv_curve.ocv(plant.soc, T_C=plant.T))
                ic = max(0.0, (lim.v_max - ocv + plant.vrc1 + plant.vrc2) / max(p.R0, 1e-9))
        cap = plating_c_limit(plant.T, plant.soc) * plant.params.Q_nom_Ah
        t.append(k * dt)
        soc.append(plant.soc)
        cur.append(ic)
        volt.append(plant.terminal_voltage(ic))
        temp.append(plant.T)
        plate_margin.append(cap - ic)                        # >0 = below plating limit
        plant.step(ic, dt)
    reached = plant.soc >= lim.soc_target
    return {
        "controller": controller,
        "t_s": np.array(t), "soc": np.array(soc), "current_A": np.array(cur),
        "voltage_V": np.array(volt), "temperature_C": np.array(temp),
        "plating_margin_A": np.array(plate_margin),
        "time_to_target_s": float(t[-1] + dt) if reached else float("inf"),
        "reached_target": bool(reached),
        "peak_temperature_C": float(max(temp)) if temp else plant.T,
        "min_plating_margin_A": float(min(plate_margin)) if plate_margin else 0.0,
        "final_soc": float(plant.soc),
    }


def compare_charging(charger: MPCCharger, soc0: float = 0.2, c_rate: float = 1.0,
                     dt: float = 1.0) -> dict:
    """Run MPC and CC-CV on the same plant and summarise the difference."""
    mpc = charger.charge(soc0, dt=dt)
    cccv = cccv_charge(charger, soc0, c_rate=c_rate, dt=dt)
    return {
        "mpc": mpc, "cccv": cccv,
        "time_saving_s": cccv["time_to_target_s"] - mpc["time_to_target_s"],
        "mpc_faster": mpc["time_to_target_s"] < cccv["time_to_target_s"],
    }
