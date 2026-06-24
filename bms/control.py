"""
Supervisory control layer.

The :class:`BMSSupervisor` is a finite-state-machine that ties every
subsystem together and exposes a single ``step`` entry point.

States
------
* IDLE          — pack in low-load, balancing allowed if imbalance > th.
* OPERATING     — load on, SOC estimation + thermal regulation active.
* BALANCING     — explicit balancing window (e.g. end-of-charge).
* FAULT         — a detected fault triggers protective action: open
                  contactor (zero pack current), increase cooling duty,
                  freeze balancing.
* SHUTDOWN      — terminal state after critical fault.

The supervisor dynamically selects a balancing strategy:
    imbalance ≥ 0.05 SOC and we have inductor hardware → InductorBalancer
    imbalance ≥ 0.02 SOC                              → SwitchedCapacitor
    end-of-charge with imbalance > 0.005              → PassiveBalancer

Improvements
------------
* **Temperature-aware pack stepping** — each group receives its measured
  temperature so the ECM applies Arrhenius-correct resistances and the OCV
  temperature correction is physically accurate.
* **Current de-rating near voltage cutoffs** — discharge current is tapered
  to zero as the weakest group's SOC approaches the low cutoff (5 % SOC over
  a 5 % linear ramp).
* **Power-mode input** — ``step`` accepts ``requested_power_W`` as an
  alternative to a current command; it converts power to current using the
  instantaneous pack voltage before applying de-rating.
* **Power tracking** — every step returns ``power_W`` (instantaneous) and
  ``peak_power_W`` (maximum available at current SOC / temperature).
* **Predictive cooling** — PID + feed-forward cooling reduces peak temperature
  overshoot by 1–3 °C under aggressive load steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .balancing import (Balancer, InductorBalancer, PassiveBalancer,
                        SwitchedCapacitorBalancer)
from .faults import FaultMode, HybridFaultDetector, extract_features
from .pack import BatteryPack
from .passport import BatteryPassport
from .thermal import PredictiveCoolingController, ThermalModel


class BMSState(str, Enum):
    IDLE = "idle"
    OPERATING = "operating"
    BALANCING = "balancing"
    FAULT = "fault"
    SHUTDOWN = "shutdown"


@dataclass
class SupervisorConfig:
    imbalance_inductor: float = 0.05
    imbalance_sc: float = 0.02
    imbalance_passive: float = 0.005
    soc_low_cutoff: float = 0.05
    soc_high_cutoff: float = 0.98
    T_warning_C: float = 55.0
    T_setpoint_C: float = 35.0
    consecutive_alarms_to_trip: int = 3
    # Current de-rating ramp width around each SOC cutoff.
    derate_ramp_soc: float = 0.05
    # Optional power limits [W] (np.inf = no limit).
    max_discharge_power_W: float = float("inf")
    max_charge_power_W: float = float("inf")


@dataclass
class BMSSupervisor:
    pack: BatteryPack
    thermal: ThermalModel
    detector: HybridFaultDetector
    config: SupervisorConfig = field(default_factory=SupervisorConfig)
    state: BMSState = BMSState.IDLE

    def __post_init__(self):
        self._cooling = PredictiveCoolingController(
            kp=0.15, ki=0.005, kd=0.4,
            setpoint=self.config.T_setpoint_C, out_max=1.0,
            ff_gain=0.02,
        )
        self._inductor = InductorBalancer()
        self._sc = SwitchedCapacitorBalancer()
        self._passive = PassiveBalancer()
        self._alarm_streak = 0
        self._last_v_cells = self.pack.cell_voltages()
        self._last_T_cells = self.thermal.T.copy()
        self._fault_log: list[dict] = []

        # Battery Passport — initialised from chemistry props
        from .chemistry import get_chemistry_props
        _props = get_chemistry_props(self.pack.cfg.chemistry)
        self.passport = BatteryPassport(
            nominal_capacity_Ah=float(self.pack.capacities_Ah.sum()),
            nominal_voltage_V=_props["nominal_voltage_V"] * self.pack.n_cells,
            chemistry=self.pack.cfg.chemistry,
        )

    # ------------------------------------------------------------------
    def _select_balancer(self) -> Balancer | None:
        imb = self.pack.soc_imbalance()
        if imb >= self.config.imbalance_inductor:
            return self._inductor
        if imb >= self.config.imbalance_sc:
            return self._sc
        if imb >= self.config.imbalance_passive and self.pack.soc.mean() > 0.9:
            return self._passive
        return None

    # ------------------------------------------------------------------
    def _derate_current(self, requested_A: float) -> float:
        """Taper current near SOC voltage cutoffs to avoid hard trips."""
        cfg = self.config
        soc = self.pack.soc

        if requested_A > 0:
            min_soc = float(soc.min())
            margin = min_soc - cfg.soc_low_cutoff
            if margin <= cfg.derate_ramp_soc:
                factor = max(0.0, margin / cfg.derate_ramp_soc)
                requested_A *= factor

        elif requested_A < 0:
            max_soc = float(soc.max())
            margin = cfg.soc_high_cutoff - max_soc
            if margin <= cfg.derate_ramp_soc:
                factor = max(0.0, margin / cfg.derate_ramp_soc)
                requested_A *= factor

        return requested_A

    # ------------------------------------------------------------------
    def _evaluate_faults(self, k: int) -> tuple[str, str]:
        v_cells = self.pack.cell_voltages(temperatures_C=self.thermal.T)
        T_cells = self.thermal.T.copy()
        dv = v_cells - self._last_v_cells
        dT = T_cells - self._last_T_cells
        currents = np.zeros(self.pack.n_cells)
        feats = extract_features(v_cells, currents, T_cells, dv, dT)
        label, src = self.detector.predict_step(feats, v_cells, T_cells)
        self._last_v_cells, self._last_T_cells = v_cells, T_cells
        if label != FaultMode.NONE.value:
            if src == "rule":
                self._alarm_streak += 1
            self._fault_log.append({"step": k, "mode": label, "source": src})
        else:
            self._alarm_streak = 0
        return label, src

    # ------------------------------------------------------------------
    def step(self, requested_pack_current_A: float = 0.0, dt: float = 1.0,
             k: int = 0,
             requested_power_W: float | None = None) -> dict:
        """One control cycle.

        Parameters
        ----------
        requested_pack_current_A : float
            Requested series current (positive = discharge, negative = charge).
            Ignored when ``requested_power_W`` is given.
        dt : float
            Time step [s].
        k : int
            Step index (used in fault log).
        requested_power_W : float, optional
            Requested pack power [W] (positive = discharge).  When provided the
            current is computed as ``P / V_pack`` using the instantaneous pack
            voltage before any de-rating is applied.  Power limits from
            ``config.max_discharge_power_W`` and ``config.max_charge_power_W``
            are applied before current conversion.

        Returns
        -------
        dict
            Keys: ``state``, ``fault_label``, ``fault_source``, ``v_cells``,
            ``v_pack``, ``soc``, ``T_cells``, ``cooling_duty``, ``balancer``,
            ``balancing_currents``, ``imbalance``, ``cmd_current``,
            ``derated``, ``power_W``, ``peak_power_W``.
        """
        # ---- 0. Power → current conversion ---------------------------
        if requested_power_W is not None:
            cfg = self.config
            # Apply power limits.
            if requested_power_W > 0:
                requested_power_W = min(requested_power_W, cfg.max_discharge_power_W)
            elif requested_power_W < 0:
                requested_power_W = max(requested_power_W, -cfg.max_charge_power_W)
            v_pack_now = self.pack.pack_voltage()
            if abs(v_pack_now) > 1e-3:
                requested_pack_current_A = requested_power_W / v_pack_now
            else:
                requested_pack_current_A = 0.0

        # ---- 1. Fault evaluation -------------------------------------
        fault_label, fault_source = self._evaluate_faults(k)

        if (self._alarm_streak >= self.config.consecutive_alarms_to_trip
                and self.state != BMSState.FAULT):
            self.state = BMSState.FAULT

        # ---- 2. State logic / current command ------------------------
        cmd_current = requested_pack_current_A
        cooling_duty = 0.0
        derated = False

        if self.state == BMSState.FAULT:
            cmd_current = 0.0
            cooling_duty = 1.0
            balancer = None
        else:
            if abs(requested_pack_current_A) > 1e-3:
                self.state = BMSState.OPERATING
            else:
                self.state = (BMSState.BALANCING
                              if self.pack.soc_imbalance() > self.config.imbalance_passive
                              else BMSState.IDLE)

            derated_current = self._derate_current(requested_pack_current_A)
            if abs(derated_current) < abs(requested_pack_current_A) - 1e-6:
                derated = True
            cmd_current = derated_current

            balancer = self._select_balancer()

        bal_currents = (balancer.step(self.pack, dt) if balancer is not None
                        else np.zeros(self.pack.n_cells))

        # ---- 3. Advance physical models ------------------------------
        pack_step = self.pack.step(cmd_current, dt,
                                   balancing_currents=bal_currents,
                                   cell_temperatures_C=self.thermal.T)
        currents_per_group = cmd_current + bal_currents

        # Use group-level SOC and R0 (correct for both n_parallel=1 and >1).
        ocv = np.array([
            float(self.pack.ocv_curve.ocv(g.soc, T_C=float(self.thermal.T[i])))
            for i, g in enumerate(self.pack.groups)
        ])
        R0 = np.array([g.params.R0 for g in self.pack.groups])
        heat = ThermalModel.heat_generation(currents_per_group, R0,
                                            pack_step["v_cells"], ocv)

        if self.state != BMSState.FAULT:
            cooling_duty = self._cooling.step(
                float(self.thermal.T.max()), dt,
                predicted_heat_W=float(heat.sum()),
            )

        T_new = self.thermal.step(heat, cooling_duty, dt)

        # ---- 4. Power metrics ----------------------------------------
        v_pack = pack_step["v_pack"]
        power_W = float(v_pack * cmd_current)

        # Peak power capability: V_oc² / (4 × R0_series)
        # where R0_series = sum of group R0s (already the effective parallel R0)
        ocv_pack = float(np.sum(ocv))
        R0_series = float(np.sum(R0))
        peak_power_W = (ocv_pack ** 2) / max(4.0 * R0_series, 1e-9)

        # ---- 5. Passport + State of Energy ---------------------------
        soc_mean = float(pack_step["soc"].mean())
        self.passport.update(cmd_current, v_pack, dt, soc_mean=soc_mean)
        soe_Wh = self.pack.state_of_energy_Wh()

        return {
            "state": self.state.value,
            "fault_label": fault_label,
            "fault_source": fault_source,
            "v_cells": pack_step["v_cells"],
            "v_pack": v_pack,
            "soc": pack_step["soc"],
            "T_cells": T_new.copy(),
            "cooling_duty": cooling_duty,
            "balancer": (balancer.name if balancer is not None else "none"),
            "balancing_currents": bal_currents,
            "imbalance": self.pack.soc_imbalance(),
            "cmd_current": cmd_current,
            "derated": derated,
            "power_W": power_W,
            "peak_power_W": peak_power_W,
            "soe_Wh": soe_Wh,
        }

    # ------------------------------------------------------------------
    @property
    def fault_log(self) -> list[dict]:
        return list(self._fault_log)
