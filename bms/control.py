"""
Supervisory control layer.

The :class:`BMSSupervisor` is a finite-state-machine that ties every
subsystem together and exposes a single ``step`` entry point.

States
------
* IDLE          — pack in low-load, balancing allowed if imbalance > th.
* PRECHARGE     — HV contactor pre-charging; pack current is gated to zero
                  until the DC-link reaches the target ratio of pack voltage.
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

from .balancing import Balancer, InductorBalancer, PassiveBalancer, SwitchedCapacitorBalancer
from .faults import FaultInjector, FaultMode, HybridFaultDetector, extract_features
from .pack import BatteryPack
from .passport import BatteryPassport
from .thermal import PredictiveCoolingController, ThermalModel


class BMSState(str, Enum):
    IDLE = "idle"
    PRECHARGE = "precharge"
    OPERATING = "operating"
    BALANCING = "balancing"
    FAULT = "fault"
    SHUTDOWN = "shutdown"


class ContactorState(str, Enum):
    """Physical high-voltage contactor sequence state."""

    OPEN = "open"
    PRECHARGING = "precharging"
    CLOSED = "closed"
    FAULT = "fault"


@dataclass
class PrechargeContactorSequencer:
    """Safely close an HV contactor after the DC-link capacitor is charged.

    Call :meth:`start` to close the pre-charge relay.  Feed the measured
    DC-link voltage to :meth:`update`; the main contactor closes only when
    that voltage reaches ``target_ratio`` of pack voltage before timeout.
    """

    target_ratio: float = 0.95
    timeout_s: float = 5.0
    min_pack_voltage_V: float = 1.0
    state: ContactorState = ContactorState.CLOSED
    elapsed_s: float = 0.0

    @property
    def is_closed(self) -> bool:
        return self.state == ContactorState.CLOSED

    def open(self) -> None:
        self.state = ContactorState.OPEN
        self.elapsed_s = 0.0

    def start(self) -> bool:
        """Begin pre-charge from an open contactor.  Returns success."""
        if self.state == ContactorState.FAULT:
            return False
        self.state = ContactorState.PRECHARGING
        self.elapsed_s = 0.0
        return True

    def update(self, pack_voltage_V: float, dc_link_voltage_V: float | None,
               dt: float) -> ContactorState:
        """Advance the sequence using the measured DC-link voltage."""
        if self.state != ContactorState.PRECHARGING:
            return self.state
        self.elapsed_s += max(0.0, float(dt))
        if pack_voltage_V < self.min_pack_voltage_V:
            self.state = ContactorState.FAULT
        elif (dc_link_voltage_V is not None
              and dc_link_voltage_V >= self.target_ratio * pack_voltage_V):
            self.state = ContactorState.CLOSED
        elif self.elapsed_s >= self.timeout_s:
            self.state = ContactorState.FAULT
        return self.state

    def reset(self) -> None:
        """Clear a pre-charge fault; an explicit new start is still needed."""
        self.state = ContactorState.OPEN
        self.elapsed_s = 0.0


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
    # Predictive-cooling controller gains (were hard-coded in the supervisor).
    cooling_kp: float = 0.15
    cooling_ki: float = 0.005
    cooling_kd: float = 0.4
    cooling_ff_gain: float = 0.02
    precharge_target_ratio: float = 0.95
    precharge_timeout_s: float = 5.0


@dataclass
class BMSSupervisor:
    pack: BatteryPack
    thermal: ThermalModel
    detector: HybridFaultDetector
    config: SupervisorConfig = field(default_factory=SupervisorConfig)
    state: BMSState = BMSState.IDLE
    # Optional fault injector: when set, faults are applied to the measured
    # signals each step so they actually reach the detector (rule + ML).
    injector: FaultInjector | None = None

    def __post_init__(self):
        self._cooling = PredictiveCoolingController(
            kp=self.config.cooling_kp, ki=self.config.cooling_ki,
            kd=self.config.cooling_kd,
            setpoint=self.config.T_setpoint_C, out_max=1.0,
            ff_gain=self.config.cooling_ff_gain,
        )
        self._inductor = InductorBalancer()
        self._sc = SwitchedCapacitorBalancer()
        self._passive = PassiveBalancer()
        self._alarm_streak = 0
        self._last_v_cells = self.pack.cell_voltages()
        self._last_T_cells = self.thermal.T.copy()
        self._last_cell_currents = np.zeros(self.pack.n_cells)
        self._fault_log: list[dict] = []
        # Existing simulations represent an already connected battery.  The
        # sequence is therefore initially closed; callers explicitly open and
        # start it when modelling vehicle key-on/pre-charge behaviour.
        self.contactor = PrechargeContactorSequencer(
            target_ratio=self.config.precharge_target_ratio,
            timeout_s=self.config.precharge_timeout_s,
        )

        # Battery Passport — initialised from chemistry props
        from .chemistry import get_chemistry_props
        _props = get_chemistry_props(self.pack.cfg.chemistry)
        self._v_min_cell = float(_props["v_min"])
        self.passport = BatteryPassport(
            nominal_capacity_Ah=float(np.mean(self.pack.capacities_Ah)),
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
    def _evaluate_faults(self, k: int, dt: float) -> tuple[str, str]:
        v_cells = self.pack.cell_voltages(temperatures_C=self.thermal.T)
        T_cells = self.thermal.T.copy()
        # Use the currents actually applied last step (not zeros), so the
        # current-dependent features and the short-circuit rule are live.
        currents = self._last_cell_currents.copy()

        # Inject faults into the *measured* signals so they reach the detector.
        # With no injector this is a no-op and detection runs on clean signals.
        if self.injector is not None:
            v_cells = self.injector.apply_to_voltage_meas(v_cells, k)
            T_cells = self.injector.apply_to_temperatures(T_cells, k, dt)
            currents = self.injector.apply_to_currents(currents, k)

        dv = v_cells - self._last_v_cells
        dT = T_cells - self._last_T_cells
        feats = extract_features(v_cells, currents, T_cells, dv, dT)
        label, src = self.detector.predict_step(feats, v_cells, T_cells, currents)
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
             requested_power_W: float | None = None,
             dc_link_voltage_V: float | None = None) -> dict:
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
            ``derated``, ``power_W``, ``peak_power_W``, ``soe_Wh``,
            ``contactor_state``, ``precharge_elapsed_s``.
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
        fault_label, fault_source = self._evaluate_faults(k, dt)

        # Pre-charge gates all current until the DC link is close enough to
        # pack voltage to close the main contactor.  A timeout is handled as a
        # fault, never by force-closing the contactor.
        precharge_gating = False
        if self.contactor.state == ContactorState.PRECHARGING:
            self.contactor.update(self.pack.pack_voltage(), dc_link_voltage_V, dt)
            if self.contactor.state == ContactorState.FAULT:
                self.state = BMSState.FAULT
            elif not self.contactor.is_closed:
                self.state = BMSState.PRECHARGE
                requested_pack_current_A = 0.0
                precharge_gating = True
        elif self.contactor.state in (ContactorState.OPEN, ContactorState.FAULT):
            requested_pack_current_A = 0.0

        # Thermal runaway is critical → latch to terminal SHUTDOWN at once.
        if (fault_source == "rule"
                and fault_label == FaultMode.THERMAL_RUNAWAY.value
                and self.state != BMSState.SHUTDOWN):
            self.state = BMSState.SHUTDOWN
        # Short circuit is an acute safety threat → trip contactor to FAULT immediately.
        elif (fault_source == "rule"
                and fault_label == FaultMode.SHORT_CIRCUIT.value
                and self.state not in (BMSState.FAULT, BMSState.SHUTDOWN)):
            self.state = BMSState.FAULT
        elif (self._alarm_streak >= self.config.consecutive_alarms_to_trip
                and self.state not in (BMSState.FAULT, BMSState.SHUTDOWN)):
            self.state = BMSState.FAULT

        # ---- 2. State logic / current command ------------------------
        cmd_current = requested_pack_current_A
        cooling_duty = 0.0
        derated = False

        if self.state in (BMSState.FAULT, BMSState.SHUTDOWN):
            if self.contactor.state != ContactorState.FAULT:
                self.contactor.open()
            cmd_current = 0.0
            cooling_duty = 1.0
            balancer = None
        elif precharge_gating:
            self.state = BMSState.PRECHARGE
            cmd_current = 0.0
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
        # Remember the applied per-group currents for next step's fault eval.
        self._last_cell_currents = np.asarray(currents_per_group, float)

        # Use group-level SOC and R0 (correct for both n_parallel=1 and >1).
        ocv = np.array([
            float(self.pack.ocv_curve.ocv(g.soc, T_C=float(self.thermal.T[i])))
            for i, g in enumerate(self.pack.groups)
        ])
        R0 = np.array([g.params.R0 for g in self.pack.groups])
        heat = ThermalModel.heat_generation(currents_per_group, R0,
                                            pack_step["v_cells"], ocv)

        if self.state not in (BMSState.FAULT, BMSState.SHUTDOWN):
            cooling_duty = self._cooling.step(
                float(self.thermal.T.max()), dt,
                predicted_heat_W=float(heat.sum()),
            )

        T_new = self.thermal.step(heat, cooling_duty, dt)

        # ---- 4. Power metrics ----------------------------------------
        v_pack = pack_step["v_pack"]
        power_W = float(v_pack * cmd_current)

        # Peak *deliverable* discharge power: bounded by the weakest series
        # group reaching the min-voltage cutoff (consistent with
        # diagnostics.compute_crate_map).  NOT the matched-load V_oc²/(4·R0),
        # which occurs at V_terminal = V_oc/2 — far below cutoff and physically
        # unreachable — and which also ignores the R1/R2 transient drops.
        ocv_pack = float(np.sum(ocv))
        R0_series = float(np.sum(R0))
        i_max_per_group = np.maximum(0.0, (ocv - self._v_min_cell)
                                     / np.maximum(R0, 1e-9))
        i_max_pack = float(i_max_per_group.min())      # weakest group limits I
        v_pack_at_imax = ocv_pack - R0_series * i_max_pack
        peak_power_W = max(0.0, v_pack_at_imax * i_max_pack)

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
            "contactor_state": self.contactor.state.value,
            "precharge_elapsed_s": self.contactor.elapsed_s,
        }

    # ------------------------------------------------------------------
    @property
    def fault_log(self) -> list[dict]:
        return list(self._fault_log)

    # ------------------------------------------------------------------
    def clear_fault(self) -> bool:
        """Operator reset: clear a latched ``FAULT`` and return to ``IDLE``.

        Resets the rule-alarm streak so the pack can resume operation.
        ``SHUTDOWN`` (entered on a critical fault such as thermal runaway) is
        terminal and is deliberately **not** cleared.  Returns ``True`` if a
        fault was cleared, ``False`` if there was nothing to clear or the state
        is terminal.
        """
        if self.state == BMSState.FAULT:
            self.state = BMSState.IDLE
            self._alarm_streak = 0
            if self.contactor.state == ContactorState.FAULT:
                self.contactor.reset()
            return True
        return False

    # ------------------------------------------------------------------
    def open_contactors(self) -> None:
        """Open the HV contactors, for example at vehicle key-off."""
        self.contactor.open()
        if self.state not in (BMSState.FAULT, BMSState.SHUTDOWN):
            self.state = BMSState.IDLE

    def start_precharge(self) -> bool:
        """Start a measured DC-link pre-charge sequence.

        While pre-charging, :meth:`step` requires no special mode; pass the
        latest ``dc_link_voltage_V`` and it will suppress pack current until
        the main contactor is safe to close.
        """
        if self.state == BMSState.SHUTDOWN:
            return False
        started = self.contactor.start()
        if started:
            self.state = BMSState.PRECHARGE
        return started

    # ------------------------------------------------------------------
    def state_of_power(self, config=None) -> dict:
        """State-of-power limits for the present (rested) pack/thermal state.

        Convenience wrapper around :class:`bms.sop.StateOfPower` using this
        supervisor's own pack and measured temperatures, so SOP is reachable
        without re-wiring the pack by hand.  Returns limits keyed by horizon
        in seconds.  Pass a :class:`bms.sop.SOPConfig` to set horizons/limits.
        """
        from .sop import SOPConfig, StateOfPower
        calc = StateOfPower(config or SOPConfig())
        return calc.calculate(self.pack, temperatures_C=self.thermal.T)
