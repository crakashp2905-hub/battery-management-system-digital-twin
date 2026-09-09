"""Tests: State-of-Safety, HV safety (IMD/weld), sensor FDI."""

from __future__ import annotations

import numpy as np

import bms


class TestStateOfSafety:
    @staticmethod
    def _result(**over):
        r = {"T_cells": np.full(4, 25.0), "v_cells": np.full(4, 3.7),
             "imbalance": 0.01}
        r.update(over)
        return r

    def test_nominal_is_fully_safe(self):
        sos = bms.state_of_safety(self._result(), chemistry="nmc", soh=0.98)
        assert sos["sos"] == 1.0
        assert sos["severity"] == "ok"
        assert sos["worst_signal"] == "none"

    def test_hot_cell_drives_sos_to_zero(self):
        sos = bms.state_of_safety(
            self._result(T_cells=np.array([25.0, 25.0, 72.0, 25.0])),
            chemistry="nmc", soh=0.98)
        assert sos["sos"] == 0.0
        assert sos["severity"] == "critical"
        assert sos["worst_signal"] == "temperature"

    def test_degrades_before_threshold_trips(self):
        # 55 °C is between warn (45) and crit (70) → partial penalty, not 0 or 1.
        sos = bms.state_of_safety(
            self._result(T_cells=np.full(4, 55.0)), chemistry="nmc", soh=0.98)
        assert 0.0 < sos["sos"] < 1.0
        assert sos["severity"] in {"info", "warning"}

    def test_worst_signal_governs(self):
        # Low SoH plus a hot cell → temperature (critical) dominates.
        sos = bms.state_of_safety(
            self._result(T_cells=np.array([25.0, 71.0, 25.0, 25.0])),
            chemistry="nmc", soh=0.5)
        assert sos["worst_signal"] == "temperature"
        assert "soh" in sos["breakdown"]           # still reported in the breakdown

    def test_gas_vent_is_critical(self):
        state = bms.CellMechanicalState(pressure_kPa=250.0)
        state.vent_event = True
        sos = bms.state_of_safety(self._result(), mechanical_state=state,
                                  chemistry="nmc")
        assert sos["breakdown"]["gas_pressure"] == 1.0


class TestInsulationMonitor:
    def test_ohm_per_volt_thresholds(self):
        imd = bms.InsulationMonitor(warn_ohm_per_volt=500.0, fault_ohm_per_volt=100.0)
        imd.update(400.0, 1e-3)                    # 400 kΩ → 1000 Ω/V
        assert imd.status(400.0) == "ok"
        imd.update(400.0, 2e-3)                    # 200 kΩ → 500 Ω/V → warning edge
        assert imd.status(400.0) in {"ok", "warning"}
        imd.update(400.0, 2e-2)                    # 20 kΩ → 50 Ω/V → fault
        assert imd.status(400.0) == "fault"

    def test_no_leakage_is_infinite_resistance(self):
        imd = bms.InsulationMonitor()
        imd.update(400.0, 0.0)
        assert imd.resistance_ohm == float("inf")
        assert imd.status(400.0) == "ok"


class TestContactorWeldDetector:
    def test_detects_weld_when_link_stays_high(self):
        wd = bms.ContactorWeldDetector(settle_s=2.0)      # bleed_ratio 0.5 → 200 V
        welded = False
        for _ in range(4):
            welded = wd.update(True, 395.0, 400.0, 1.0)   # link tracks pack → welded
        assert welded

    def test_healthy_contactor_bleeds_down(self):
        wd = bms.ContactorWeldDetector(settle_s=2.0)
        welded = False
        for v in (300.0, 120.0, 20.0, 2.0):
            welded = wd.update(True, v, 400.0, 1.0)       # < 200 V by settle → healthy
        assert not welded

    def test_reset_and_closed_command_clears_timer(self):
        wd = bms.ContactorWeldDetector(settle_s=2.0)
        wd.update(False, 400.0, 400.0, 5.0)               # commanded closed → no fault
        assert not wd.welded


class TestSensorFDI:
    def test_ok_signals_pass(self):
        fdi = bms.SensorFDI.for_chemistry("nmc")
        assert not fdi.check(3.7, 5.0, 25.0)["any_fault"]

    def test_dropout_out_of_range_and_stuck(self):
        fdi = bms.SensorFDI.for_chemistry("nmc")
        assert fdi.check(float("nan"), 5.0, 25.0)["voltage"] == "dropout"
        assert fdi.check(9.9, 5.0, 25.0)["voltage"] == "out_of_range"
        # Feed a perfectly flat voltage past the stuck window → stuck.
        mon = bms.SensorMonitor(lo=2.5, hi=4.5, max_rate=1.0, stuck_window=10)
        labels = [mon.check(3.7) for _ in range(12)]
        assert labels[-1] == "stuck"

    def test_rate_spike_flagged(self):
        mon = bms.SensorMonitor(lo=2.5, hi=4.5, max_rate=0.2)
        mon.check(3.7)
        assert mon.check(4.4) == "rate"            # 0.7 V jump > 0.2 V/step

    def test_virtual_voltage_replaces_dropped_sensor(self):
        p = bms.ECMParameters(R0=0.025, Q_nom_Ah=2.3)
        ocv = bms.OCVSOC()
        v_hat = bms.virtual_cell_voltage(p, ocv, soc=0.6, current=2.0)
        true = float(ocv.ocv(0.6, 2.0)) - 0.025 * 2.0
        assert abs(v_hat - true) < 1e-9
