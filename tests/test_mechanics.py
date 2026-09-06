"""Tests: mechanics."""

from __future__ import annotations

import numpy as np
import pytest

import bms


class TestMechanics:
    def test_pressure_rises_with_temperature_and_vents(self):
        pm = bms.PressureModel()
        st = bms.CellMechanicalState()
        max_p, vented = st.pressure_kPa, False
        for T in np.linspace(25, 95, 90):
            st = pm.update(st, float(T), soc=0.9, dt=1.0)
            max_p = max(max_p, st.pressure_kPa)
            vented = vented or st.vented
        assert max_p > 300.0                 # pressure builds substantially
        assert vented                        # crosses the safety-vent threshold
        assert st.h2_ppm > 0.0 and st.swelling_mm > 0.0

    def test_pressure_warning_leads_temperature(self):
        # The headline result: pressure/gas trips *before* the temperature rule.
        pm = bms.PressureModel()
        det = bms.MechanicalFaultDetector()
        st = bms.CellMechanicalState()
        t_ramp = np.linspace(25, 100, 150)
        t_temp = next(k for k, T in enumerate(t_ramp) if T >= 70.0)  # NMC runaway rule
        t_press, prev_p = None, st.pressure_kPa
        for k, T in enumerate(t_ramp):
            st = pm.update(st, float(T), 0.9, 1.0)
            label, src = det.predict_step(st, prev_pressure_kPa=prev_p, dt=1.0)
            if t_press is None and label != "none":
                t_press = k
                assert src == "rule"
            prev_p = st.pressure_kPa
        assert t_press is not None
        assert t_press < t_temp              # pressure LEADS temperature

    def test_venting_releases_gas_and_heat(self):
        pm = bms.PressureModel()
        st = bms.CellMechanicalState()
        heat = 0.0
        for _ in range(200):
            st = pm.update(st, 90.0, 0.9, 1.0)
            if st.vent_event:
                heat = pm.vent_heat_J(st)
                break
        assert st.vented and heat > 0.0
        assert st.pressure_kPa < pm.params.vent_pressure_kPa   # dropped after release

    def test_internal_short_via_coulombic_efficiency(self):
        det = bms.MechanicalFaultDetector()
        st = bms.CellMechanicalState()       # benign pressure/swelling
        assert det.predict_step(st, coulombic_efficiency=0.95) == ("internal_short", "rule")
        assert det.predict_step(st, coulombic_efficiency=1.0)[0] == "none"

    def test_swelling_detected(self):
        det = bms.MechanicalFaultDetector()
        assert det.predict_step(bms.CellMechanicalState(swelling_mm=2.0))[0] == "swelling"

    def test_coulombic_efficiency_helper(self):
        assert bms.coulombic_efficiency(10.0, 9.8) == pytest.approx(0.98)
        assert bms.coulombic_efficiency(0.0, 0.0) == 1.0
        assert bms.coulombic_efficiency(10.0, 12.0) == 1.0     # capped at 1.0

    def test_new_fault_modes_and_can_codes(self):
        names = {"gas_venting", "internal_short", "swelling", "electrolyte_leak"}
        assert names <= {m.value for m in bms.FaultMode}
        assert names <= set(bms.BMSCanBus.FAULT_CODES)

