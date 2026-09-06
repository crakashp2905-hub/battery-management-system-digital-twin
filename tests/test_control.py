"""Tests: control."""

from __future__ import annotations

import numpy as np
import pytest

import bms


class TestPowerMode:
    def test_power_profile_shape(self):
        p = bms.generate_power_profile(100.0, dt=1.0, mode="constant",
                                       p_rate=1.0, nominal_capacity_Ah=2.3,
                                       nominal_voltage_V=3.7)
        assert p.shape == (100,)

    def test_power_profile_positive_for_discharge(self):
        p = bms.generate_power_profile(60.0, dt=1.0, mode="constant")
        assert np.all(p >= 0)

    def test_power_profile_magnitude(self):
        # 1-C rate × 2.3 Ah × 3.7 V ≈ 8.51 W
        p = bms.generate_power_profile(100.0, dt=1.0, mode="constant",
                                       p_rate=1.0, nominal_capacity_Ah=2.3,
                                       nominal_voltage_V=3.7)
        assert p.mean() == pytest.approx(2.3 * 3.7, rel=0.01)

    def test_supervisor_power_mode_returns_keys(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=0))
        thermal = bms.ThermalModel(n_cells=4)
        det = bms.HybridFaultDetector()
        sup = bms.BMSSupervisor(pack, thermal, det)
        out = sup.step(requested_power_W=5.0, dt=1.0)
        assert "power_W" in out
        assert "peak_power_W" in out
        assert out["peak_power_W"] > 0

    def test_supervisor_power_mode_current_conversion(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=0))
        thermal = bms.ThermalModel(n_cells=4)
        det = bms.HybridFaultDetector()
        sup = bms.BMSSupervisor(pack, thermal, det)
        v_pack = pack.pack_voltage()
        P_req = 5.0
        out = sup.step(requested_power_W=P_req, dt=1.0)
        # cmd_current ≈ P / V_pack  (before de-rating)
        expected_I = P_req / v_pack
        assert abs(out["cmd_current"]) <= abs(expected_I) + 0.01

    def test_current_mode_still_returns_power_keys(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=0))
        thermal = bms.ThermalModel(n_cells=4)
        det = bms.HybridFaultDetector()
        sup = bms.BMSSupervisor(pack, thermal, det)
        out = sup.step(1.0, 1.0)
        assert "power_W" in out
        assert "peak_power_W" in out



class TestStateOfPower:
    @staticmethod
    def _pack() -> bms.BatteryPack:
        return bms.BatteryPack(bms.PackConfig(n_cells=4, seed=11))

    def test_multi_horizon_traction_limit_decreases_with_duration(self):
        pack = self._pack()
        sop = bms.StateOfPower(bms.SOPConfig(
            max_discharge_current_A=1_000.0, max_charge_current_A=1_000.0,
        ))
        limits = sop.calculate(pack)
        assert set(limits) == {2.0, 10.0, 30.0}
        assert limits[2.0].traction_current_A > limits[30.0].traction_current_A > 0.0
        assert limits[2.0].regen_power_W > 0.0

    def test_temperature_derating_caps_both_current_directions(self):
        pack = self._pack()
        sop = bms.StateOfPower(bms.SOPConfig(
            max_discharge_current_A=100.0, max_charge_current_A=80.0,
            temperature_warning_C=40.0, temperature_limit_C=60.0,
        ))
        limits = sop.calculate(pack, temperatures_C=np.full(pack.n_cells, 50.0))
        limit = limits[10.0]
        assert limit.temperature_derate == pytest.approx(0.5)
        assert limit.traction_current_A <= 50.0
        assert limit.regen_current_A <= 40.0



class TestPrechargeContactors:
    def test_precharge_blocks_current_until_dc_link_is_charged(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=4))
        sup = bms.BMSSupervisor(pack, bms.ThermalModel(n_cells=4),
                                bms.HybridFaultDetector())
        sup.open_contactors()
        assert sup.start_precharge()
        first = sup.step(10.0, 1.0, dc_link_voltage_V=0.0)
        assert first["state"] == "precharge"
        assert first["cmd_current"] == 0.0
        closed = sup.step(10.0, 1.0, dc_link_voltage_V=pack.pack_voltage())
        assert closed["contactor_state"] == "closed"
        assert closed["cmd_current"] > 0.0

    def test_precharge_timeout_faults_without_closing_main_contactor(self):
        seq = bms.PrechargeContactorSequencer(timeout_s=2.0)
        seq.open()
        assert seq.start()
        assert seq.update(400.0, 0.0, 1.0) == bms.ContactorState.PRECHARGING
        assert seq.update(400.0, 0.0, 1.0) == bms.ContactorState.FAULT



class TestSoHAwareControl:
    def _sup(self, **cfg_kw):
        cfg = bms.SupervisorConfig(**cfg_kw)
        return bms.BMSSupervisor(
            bms.BatteryPack(bms.PackConfig(n_cells=4, seed=1)),
            bms.ThermalModel(n_cells=4), bms.HybridFaultDetector(), config=cfg)

    def test_off_by_default_leaves_current_unchanged(self):
        sup = self._sup()                      # soh_aware defaults False
        sup.set_soh(0.60)                       # very aged
        assert sup.step(10.0, 1.0)["cmd_current"] == pytest.approx(10.0)

    def test_capacity_soh_derates_current(self):
        sup = self._sup(soh_aware=True)
        sup.set_soh(1.0)
        assert sup.step(10.0, 1.0)["cmd_current"] == pytest.approx(10.0)
        sup.set_soh(0.70)                       # at floor → min factor 0.5
        out = sup.step(10.0, 1.0)
        assert out["cmd_current"] == pytest.approx(5.0)
        assert out["derated"] is True

    def test_soh_derating_is_monotonic(self):
        sup = self._sup(soh_aware=True)
        vals = []
        for soh in (1.0, 0.90, 0.85, 0.80, 0.70, 0.60):
            sup.set_soh(soh)
            vals.append(sup.step(10.0, 1.0)["cmd_current"])
        assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))   # non-increasing
        assert vals[0] == pytest.approx(10.0)
        assert vals[-1] == pytest.approx(5.0)

    def test_plating_cap_limits_cold_charge(self):
        sup = self._sup(soh_aware=True, plating_aware_charge=True)
        sup.thermal.T[:] = 0.0                  # cold → low plating C-limit
        out = sup.step(-30.0, 1.0)              # aggressive charge request
        assert -30.0 < out["cmd_current"] < 0.0
        assert abs(out["cmd_current"]) < 15.0   # capped well below the request

    def test_set_soh_clamped_and_reported(self):
        sup = self._sup(soh_aware=True)
        sup.set_soh(1.5, soh_resistance=0.5)    # out-of-range inputs
        assert sup.soh_capacity == 1.0
        assert sup.step(1.0, 1.0)["soh_capacity"] == 1.0

