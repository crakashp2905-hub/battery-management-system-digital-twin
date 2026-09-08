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


class TestPrechargeCircuit:
    def test_rc_charges_toward_pack_voltage(self):
        circ = bms.PrechargeCircuit(resistance_ohm=22.0, capacitance_F=1e-3)
        assert circ.tau_s == pytest.approx(0.022)
        vp = 400.0
        # After 3τ the bus should be ~95 % of pack; at 5τ ~99 %.
        for _ in range(3):
            circ.step(vp, circ.tau_s)
        assert circ.ratio(vp) == pytest.approx(1 - np.exp(-3), abs=0.02)
        for _ in range(2):
            circ.step(vp, circ.tau_s)
        assert circ.v_dc == pytest.approx(vp * (1 - np.exp(-5)), abs=vp * 0.01)

    def test_peak_inrush_is_at_t0_and_tracked(self):
        circ = bms.PrechargeCircuit(resistance_ohm=20.0, capacitance_F=1e-3)
        vp = 400.0
        first = circ.step(vp, 0.001)
        # Peak inrush = V_pack / R at t=0 (empty cap).
        assert first["inrush_current_A"] == pytest.approx(vp / 20.0)
        assert circ.peak_inrush_A == pytest.approx(vp / 20.0)
        later = circ.step(vp, 0.001)
        assert later["inrush_current_A"] < first["inrush_current_A"]  # decays
        assert circ.peak_inrush_A == pytest.approx(vp / 20.0)         # peak retained

    def test_resistor_energy_approaches_half_c_v_squared(self):
        circ = bms.PrechargeCircuit(resistance_ohm=22.0, capacitance_F=1e-3)
        vp = 400.0
        for _ in range(2000):
            circ.step(vp, 0.001)          # ~90τ → essentially fully charged
        expected = 0.5 * 1e-3 * vp ** 2   # energy burned in R equals energy stored
        assert circ.resistor_energy_J == pytest.approx(expected, rel=0.02)

    def test_reset_clears_state(self):
        circ = bms.PrechargeCircuit()
        circ.step(400.0, 0.05)
        assert circ.v_dc > 0.0
        circ.reset()
        assert circ.v_dc == 0.0 and circ.peak_inrush_A == 0.0
        assert circ.resistor_energy_J == 0.0

    def test_supervisor_simulate_precharge_closes_without_external_voltage(self):
        cfg = bms.SupervisorConfig(simulate_precharge=True,
                                   precharge_resistance_ohm=5.0,
                                   precharge_capacitance_F=1e-3)
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=7))
        sup = bms.BMSSupervisor(pack, bms.ThermalModel(n_cells=4),
                                bms.HybridFaultDetector(), config=cfg)
        sup.open_contactors()
        assert sup.start_precharge()
        # No dc_link_voltage_V supplied — the internal RC plant provides it.
        closed = None
        for _ in range(20):
            out = sup.step(10.0, 0.02, dc_link_voltage_V=None)
            if out["contactor_state"] == "closed":
                closed = out
                break
        assert closed is not None                 # bus charged and main contactor closed
        assert sup.precharge_circuit.peak_inrush_A > 0.0



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


class TestOnlineEstimation:
    def _sup(self, **cfg_kw):
        cfg = bms.SupervisorConfig(**cfg_kw)
        return bms.BMSSupervisor(
            bms.BatteryPack(bms.PackConfig(n_cells=4, seed=3)),
            bms.ThermalModel(n_cells=4), bms.HybridFaultDetector(), config=cfg)

    def test_off_by_default_returns_nan(self):
        out = self._sup().step(1.0, 1.0)
        for key in ("soc_estimated", "soh_estimated", "soc_sigma",
                    "soh_sigma", "capacity_est_Ah"):
            assert key in out
            assert np.isnan(out[key])          # no filter running

    def test_online_joint_ekf_tracks_soc_and_soh(self):
        sup = self._sup(estimate_online=True)
        out = None
        for k in range(300):
            out = sup.step(2.0, 1.0, k=k)       # steady discharge
        soc_true = float(np.mean(out["soc"]))
        assert abs(out["soc_estimated"] - soc_true) < 0.03   # SoC tracks truth
        assert out["soc_sigma"] > 0.0                         # reports uncertainty
        assert 0.8 <= out["soh_estimated"] <= 1.05            # SoH near BoL
        assert out["capacity_est_Ah"] > 0.0

    def test_one_filter_drives_soh_aware_control(self):
        # SoC and SoH come from the *same* online filter, and that SoH is what
        # feeds the control layer (config.online_feeds_soh, on by default).
        sup = self._sup(estimate_online=True, soh_aware=True)
        out = sup.step(1.0, 1.0)
        assert out["soh_capacity"] == pytest.approx(out["soh_estimated"])

