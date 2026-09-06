"""Tests: regression."""

from __future__ import annotations

import numpy as np
import pytest

import bms


class TestFixes:
    def test_lstm_training_reproducible(self):
        # Same seed → identical training (fit() must use the instance RNG, not
        # the global np.random state).
        rng = np.random.default_rng(0)
        X = rng.normal(size=(6, 20, 3))
        Y = rng.uniform(0.2, 0.9, size=(6, 20))
        a = bms.LSTMEstimator(input_size=3, hidden_size=8, seed=7)
        b = bms.LSTMEstimator(input_size=3, hidden_size=8, seed=7)
        a.fit(X, Y, epochs=5, lr=1e-2)
        b.fit(X, Y, epochs=5, lr=1e-2)
        assert np.allclose(a.predict(X[0]), b.predict(X[0]))

    def test_short_circuit_rule_trips_on_overcurrent(self):
        det = bms.HybridFaultDetector()          # rule-only
        v = np.full(4, 3.7)
        T = np.full(4, 30.0)
        currents = np.array([0.0, 0.0, 40.0, 0.0])   # cell 2 over-current
        feats = bms.extract_features(v, currents, T, np.zeros(4), np.zeros(4))
        label, src = det.predict_step(feats, v, T, currents)
        assert (label, src) == ("short_circuit", "rule")
        # Without a current vector the short-circuit rule stays silent.
        label2, _ = det.predict_step(feats, v, T)
        assert label2 == "none"

    def test_supervisor_short_circuit_via_injector(self):
        det = bms.HybridFaultDetector()
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=5))
        thermal = bms.ThermalModel(n_cells=4)
        inj = bms.FaultInjector([bms.FaultSpec(bms.FaultMode.SHORT_CIRCUIT,
                                               start_step=0, cell_index=1)])
        sup = bms.BMSSupervisor(pack, thermal, det, injector=inj)
        for k in range(5):
            sup.step(1.0, 1.0, k=k)
        assert any(e["mode"] == "short_circuit" for e in sup.fault_log)
        assert sup.state == bms.BMSState.FAULT

    def test_parallel_circulating_currents_sum_to_group(self):
        # 1S2P with mismatched SOC → circulating currents even at I_group = 0.
        pack = bms.BatteryPack(bms.PackConfig(n_cells=1, n_parallel=2, seed=1))
        grp = pack.groups[0]
        grp.cells[0].reset(0.90)
        grp.cells[1].reset(0.60)
        e, g = grp._emf_and_conductance(pack.ocv_curve, 25.0)
        v_common = (np.dot(e, g) - 0.0) / g.sum()
        cell_I = (e - v_common) * g
        assert abs(cell_I.sum()) < 1e-9          # KCL: currents sum to I_group=0
        assert np.max(np.abs(cell_I)) > 1e-3     # but individual currents flow

    def test_heat_generation_symmetric_and_ohmic_floor(self):
        R0, ocv, I = np.array([0.02]), np.array([3.8]), 10.0
        # Consistent inputs (|OCV−V_t| ≥ R0·I): heat = |I·(OCV−V_t)|.
        q_dis = bms.ThermalModel.heat_generation(np.array([I]), R0,
                                                 np.array([3.5]), ocv)
        q_chg = bms.ThermalModel.heat_generation(np.array([-I]), R0,
                                                 np.array([4.1]), ocv)
        assert q_dis[0] == pytest.approx(abs(I * (3.8 - 3.5)))
        assert q_chg[0] == pytest.approx(abs(-I * (3.8 - 4.1)))
        assert q_chg[0] > 0.0                     # charge heat no longer dropped
        # Ohmic floor guards numerically-inconsistent inputs.
        q_floor = bms.ThermalModel.heat_generation(np.array([5.0]), R0,
                                                   np.array([3.79]), ocv)
        assert q_floor[0] == pytest.approx(5.0 ** 2 * 0.02)

    def test_peak_power_is_deliverable(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=2))
        thermal = bms.ThermalModel(n_cells=4)
        sup = bms.BMSSupervisor(pack, thermal, bms.HybridFaultDetector())
        out = sup.step(2.0, 1.0, k=0)
        assert np.isfinite(out["peak_power_W"]) and out["peak_power_W"] > 0.0
        # Peak capability must exceed the modest power actually drawn.
        assert out["peak_power_W"] >= out["power_W"]

    def test_thermal_stable_at_large_dt(self):
        tm = bms.ThermalModel(n_cells=8)
        heat = np.full(8, 5.0)
        T = tm.T
        for _ in range(50):
            T = tm.step(heat, cooling_duty=1.0, dt=60.0)   # dt ≫ CFL limit
        assert np.all(np.isfinite(T)) and np.all(T < 200.0)

    def test_ekf_reports_bounded_soc_without_internal_clip(self):
        params = bms.ECMParameters.for_nmc()
        ocv = bms.OCVSOC()
        ecm = bms.SecondOrderECM(params=params, ocv_curve=ocv)
        ecm.reset(0.8)
        current = np.full(300, 1.0)
        sim = ecm.simulate(current, dt=1.0)
        ekf = bms.EKFEstimator(params=params, ocv_curve=ocv)
        ekf.reset(0.8)
        est = ekf.run(current, sim["v_terminal"], dt=1.0)
        assert np.all(est >= 0.0) and np.all(est <= 1.0)     # reported SOC bounded
        assert abs(est[-1] - sim["soc"][-1]) < 0.05          # still tracks truth

    def test_passport_series_capacity_efc(self):
        # 4S1P pack: nominal pack capacity is cell capacity (~2.3 Ah), not 4*2.3.
        # Discharging 2.3 A for 1 hr (2.3 Ah) should accumulate ~1.0 EFC, not 0.25.
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, n_parallel=1, seed=1))
        thermal = bms.ThermalModel(n_cells=4)
        sup = bms.BMSSupervisor(pack, thermal, bms.HybridFaultDetector())
        cell_cap = float(np.mean(pack.capacities_Ah))
        assert sup.passport.nominal_capacity_Ah == pytest.approx(cell_cap, rel=1e-3)
        # Discharge cell_cap Ah
        sup.step(requested_pack_current_A=cell_cap, dt=3600.0, k=0)
        assert sup.passport.equivalent_full_cycles == pytest.approx(1.0, rel=1e-3)

    def test_switched_capacitor_bidirectional_efficiency(self):
        # When cell 0 has lower voltage than cell 1 (v[0] < v[1]), cell 1 is the donor.
        # Net charge into cell 0 must be attenuated by efficiency, never > donor loss.
        sc = bms.SwitchedCapacitorBalancer(efficiency=0.95)
        pack = bms.BatteryPack(bms.PackConfig(n_cells=2, seed=0))
        pack.cells[0].reset(0.40)
        pack.cells[1].reset(0.80)
        currents = sc.step(pack, dt=1.0)
        # currents[1] > 0 (leaves cell 1), currents[0] < 0 (arrives at cell 0)
        assert currents[1] > 0.0
        assert currents[0] < 0.0
        assert abs(currents[0]) == pytest.approx(0.95 * currents[1])

    def test_short_circuit_instant_trip(self):
        # Short circuit must trip contactor to FAULT on the very first step (k=0).
        det = bms.HybridFaultDetector()
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=0))
        thermal = bms.ThermalModel(n_cells=4)
        inj = bms.FaultInjector([bms.FaultSpec(bms.FaultMode.SHORT_CIRCUIT,
                                               start_step=0, cell_index=0)])
        sup = bms.BMSSupervisor(pack, thermal, det, injector=inj)
        sup.step(1.0, 1.0, k=0)
        assert sup.state == bms.BMSState.FAULT

    def test_undervoltage_rule_and_injection(self):
        det = bms.HybridFaultDetector()
        # Cell 0 dropped below 3.0 V (NMC v_min) but above 0.5 V (v_dropout)
        v = np.array([2.80, 3.7, 3.7, 3.7])
        T = np.full(4, 25.0)
        feats = bms.extract_features(v, np.zeros(4), T, np.zeros(4), np.zeros(4))
        label, src = det.predict_step(feats, v, T)
        assert (label, src) == ("undervoltage", "rule")

    def test_continuous_vs_pulse_crate_map(self):
        params = bms.ECMParameters.for_nmc()
        _, _, cmap_pulse = bms.compute_crate_map(params, chemistry="nmc", continuous=False)
        _, _, cmap_cont = bms.compute_crate_map(params, chemistry="nmc", continuous=True)
        # Continuous C-rate includes R1 + R2, so it must be strictly lower than pulse C-rate
        valid = (cmap_pulse > 0.0)
        assert np.all(cmap_cont[valid] < cmap_pulse[valid])



class TestRound2Fixes:
    def test_thermal_runaway_injector_trips_supervisor(self):
        det = bms.HybridFaultDetector()
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=7))
        thermal = bms.ThermalModel(n_cells=4)
        inj = bms.FaultInjector([bms.FaultSpec(bms.FaultMode.THERMAL_RUNAWAY,
                                               start_step=0, cell_index=2)])
        sup = bms.BMSSupervisor(pack, thermal, det, injector=inj)
        for k in range(3):
            sup.step(0.5, 1.0, k=k)
        assert sup.state == bms.BMSState.SHUTDOWN
        assert any(e["mode"] == "thermal_runaway" for e in sup.fault_log)

    def test_supervisor_state_of_power_available(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=8))
        thermal = bms.ThermalModel(n_cells=4)
        sup = bms.BMSSupervisor(pack, thermal, bms.HybridFaultDetector())
        limits = sup.state_of_power()
        assert set(limits) == {2.0, 10.0, 30.0}
        assert all(v.traction_power_W >= 0.0 for v in limits.values())

    def test_sop_default_voltages_track_pack_current(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=9))
        sop = bms.StateOfPower(bms.SOPConfig(max_discharge_current_A=1e4))
        limits = sop.calculate(pack, pack_current_A=5.0)
        assert limits[2.0].traction_current_A > 0.0
        assert np.isfinite(limits[2.0].traction_power_W)

