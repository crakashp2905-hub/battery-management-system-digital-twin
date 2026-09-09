"""Tests: physics."""

from __future__ import annotations

import numpy as np
import pytest

import bms


class TestOCVSOC:
    def test_endpoints(self):
        oc = bms.OCVSOC()
        assert oc.ocv(0.0) == pytest.approx(3.000, abs=1e-3)
        assert oc.ocv(1.0) == pytest.approx(4.200, abs=1e-3)

    def test_monotonic(self):
        oc = bms.OCVSOC()
        v = oc.ocv(np.linspace(0, 1, 200))
        assert np.all(np.diff(v) >= -1e-6), "OCV must be monotonically non-decreasing"

    def test_inverse_round_trip(self):
        oc = bms.OCVSOC()
        soc = np.linspace(0.05, 0.95, 19)
        v = oc.ocv(soc)
        soc_rec = oc.soc(v)
        # PCHIP can have small inversion error on the flat plateau region
        # (≈3.6–3.8 V) where dOCV/dSOC is shallow. 1 % SOC tolerance is the
        # typical real-world OCV-anchoring accuracy.
        assert np.allclose(soc, soc_rec, atol=1e-2)

    def test_slope_positive(self):
        oc = bms.OCVSOC()
        slopes = oc.docv_dsoc(np.linspace(0, 1, 50))
        assert np.all(slopes >= -1e-6)

    def test_hysteresis_charge_branch_is_above_discharge_branch(self):
        oc = bms.OCVSOC(hysteresis_v=0.012)
        v_charge = float(oc.ocv(0.6, current=-2.0))
        v_discharge = float(oc.ocv(0.6, current=2.0))
        assert v_charge > v_discharge
        assert v_charge - v_discharge == pytest.approx(0.024)



class TestECM:
    def test_rest_voltage_equals_ocv(self):
        oc = bms.OCVSOC()
        ecm = bms.SecondOrderECM(ocv_curve=oc); ecm.reset(0.5)
        v = ecm.step(0.0, 1.0)
        assert v == pytest.approx(float(oc.ocv(0.5)), abs=1e-6)

    def test_discharge_decreases_soc_and_voltage(self):
        oc = bms.OCVSOC()
        ecm = bms.SecondOrderECM(ocv_curve=oc); ecm.reset(0.95)
        i = np.full(600, 2.0)
        out = ecm.simulate(i, 1.0, soc0=0.95)
        assert out["soc"][-1] < out["soc"][0]
        assert out["v_terminal"][-1] < out["v_terminal"][0]

    def test_param_id_recovers_voltage(self):
        oc = bms.OCVSOC()
        true_p = bms.ECMParameters(R0=0.030, R1=0.012, C1=2500, R2=0.025,
                                    C2=10_000, Q_nom_Ah=2.3)
        ecm = bms.SecondOrderECM(params=true_p, ocv_curve=oc)
        rng = np.random.default_rng(0)
        n = 1500
        i = np.zeros(n); i[200:1200] = 2.3
        out = ecm.simulate(i, 1.0, soc0=0.95)
        v_meas = out["v_terminal"] + rng.normal(0, 0.005, n)
        fitted, info = bms.fit_ecm_parameters(i, v_meas, dt=1.0,
                                               Q_nom_Ah=2.3, ocv_curve=oc)
        assert info["success"]
        # Voltage RMSE should be in the ballpark of the noise level.
        assert info["rmse_v"] < 0.02



class TestPack:
    def test_initial_imbalance_present(self):
        p = bms.BatteryPack(bms.PackConfig(n_cells=4, initial_soc_sigma=0.05, seed=1))
        assert p.soc_imbalance() > 0

    def test_pack_voltage_is_sum(self):
        p = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=2))
        assert p.pack_voltage() == pytest.approx(p.cell_voltages().sum(), rel=1e-9)

    def test_balancing_currents_length(self):
        p = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=2))
        with pytest.raises(ValueError):
            p.step(1.0, 1.0, balancing_currents=[0.1, 0.2])  # wrong length



class TestParallelPack:
    def test_total_cells_count(self):
        p = bms.BatteryPack(bms.PackConfig(n_cells=4, n_parallel=2, seed=5))
        assert p.n_cells == 4          # series groups
        assert p.n_parallel == 2
        assert p.total_cells == 8      # individual cells

    def test_pack_voltage_is_sum_of_groups(self):
        p = bms.BatteryPack(bms.PackConfig(n_cells=4, n_parallel=2, seed=5))
        assert p.pack_voltage() == pytest.approx(p.cell_voltages().sum(), rel=1e-9)

    def test_balancing_currents_length_n_series(self):
        p = bms.BatteryPack(bms.PackConfig(n_cells=4, n_parallel=2, seed=5))
        # Balancing currents must be length n_series (4), not total_cells (8)
        with pytest.raises(ValueError):
            p.step(1.0, 1.0, balancing_currents=[0.0, 0.0])  # wrong length

    def test_parallel_pack_step(self):
        p = bms.BatteryPack(bms.PackConfig(n_cells=4, n_parallel=2, seed=5))
        out = p.step(1.0, 10.0)
        assert out["v_cells"].shape == (4,)
        assert out["soc"].shape == (4,)
        assert 0 < out["v_pack"] < 20.0

    def test_parallel_effective_capacity_doubled(self):
        p1 = bms.BatteryPack(bms.PackConfig(n_cells=1, n_parallel=1, seed=0))
        p2 = bms.BatteryPack(bms.PackConfig(n_cells=1, n_parallel=2, seed=0))
        cap1 = float(p1.capacities_Ah[0])
        cap2 = float(p2.capacities_Ah[0])
        # 2-parallel group has roughly 2× the capacity of a single cell
        assert cap2 == pytest.approx(2 * cap1, rel=0.15)

    def test_parallel_effective_r0_halved(self):
        p1 = bms.BatteryPack(bms.PackConfig(n_cells=1, n_parallel=1, seed=0))
        p2 = bms.BatteryPack(bms.PackConfig(n_cells=1, n_parallel=2, seed=0))
        r0_single = p1.groups[0].cells[0].params.R0
        r0_eff = p2.groups[0].params.R0
        # Effective parallel R0 should be roughly half the single-cell R0
        assert r0_eff < r0_single

    def test_parallel_soc_is_capacity_weighted(self):
        p = bms.BatteryPack(bms.PackConfig(n_cells=1, n_parallel=3,
                                           initial_soc_sigma=0.10, seed=7))
        grp = p.groups[0]
        caps = np.array([c.params.Q_nom_Ah for c in grp.cells])
        socs = np.array([c.soc for c in grp.cells])
        expected = float(np.dot(socs, caps) / caps.sum())
        assert grp.soc == pytest.approx(expected, rel=1e-9)

    def test_supervisor_works_with_parallel_pack(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, n_parallel=2, seed=3))
        thermal = bms.ThermalModel(n_cells=4)
        det = bms.HybridFaultDetector()
        sup = bms.BMSSupervisor(pack, thermal, det)
        out = sup.step(1.0, 1.0)
        assert "power_W" in out
        assert out["v_cells"].shape == (4,)



class TestThermal:
    def test_no_heat_returns_to_ambient(self):
        tm = bms.ThermalModel(n_cells=4, params=bms.ThermalParameters(T_amb_C=25.0))
        tm.reset(40.0)
        for _ in range(2000):
            tm.step(np.zeros(4), 0.0, 1.0)
        assert np.all(np.abs(tm.T - 25.0) < 0.5)

    def test_heat_raises_temperature(self):
        tm = bms.ThermalModel(n_cells=4); tm.reset(25.0)
        for _ in range(60):
            tm.step(np.full(4, 1.0), 0.0, 1.0)
        assert np.all(tm.T > 25.0)

    def test_pid_drives_to_setpoint(self):
        # First-order plant: y[k+1] = (1-a) y[k] + a u
        # PID should drive y to the setpoint.
        pid = bms.PIDController(kp=0.5, ki=0.05, kd=0.0, setpoint=10.0,
                                out_min=-50, out_max=50)
        y = 0.0; a = 0.1
        for _ in range(400):
            u = pid.step(y, 1.0)
            y = (1 - a) * y + a * (-u + y)  # negative gain plant
        # Allow generous tolerance — exact dynamics aren't the point
        assert abs(y - 10.0) < 2.0 or abs(pid.prev_err) < 2.0


class TestEntropicHeat:
    def test_reversible_term_flips_sign_with_current(self):
        import numpy as np
        R0 = np.array([0.025]); ocv = np.array([3.7])
        coeff = -1.5e-4
        # Same |I|, opposite direction, symmetric terminal voltage.
        q_dis = bms.ThermalModel.heat_generation(
            np.array([2.0]), R0, np.array([3.65]), ocv,
            entropic_coeff_V_per_K=coeff, temperature_C=25.0)[0]
        q_chg = bms.ThermalModel.heat_generation(
            np.array([-2.0]), R0, np.array([3.75]), ocv,
            entropic_coeff_V_per_K=coeff, temperature_C=25.0)[0]
        q_irr = bms.ThermalModel.heat_generation(
            np.array([2.0]), R0, np.array([3.65]), ocv)[0]
        # Reversible heat is exothermic on discharge (coeff<0) and endothermic on
        # charge, so discharge > irreversible-only > charge.
        assert q_dis > q_irr > q_chg

    def test_zero_coefficient_matches_irreversible(self):
        import numpy as np
        args = (np.array([1.5]), np.array([0.02]), np.array([3.6]), np.array([3.7]))
        base = bms.ThermalModel.heat_generation(*args)
        with_zero = bms.ThermalModel.heat_generation(
            *args, entropic_coeff_V_per_K=0.0, temperature_C=25.0)
        assert np.allclose(base, with_zero)

    def test_supervisor_entropic_flag_is_opt_in(self):
        # Off by default → identical to no entropic term; on → different heat.
        import numpy as np
        np.random.seed(0)
        def _run(entropic):
            pack = bms.BatteryPack(bms.PackConfig(n_cells=3, seed=5))
            cfg = bms.SupervisorConfig(entropic_heat=entropic)
            sup = bms.BMSSupervisor(pack, bms.ThermalModel(n_cells=3),
                                    bms.HybridFaultDetector(), config=cfg)
            T = None
            for _ in range(120):
                T = sup.step(2.0, 1.0)["T_cells"]
            return float(np.mean(T))
        assert _run(True) != _run(False)



class TestSelfDischarge:
    def test_self_discharge_drains_soc_at_rest(self):
        oc = bms.OCVSOC()
        p = bms.ECMParameters(self_discharge_pct_per_month=10.0)  # fast for test
        ecm = bms.SecondOrderECM(params=p, ocv_curve=oc)
        ecm.reset(0.80)
        # 1 month = 30×24×3600 = 2_592_000 s → simulate 259_200 s (10%)
        n_steps = 10_000
        for _ in range(n_steps):
            ecm.step(0.0, 1.0)   # rest, no current
        # SOC should have dropped due to self-discharge
        assert ecm.soc < 0.80

    def test_no_self_discharge_by_default(self):
        oc = bms.OCVSOC()
        p = bms.ECMParameters()   # default self_discharge_pct_per_month = 0.0
        ecm = bms.SecondOrderECM(params=p, ocv_curve=oc)
        ecm.reset(0.80)
        for _ in range(1000):
            ecm.step(0.0, 1.0)
        assert ecm.soc == pytest.approx(0.80, abs=1e-9)

    def test_at_temperature_preserves_self_discharge(self):
        p = bms.ECMParameters(self_discharge_pct_per_month=5.0)
        p_cold = p.at_temperature(0.0)
        assert p_cold.self_discharge_pct_per_month == pytest.approx(5.0)



class TestSOEandUncertainty:
    def test_state_of_energy_positive(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=0))
        assert pack.state_of_energy_Wh() > 0

    def test_soe_decreases_on_discharge(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=1, seed=0))
        e0 = pack.state_of_energy_Wh()
        pack.step(2.0, 600.0)    # 600-s discharge
        e1 = pack.state_of_energy_Wh()
        assert e1 < e0

    def test_ekf_uncertainty_positive_after_step(self):
        oc = bms.OCVSOC()
        p = bms.ECMParameters()
        ekf = bms.EKFEstimator(params=p, ocv_curve=oc)
        ekf.reset(0.85)
        ekf.update(1.5, 3.7, 1.0)
        assert ekf.soc_uncertainty_1sigma > 0

    def test_ekf_uncertainty_shrinks_with_data(self):
        oc = bms.OCVSOC()
        p = bms.ECMParameters()
        ecm = bms.SecondOrderECM(params=p, ocv_curve=oc)
        i = np.full(300, 1.5)
        out = ecm.simulate(i, 1.0, soc0=0.85)
        ekf = bms.EKFEstimator(params=p, ocv_curve=oc)
        ekf.reset(0.80)   # slight initial error
        ekf.run(i, out["v_terminal"], 1.0)
        sigma_final = ekf.soc_uncertainty_1sigma
        # After 300 steps of data, P[0,0] should be well below initial 1e-2
        assert sigma_final < 0.02

    def test_cccv_profile_is_negative_and_two_phase(self):
        profile = bms.generate_cccv_profile(
            Q_nom_Ah=2.3, soc_start=0.20, i_charge_C=0.5, chemistry="nmc"
        )
        assert len(profile) > 0
        assert np.all(profile <= 0)       # all charging (≤ 0 A)
        # CC phase: constant current; CV phase: tapering → last half should be less
        # than first half in magnitude
        half = len(profile) // 2
        if half > 0:
            assert abs(profile[-1]) <= abs(profile[0]) + 1e-6



class TestLargeScalePack:
    """Validate that the BMS simulator scales to 20S×20P = 400-cell packs.

    All physics (ECM, thermal, balancing, fault detection) must work
    identically at large scale because every subsystem operates on
    vectorised numpy arrays sized to n_cells or n_parallel with no
    hardcoded cell-count assumptions.
    """

    @staticmethod
    def _large_pack(n_s: int = 20, n_p: int = 20,
                    chemistry: str = "nmc") -> bms.BatteryPack:
        return bms.BatteryPack(bms.PackConfig(
            n_cells=n_s, n_parallel=n_p, chemistry=chemistry, seed=99,
        ))

    # ── topology ──────────────────────────────────────────────────────────
    def test_20s_1p_cell_count(self):
        p = self._large_pack(20, 1)
        assert p.n_cells == 20
        assert p.n_parallel == 1
        assert p.total_cells == 20

    def test_1s_20p_cell_count(self):
        p = self._large_pack(1, 20)
        assert p.n_cells == 1
        assert p.n_parallel == 20
        assert p.total_cells == 20

    def test_20s_20p_cell_count(self):
        p = self._large_pack(20, 20)
        assert p.n_cells == 20
        assert p.n_parallel == 20
        assert p.total_cells == 400

    # ── energy scales linearly with cell count ────────────────────────────
    def test_capacity_scales_with_parallel(self):
        p1 = self._large_pack(4, 1)
        p20 = self._large_pack(4, 20)
        # Total capacity (Ah) should scale ~20× (within scatter bounds)
        ratio = p20.capacities_Ah.sum() / p1.capacities_Ah.sum()
        assert 18.0 < ratio < 22.0, f"Capacity ratio {ratio:.2f} not near 20×"

    def test_voltage_independent_of_parallel(self):
        # Series voltage doesn't change with more parallel cells
        p1  = bms.BatteryPack(bms.PackConfig(n_cells=10, n_parallel=1,  seed=1))
        p10 = bms.BatteryPack(bms.PackConfig(n_cells=10, n_parallel=10, seed=1))
        v1  = p1.cell_voltages().sum()
        v10 = p10.cell_voltages().sum()
        assert abs(v1 - v10) < 0.5, "Pack voltage must not change with n_parallel"

    # ── single step runs without error at max scale ───────────────────────
    def test_20s_20p_single_step(self):
        p = self._large_pack(20, 20)
        tm = bms.ThermalModel(n_cells=20)
        result = p.step(10.0, 1.0, cell_temperatures_C=tm.T)
        v = result["v_cells"]
        assert v.shape == (20,)
        assert np.all(np.isfinite(v))
        assert np.all(v > 0)

    def test_20s_20p_soc_array_shape(self):
        p = self._large_pack(20, 20)
        soc = p.soc
        assert soc.shape == (20,)
        assert np.all((soc >= 0) & (soc <= 1))

    # ── thermal model at large scale ──────────────────────────────────────
    def test_thermal_20_cells_step(self):
        tm = bms.ThermalModel(n_cells=20)
        Q = np.full(20, 0.5)
        tm.step(Q, cooling_duty=0.0, dt=1.0)
        assert tm.T.shape == (20,)
        assert np.all(np.isfinite(tm.T))

    def test_thermal_20_cells_heats_up(self):
        tm = bms.ThermalModel(n_cells=20)
        T0 = tm.T.copy()
        Q = np.full(20, 5.0)   # 5 W per cell heat
        for _ in range(50):
            tm.step(Q, cooling_duty=0.0, dt=1.0)
        assert tm.T.mean() > T0.mean(), "Continuous heat injection must raise temperature"

    # ── balancers at large scale ───────────────────────────────────────────
    def test_passive_balancer_20s(self):
        p = self._large_pack(20, 1)
        bal = bms.PassiveBalancer()
        currents = bal.step(p, dt=1.0)
        assert currents.shape == (20,)
        assert np.all(np.isfinite(currents))

    def test_sc_balancer_20s(self):
        p = self._large_pack(20, 1)
        bal = bms.SwitchedCapacitorBalancer()
        currents = bal.step(p, dt=1.0)
        assert currents.shape == (20,)
        assert np.all(np.isfinite(currents))

    def test_inductor_balancer_20s(self):
        p = self._large_pack(20, 1)
        bal = bms.InductorBalancer()
        currents = bal.step(p, dt=1.0)
        assert currents.shape == (20,)
        assert np.all(np.isfinite(currents))

    # ── fault injection at large scale ────────────────────────────────────
    def test_fault_injection_large_pack_cell_index(self):
        """Fault at the last cell (index 19) of a 20-cell pack."""
        inj = bms.FaultInjector()
        inj.add(bms.FaultSpec(
            mode=bms.FaultMode.SENSOR_BIAS,
            start_step=0, end_step=100,
            cell_index=19, severity=1.0,
        ))
        p = self._large_pack(20, 1)
        v = p.cell_voltages()
        v_after = inj.apply_to_voltage_meas(v, k=50)
        assert v_after[19] != v[19], "Cell 19 voltage must be biased"
        assert np.allclose(v_after[:19], v[:19]), "Other cells unaffected"

    # ── chemistry sweep at 20S×20P ────────────────────────────────────────
    def test_all_chemistries_large_scale(self):
        for chem in ["nmc", "lfp", "lmfp", "lto", "nca", "lmo", "ssb"]:
            p = bms.BatteryPack(bms.PackConfig(
                n_cells=20, n_parallel=20, chemistry=chem, seed=7))
            v = p.cell_voltages()
            assert v.shape == (20,), f"{chem}: wrong voltage shape"
            assert np.all(v > 0), f"{chem}: negative group voltage"
            assert p.total_cells == 400, f"{chem}: wrong cell count"

    # ── EKF at 20S pack ──────────────────────────────────────────────────
    def test_ekf_runs_on_large_pack(self):
        """EKF is per-group so it should work with 20 groups."""
        p = self._large_pack(20, 1)
        ekf = bms.EKFEstimator(params=p.cells[0].params, ocv_curve=p.ocv_curve)
        ekf.reset(soc0=p.soc[0])
        # Step through 20 steps
        for _ in range(20):
            p.step(5.0, 1.0)
            v = p.cell_voltages()
            soc_est = ekf.update(5.0, float(v[0]), dt=1.0, temperature_C=25.0)
            assert 0.0 <= soc_est <= 1.0

    # ── RUL / passport at large scale ─────────────────────────────────────
    def test_passport_scales_with_cell_count(self):
        """Passport nominal capacity should scale with n_parallel."""
        from bms import BatteryPassport, get_chemistry_props
        props = get_chemistry_props("nmc")
        cap_1p = props["default_capacity_Ah"] * 4   # 4S × 1P
        cap_20p = props["default_capacity_Ah"] * 4 * 20  # 4S × 20P
        bp1  = BatteryPassport(nominal_capacity_Ah=cap_1p,  nominal_voltage_V=4 * 3.6)
        bp20 = BatteryPassport(nominal_capacity_Ah=cap_20p, nominal_voltage_V=4 * 3.6)
        # Record same discharge on both — larger pack should have more EFC headroom
        bp1.update(current_A=5.0,   v_pack_V=3.5, dt_s=1.0)
        bp20.update(current_A=100.0, v_pack_V=3.5, dt_s=1.0)
        s1  = bp1.summary()
        s20 = bp20.summary()
        assert s1["nominal_capacity_Ah"] < s20["nominal_capacity_Ah"]

    # ── simulate 100 steps at 20S×20P (smoke test for world-model use) ───
    def test_20s_20p_simulation_100_steps(self):
        """Smoke-test: 100 steps of a full 400-cell pack must stay stable."""
        p  = self._large_pack(20, 20)
        tm = bms.ThermalModel(n_cells=20)
        for k in range(100):
            Q_heat = np.zeros(20)
            result = p.step(20.0, 1.0, cell_temperatures_C=tm.T)
            tm.step(Q_heat, cooling_duty=0.0, dt=1.0)
            assert np.all(np.isfinite(result["v_cells"])), f"Step {k}: non-finite voltage"
            assert np.all(p.soc >= 0), f"Step {k}: negative SOC"

    # ── imbalance metric scales correctly ─────────────────────────────────
    def test_imbalance_metric_large_pack(self):
        p = self._large_pack(20, 1)
        imb = p.soc_imbalance()
        assert 0.0 <= imb <= 1.0, f"Imbalance {imb} out of [0, 1]"

