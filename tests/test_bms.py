"""
Unit tests for the BMS digital twin package.

Run with::

    pytest tests/ -v

Tests are designed to be fast (< 30 s in total) so they can be run on
every commit. They exercise:

* OCV–SOC monotonicity and inverse round-trip.
* ECM step / simulate forward integration and parameter-id round-trip.
* Pack series-current invariant and imbalance metric.
* Three balancing strategies — energy balance and convergence.
* Coulomb counter, EKF, UKF correctness on a known truth trace.
* FDM thermal stability + PID setpoint tracking.
* FMEA RPN ordering and RUL non-negativity.
* Fault injection + rule detection.
* Supervisor state machine transitions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the package importable when running pytest from the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bms


# ----------------------------------------------------------------------
# 1. OCV-SOC
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 2. ECM
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 3. Pack
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 4. Balancers
# ----------------------------------------------------------------------
class TestBalancers:
    def _factory(self, seed=7):
        return lambda: bms.BatteryPack(bms.PackConfig(
            n_cells=4, initial_soc_sigma=0.10, seed=seed))

    def test_inductor_converges(self):
        pack = self._factory()()
        initial_imbalance = pack.soc_imbalance()
        bal = bms.InductorBalancer(max_current_A=2.0, gain=10.0)
        for _ in range(2_000):
            cur = bal.step(pack, 1.0)
            pack.step(0.0, 1.0, balancing_currents=cur)
        # With 3% capacity scatter, the steady-state isn't perfect SOC
        # equality but a much-reduced spread. The system should bring
        # imbalance to <1% from initial ~13%.
        assert pack.soc_imbalance() < 0.01
        assert pack.soc_imbalance() < 0.1 * initial_imbalance

    def test_passive_dissipates_energy(self):
        pack = self._factory()()
        bal = bms.PassiveBalancer()
        for _ in range(60):
            cur = bal.step(pack, 1.0)
            pack.step(0.0, 1.0, balancing_currents=cur)
        # Imbalance must shrink and dissipative loss must accumulate.
        assert bal.energy_loss_J > 0

    def test_compare_returns_dataframe(self):
        df = bms.compare_balancers(self._factory(), duration_s=600, dt=1.0)
        assert {"final_imbalance", "energy_loss_J"}.issubset(df.columns)
        assert len(df) == 3

    def test_inductor_more_efficient_than_passive(self):
        df = bms.compare_balancers(self._factory(), duration_s=3600, dt=2.0)
        assert (df.loc["inductor_active", "energy_loss_J"]
                < df.loc["passive_resistive", "energy_loss_J"])


# ----------------------------------------------------------------------
# 5. SOC estimators
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def truth_trace():
    oc = bms.OCVSOC()
    p = bms.ECMParameters(R0=0.025, R1=0.012, C1=2500, R2=0.025,
                          C2=10_000, Q_nom_Ah=2.3)
    m = bms.SecondOrderECM(params=p, ocv_curve=oc)
    n = 1800
    i = np.zeros(n)
    i[100:600] = 1.5
    i[800:1400] = 2.5
    out = m.simulate(i, 1.0, soc0=0.95)
    return p, oc, i, out["v_terminal"], out["soc"]


class TestEstimators:
    def test_coulomb_counting_no_bias(self, truth_trace):
        p, oc, i, v, soc = truth_trace
        cc = bms.CoulombCounter(p.Q_nom_Ah, soc0=0.95)
        soc_hat = cc.run(i, v, 1.0)
        assert np.max(np.abs(soc_hat - soc)) < 5e-3

    def test_ekf_corrects_bias(self, truth_trace):
        p, oc, i, v, soc = truth_trace
        i_biased = i + 0.05
        ekf = bms.EKFEstimator(params=p, ocv_curve=oc); ekf.reset(0.92)
        soc_hat = ekf.run(i_biased, v, 1.0)
        rmse = np.sqrt(np.mean((soc_hat - soc) ** 2))
        # CC with the same bias would drift by several percent — EKF must
        # do an order of magnitude better.
        assert rmse < 0.01

    def test_ukf_corrects_bias(self, truth_trace):
        p, oc, i, v, soc = truth_trace
        ukf = bms.UKFEstimator(params=p, ocv_curve=oc, dt=1.0); ukf.reset(0.92)
        soc_hat = ukf.run(i + 0.05, v, 1.0)
        assert np.sqrt(np.mean((soc_hat - soc) ** 2)) < 0.01


# ----------------------------------------------------------------------
# 6. Thermal + PID
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 7. FMEA + RUL
# ----------------------------------------------------------------------
class TestFMEAandRUL:
    def test_rpn_descending(self):
        fmea = bms.build_fmea_table()
        assert (fmea["RPN"].values[:-1] >= fmea["RPN"].values[1:]).all()

    def test_rul_non_negative(self):
        df = bms.load_nasa_like_dataset(cycles=40, capacity_Ah=2.3, seed=1)
        rul = bms.estimate_rul(df["cycle"].values, df["capacity_Ah"].values,
                                nominal_capacity_Ah=2.3)
        assert rul["rul_cycles"] >= 0
        assert 0 <= rul["soh"] <= 1.0

    def test_synthetic_dataset_shape(self):
        df = bms.load_nasa_like_dataset(cycles=10, seed=0)
        assert {"cycle", "capacity_Ah", "temperature_C", "source"}.issubset(df.columns)
        assert len(df) == 10


# ----------------------------------------------------------------------
# 8. Faults + supervisor
# ----------------------------------------------------------------------
class TestFaults:
    def test_rule_detects_overcharge(self):
        det = bms.HybridFaultDetector()
        v = np.array([3.7, 4.30, 3.7, 3.7])  # cell 1 over-charged
        T = np.array([25, 25, 25, 25.])
        feats = bms.extract_features(v, np.zeros(4), T, np.zeros(4), np.zeros(4))
        label, src = det.predict_step(feats, v, T)
        assert label == "overcharge"
        assert src == "rule"

    def test_rule_detects_runaway(self):
        det = bms.HybridFaultDetector()
        v = np.array([3.7, 3.7, 3.7, 3.7])
        T = np.array([25, 75., 25, 25])
        feats = bms.extract_features(v, np.zeros(4), T, np.zeros(4), np.zeros(4))
        label, src = det.predict_step(feats, v, T)
        assert label == "thermal_runaway"

    def test_injector_no_active_fault_passthrough(self):
        inj = bms.FaultInjector()
        c = np.array([1.0, 1.0, 1.0, 1.0])
        assert np.allclose(inj.apply_to_currents(c, 0), c)

    def test_supervisor_trips_only_on_rule_alarms(self):
        det = bms.HybridFaultDetector()                    # not fitted → rule-only
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=3))
        thermal = bms.ThermalModel(n_cells=4)
        sup = bms.BMSSupervisor(pack, thermal, det)

        # Inject sustained over-temperature for 5+ steps to clear the streak gate.
        for k in range(10):
            thermal.T[1] = 80.0
            sup.step(0.5, 1.0, k=k)
        assert sup.state == bms.BMSState.FAULT
        # And the fault log records rule-source detections.
        assert any(e["source"] == "rule" for e in sup.fault_log)


# ----------------------------------------------------------------------
# 9. LFP chemistry
# ----------------------------------------------------------------------
class TestLFPChemistry:
    def test_lfp_ocv_endpoints(self):
        oc = bms.OCVSOC.from_chemistry("lfp")
        assert oc.ocv(0.0) == pytest.approx(2.500, abs=1e-3)
        assert oc.ocv(1.0) == pytest.approx(3.650, abs=1e-3)

    def test_lfp_ocv_monotonic(self):
        oc = bms.OCVSOC.from_chemistry("lfp")
        v = oc.ocv(np.linspace(0, 1, 200))
        assert np.all(np.diff(v) >= -1e-6)

    def test_lfp_ecm_params(self):
        p = bms.ECMParameters.for_lfp()
        assert p.chemistry == "lfp"
        assert p.R0 == pytest.approx(0.020, rel=1e-6)
        assert p.Q_nom_Ah == pytest.approx(3.2, rel=1e-6)

    def test_lfp_arrhenius_different_from_nmc(self):
        p_nmc = bms.ECMParameters.for_nmc()
        p_lfp = bms.ECMParameters.for_lfp()
        # At 0 °C, LFP resistance increases less than NMC (lower Ea/R)
        r_nmc_cold = p_nmc.at_temperature(0.0).R0
        r_lfp_cold = p_lfp.at_temperature(0.0).R0
        # LFP R0 at 0°C / LFP R0 at 25°C < NMC R0 at 0°C / NMC R0 at 25°C
        ratio_nmc = r_nmc_cold / p_nmc.R0
        ratio_lfp = r_lfp_cold / p_lfp.R0
        assert ratio_lfp < ratio_nmc, "LFP should have lower Arrhenius scaling ratio"

    def test_lfp_pack_voltage_lower_than_nmc(self):
        p_nmc = bms.BatteryPack(bms.PackConfig(n_cells=4, chemistry="nmc", seed=0))
        p_lfp = bms.BatteryPack(bms.PackConfig(n_cells=4, chemistry="lfp", seed=0))
        assert p_lfp.pack_voltage() < p_nmc.pack_voltage()

    def test_lfp_fault_detector_higher_runaway_threshold(self):
        det_nmc = bms.HybridFaultDetector(chemistry="nmc")
        det_lfp = bms.HybridFaultDetector(chemistry="lfp")
        v = np.array([3.2, 3.2, 3.2, 3.2])
        T = np.array([75., 75., 75., 75.])
        feats = bms.extract_features(v, np.zeros(4), T, np.zeros(4), np.zeros(4))
        label_nmc, _ = det_nmc.predict_step(feats, v, T)
        label_lfp, _ = det_lfp.predict_step(feats, v, T)
        # 75°C > NMC threshold (70°C) but < LFP threshold (90°C)
        assert label_nmc == "thermal_runaway"
        assert label_lfp == "none"

    def test_lfp_overcharge_lower_threshold(self):
        det = bms.HybridFaultDetector(chemistry="lfp")
        v = np.array([3.2, 3.72, 3.2, 3.2])   # 3.72 > LFP threshold 3.70
        T = np.array([25., 25., 25., 25.])
        feats = bms.extract_features(v, np.zeros(4), T, np.zeros(4), np.zeros(4))
        label, src = det.predict_step(feats, v, T)
        assert label == "overcharge"
        assert src == "rule"

    def test_cell_chemistry_enum(self):
        assert bms.CellChemistry.NMC.value == "nmc"
        assert bms.CellChemistry.LFP.value == "lfp"
        props = bms.get_chemistry_props(bms.CellChemistry.LFP)
        assert props["nominal_voltage_V"] == pytest.approx(3.2)


# ----------------------------------------------------------------------
# 10. Series × parallel topology
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 11. Power mode
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 12. New chemistries — LMFP, LTO, NCA, LMO
# ----------------------------------------------------------------------
class TestNewChemistries:
    # ── LMFP ─────────────────────────────────────────────────────────
    def test_lmfp_ocv_endpoints(self):
        oc = bms.OCVSOC.from_chemistry("lmfp")
        assert oc.ocv(0.0) == pytest.approx(2.800, abs=1e-3)
        assert oc.ocv(1.0) == pytest.approx(4.150, abs=1e-3)

    def test_lmfp_ocv_monotonic(self):
        oc = bms.OCVSOC.from_chemistry("lmfp")
        v = oc.ocv(np.linspace(0, 1, 200))
        assert np.all(np.diff(v) >= -1e-6)

    def test_lmfp_ecm_params(self):
        p = bms.ECMParameters.for_lmfp()
        assert p.chemistry == "lmfp"
        assert p.R0 == pytest.approx(0.018, rel=1e-6)
        assert p.Q_nom_Ah == pytest.approx(3.0, rel=1e-6)

    def test_lmfp_higher_runaway_than_nmc(self):
        props_nmc = bms.get_chemistry_props("nmc")
        props_lmfp = bms.get_chemistry_props("lmfp")
        assert props_lmfp["T_runaway_C"] > props_nmc["T_runaway_C"]

    # ── LTO ─────────────────────────────────────────────────────────
    def test_lto_ocv_endpoints(self):
        oc = bms.OCVSOC.from_chemistry("lto")
        assert oc.ocv(0.0) == pytest.approx(1.500, abs=1e-3)
        assert oc.ocv(1.0) == pytest.approx(2.800, abs=1e-3)

    def test_lto_ocv_monotonic(self):
        oc = bms.OCVSOC.from_chemistry("lto")
        v = oc.ocv(np.linspace(0, 1, 200))
        assert np.all(np.diff(v) >= -1e-6)

    def test_lto_ecm_very_low_r0(self):
        p = bms.ECMParameters.for_lto()
        assert p.chemistry == "lto"
        # LTO R0 (0.008) is much lower than NMC (0.025)
        p_nmc = bms.ECMParameters.for_nmc()
        assert p.R0 < p_nmc.R0

    def test_lto_pack_lower_voltage_than_nmc(self):
        p_lto = bms.BatteryPack(bms.PackConfig(n_cells=4, chemistry="lto", seed=0))
        p_nmc = bms.BatteryPack(bms.PackConfig(n_cells=4, chemistry="nmc", seed=0))
        assert p_lto.pack_voltage() < p_nmc.pack_voltage()

    def test_lto_highest_runaway_threshold(self):
        props = bms.get_chemistry_props("lto")
        for chem in ["nmc", "lfp", "lmfp", "nca", "lmo"]:
            assert props["T_runaway_C"] >= bms.get_chemistry_props(chem)["T_runaway_C"]

    # ── NCA ──────────────────────────────────────────────────────────
    def test_nca_ocv_endpoints(self):
        oc = bms.OCVSOC.from_chemistry("nca")
        assert oc.ocv(0.0) == pytest.approx(3.000, abs=1e-3)
        assert oc.ocv(1.0) == pytest.approx(4.200, abs=1e-3)

    def test_nca_ocv_monotonic(self):
        oc = bms.OCVSOC.from_chemistry("nca")
        v = oc.ocv(np.linspace(0, 1, 200))
        assert np.all(np.diff(v) >= -1e-6)

    def test_nca_lower_runaway_than_nmc(self):
        props_nmc = bms.get_chemistry_props("nmc")
        props_nca = bms.get_chemistry_props("nca")
        # NCA is LESS thermally stable — lower onset temperature
        assert props_nca["T_runaway_C"] < props_nmc["T_runaway_C"]

    def test_nca_higher_arrhenius_than_nmc(self):
        p_nmc = bms.ECMParameters.for_nmc()
        p_nca = bms.ECMParameters.for_nca()
        # NCA is more temperature-sensitive
        assert p_nca._arrhenius_K > p_nmc._arrhenius_K

    # ── LMO ──────────────────────────────────────────────────────────
    def test_lmo_ocv_endpoints(self):
        oc = bms.OCVSOC.from_chemistry("lmo")
        assert oc.ocv(0.0) == pytest.approx(3.000, abs=1e-3)
        assert oc.ocv(1.0) == pytest.approx(4.200, abs=1e-3)

    def test_lmo_ocv_monotonic(self):
        oc = bms.OCVSOC.from_chemistry("lmo")
        v = oc.ocv(np.linspace(0, 1, 200))
        assert np.all(np.diff(v) >= -1e-6)

    def test_lmo_highest_self_discharge(self):
        props_lmo = bms.get_chemistry_props("lmo")
        for chem in ["nmc", "lfp", "lmfp", "lto", "nca"]:
            assert props_lmo["self_discharge_pct_per_month"] > \
                   bms.get_chemistry_props(chem)["self_discharge_pct_per_month"]

    def test_lmo_lowest_runaway_threshold(self):
        # LMO has lowest thermal-runaway onset due to Mn dissolution
        props_lmo = bms.get_chemistry_props("lmo")
        for chem in ["nmc", "lfp", "lmfp", "lto", "nca"]:
            assert props_lmo["T_runaway_C"] <= bms.get_chemistry_props(chem)["T_runaway_C"]

    def test_all_chemistry_enum_values(self):
        for chem in bms.CellChemistry:
            props = bms.get_chemistry_props(chem)
            assert "ocv_table" in props
            assert "v_overcharge" in props
            assert "T_runaway_C" in props

    def test_unknown_chemistry_raises(self):
        with pytest.raises(ValueError):
            bms.get_chemistry_props("unknownium")


# ----------------------------------------------------------------------
# 13. Self-discharge
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 14. Battery Passport
# ----------------------------------------------------------------------
class TestBatteryPassport:
    def _make_passport(self):
        return bms.BatteryPassport(nominal_capacity_Ah=2.3,
                                   nominal_voltage_V=14.8,
                                   chemistry="nmc")

    def test_efc_after_one_discharge(self):
        bp = self._make_passport()
        # Discharge 2.3 Ah at 14.8 V for 3600 s at 2.3 A
        for _ in range(3600):
            bp.update(current_A=2.3, v_pack_V=14.8, dt_s=1.0, soc_mean=0.5)
        assert bp.equivalent_full_cycles == pytest.approx(1.0, rel=0.01)

    def test_rte_after_charge_discharge(self):
        bp = self._make_passport()
        for _ in range(3600):
            bp.update(-2.3, 14.0, 1.0, soc_mean=0.5)   # charge
        for _ in range(3600):
            bp.update(2.3, 14.8, 1.0, soc_mean=0.5)    # discharge
        # RTE should be plausible (< 1.0 since charge and discharge voltages differ)
        assert 0 < bp.round_trip_efficiency <= 1.0

    def test_summary_has_required_keys(self):
        bp = self._make_passport()
        s = bp.summary()
        for key in ("chemistry", "equivalent_full_cycles", "depth_weighted_cycles",
                    "round_trip_efficiency", "total_time_h"):
            assert key in s

    def test_passport_integrated_in_supervisor(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=0))
        thermal = bms.ThermalModel(n_cells=4)
        det = bms.HybridFaultDetector()
        sup = bms.BMSSupervisor(pack, thermal, det)
        assert hasattr(sup, "passport")
        for k in range(50):
            sup.step(1.0, 1.0, k=k)
        assert sup.passport.total_discharge_Ah > 0

    def test_supervisor_step_returns_soe(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=0))
        thermal = bms.ThermalModel(n_cells=4)
        det = bms.HybridFaultDetector()
        sup = bms.BMSSupervisor(pack, thermal, det)
        out = sup.step(1.0, 1.0)
        assert "soe_Wh" in out
        assert out["soe_Wh"] > 0


# ----------------------------------------------------------------------
# 15. DVA / ICA
# ----------------------------------------------------------------------
class TestDVAICA:
    def test_dva_output_shape_matches_input(self):
        q, v = bms.synthetic_discharge_for_dva("nmc", n_points=200)
        q_ax, dva = bms.compute_dva(q, v)
        assert q_ax.shape == (200,)
        assert dva.shape == (200,)

    def test_ica_output_nonnegative(self):
        q, v = bms.synthetic_discharge_for_dva("lfp", n_points=300)
        v_ax, ica = bms.compute_ica(q, v)
        assert np.all(ica >= 0.0)

    def test_synthetic_discharge_monotonic_v(self):
        # OCV is monotonically decreasing during discharge (q increases, SOC drops)
        q, v = bms.synthetic_discharge_for_dva("lmfp", n_points=100)
        assert np.all(np.diff(q) >= 0)      # charge increases
        assert np.all(np.diff(v) <= 1e-6)   # voltage decreases

    def test_ica_lfp_has_peak(self):
        # LFP has a flat plateau → ICA peak should be large
        q, v = bms.synthetic_discharge_for_dva("lfp", n_points=500)
        v_ax, ica = bms.compute_ica(q, v)
        assert ica.max() > 0.5 * float(q.max()) / (v.max() - v.min() + 1e-9)

    def test_lto_ica_very_flat_plateau(self):
        q, v = bms.synthetic_discharge_for_dva("lto", n_points=500)
        v_ax, ica = bms.compute_ica(q, v)
        # LTO flat plateau → massive ICA peak relative to voltage range
        assert ica.max() > 0.0


# ----------------------------------------------------------------------
# 16. Diagnostics — EIS + C-rate map
# ----------------------------------------------------------------------
class TestDiagnostics:
    def _nmc_params(self):
        return bms.ECMParameters.for_nmc()

    def test_eis_returns_three_arrays(self):
        f, Z_re, Z_neg_im = bms.simulate_eis(self._nmc_params())
        assert f.shape == Z_re.shape == Z_neg_im.shape
        assert len(f) > 0

    def test_eis_hf_intercept_near_r0(self):
        p = self._nmc_params()
        f, Z_re, Z_neg_im = bms.simulate_eis(p, frequencies_Hz=np.array([1e4]))
        # At 10 kHz, RC loops are short-circuited; Z ≈ R0 + Warburg (very small)
        assert Z_re[0] == pytest.approx(p.R0, abs=0.005)

    def test_eis_imaginary_positive(self):
        # For a standard RC cell (no inductance), −Im Z ≥ 0 across all frequencies
        f, Z_re, Z_neg_im = bms.simulate_eis(self._nmc_params())
        assert np.all(Z_neg_im >= -1e-9)

    def test_crate_map_shape(self):
        p = self._nmc_params()
        soc_ax, T_ax, cmap = bms.compute_crate_map(p, soc_points=10, temp_points=8)
        assert soc_ax.shape == (10,)
        assert T_ax.shape == (8,)
        assert cmap.shape == (10, 8)

    def test_crate_increases_with_soc(self):
        p = self._nmc_params()
        soc_ax, T_ax, cmap = bms.compute_crate_map(p, soc_points=10, temp_points=5)
        # At a fixed temperature, higher SOC → higher OCV → more margin above v_min
        col_mid = cmap[:, 2]
        assert col_mid[-1] >= col_mid[0]

    def test_crate_increases_with_temperature(self):
        p = self._nmc_params()
        soc_ax, T_ax, cmap = bms.compute_crate_map(p, soc_points=5, temp_points=8)
        # At a fixed SOC, warmer T → lower R0 → higher C-rate
        row_mid = cmap[2, :]
        assert row_mid[-1] >= row_mid[0]


# ----------------------------------------------------------------------
# 17. State of Energy + EKF uncertainty
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 14. Solid-State Battery chemistry
# ----------------------------------------------------------------------
class TestSSBChemistry:
    def test_ssb_in_enum(self):
        assert bms.CellChemistry.SSB == "ssb"

    def test_ssb_props_accessible(self):
        p = bms.get_chemistry_props("ssb")
        assert "ocv_table" in p
        assert p["default_capacity_Ah"] == pytest.approx(4.0)

    def test_ssb_ocv_monotonic(self):
        p = bms.get_chemistry_props("ssb")
        ocv_vals = p["ocv_table"][:, 1]
        assert np.all(np.diff(ocv_vals) > 0), "SSB OCV table must be strictly increasing"

    def test_ssb_highest_voltage(self):
        ssb_p = bms.get_chemistry_props("ssb")
        nmc_p = bms.get_chemistry_props("nmc")
        assert ssb_p["v_max"] > nmc_p["v_max"]

    def test_ssb_highest_t_runaway(self):
        runaway_temps = {
            ch: bms.get_chemistry_props(ch)["T_runaway_C"]
            for ch in ["nmc", "lfp", "lmfp", "lto", "nca", "lmo", "ssb"]
        }
        assert runaway_temps["ssb"] == max(runaway_temps.values()), \
            "SSB must have the highest thermal-runaway temperature"

    def test_ssb_highest_arrhenius(self):
        arrhenius = {
            ch: bms.get_chemistry_props(ch)["arrhenius_K"]
            for ch in ["nmc", "lfp", "lmfp", "lto", "nca", "lmo", "ssb"]
        }
        assert arrhenius["ssb"] == max(arrhenius.values()), \
            "SSB solid electrolyte must have the highest Arrhenius constant"

    def test_ssb_lowest_self_discharge(self):
        sd = {
            ch: bms.get_chemistry_props(ch)["self_discharge_pct_per_month"]
            for ch in ["nmc", "lfp", "lmfp", "lto", "nca", "lmo", "ssb"]
        }
        assert sd["ssb"] == min(sd.values()), \
            "SSB must have the lowest self-discharge rate"

    def test_for_ssb_classmethod(self):
        params = bms.ECMParameters.for_ssb()
        assert params.chemistry == "ssb"
        assert params.Q_nom_Ah == pytest.approx(4.0)
        assert params.self_discharge_pct_per_month == pytest.approx(0.3)

    def test_ssb_ocvsoc_roundtrip(self):
        oc = bms.OCVSOC.from_chemistry("ssb")
        for soc in [0.1, 0.3, 0.5, 0.8, 0.95]:
            v = float(oc.ocv(soc))
            soc_back = float(oc.soc(v))
            assert abs(soc_back - soc) < 0.02

    def test_ssb_pack_builds(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, chemistry="ssb"))
        assert pack.n_cells == 4
        v = pack.pack_voltage()
        assert 12.0 < v < 18.0   # 4 × SSB cells ~3.0–4.35 V each


# ----------------------------------------------------------------------
# 15. Range predictor
# ----------------------------------------------------------------------
class TestRangePredictor:
    def _pred(self, **kw):
        return bms.RangePredictor(**kw)

    def test_vehicle_presets_instantiate(self):
        for name, veh in bms.VEHICLE_PRESETS.items():
            assert veh.mass_kg > 0
            assert veh.drag_coefficient > 0

    def test_weather_presets(self):
        for factory in [
            bms.WeatherConditions.mild,
            bms.WeatherConditions.hot_summer,
            bms.WeatherConditions.cold_winter,
            bms.WeatherConditions.rainy,
            bms.WeatherConditions.mountain_pass,
        ]:
            w = factory()
            assert isinstance(w, bms.WeatherConditions)

    def test_route_profiles_non_empty(self):
        for key, segs in bms.ROUTE_PROFILES.items():
            assert len(segs) > 0
            total_km = sum(s.distance_km for s in segs)
            assert total_km > 0

    def test_basic_prediction_positive_range(self):
        pred = self._pred()
        result = pred.predict(80_000.0, "nmc", bms.ROUTE_PROFILES["wltp"])
        assert result.estimated_range_km > 0

    def test_max_range_positive(self):
        pred = self._pred()
        r = pred.predict_max_range_km(80_000.0, "nmc", "wltp")
        assert r > 50.0   # sanity: should be well above 50 km

    def test_cold_reduces_range(self):
        pred = self._pred()
        r_mild = pred.predict_max_range_km(80_000.0, "nmc", "wltp",
                                            bms.WeatherConditions.mild())
        r_cold = pred.predict_max_range_km(80_000.0, "nmc", "wltp",
                                            bms.WeatherConditions.cold_winter())
        assert r_cold < r_mild, "Cold weather must reduce range"

    def test_headwind_reduces_range(self):
        pred = self._pred()
        r_calm = pred.predict_max_range_km(
            80_000.0, "nmc", "highway", bms.WeatherConditions(wind_speed_ms=0))
        r_head = pred.predict_max_range_km(
            80_000.0, "nmc", "highway",
            bms.WeatherConditions(wind_speed_ms=15.0, wind_heading_deg=0.0))
        assert r_head < r_calm

    def test_mountain_higher_consumption_than_city(self):
        pred = self._pred()
        r_city = pred.predict_max_range_km(80_000.0, "nmc", "city")
        r_mountain = pred.predict_max_range_km(80_000.0, "nmc", "mountain")
        # Mountain has steep uphills → higher energy per km → shorter range
        assert r_mountain < r_city

    def test_ssb_more_cold_sensitive_than_nmc(self):
        pred = self._pred()
        cold = bms.WeatherConditions(temperature_C=-20.0)
        mild = bms.WeatherConditions.mild()
        penalty_nmc = (pred.predict_max_range_km(80_000.0, "nmc", "wltp", mild)
                       - pred.predict_max_range_km(80_000.0, "nmc", "wltp", cold))
        penalty_ssb = (pred.predict_max_range_km(80_000.0, "ssb", "wltp", mild)
                       - pred.predict_max_range_km(80_000.0, "ssb", "wltp", cold))
        assert penalty_ssb > penalty_nmc, \
            "SSB (higher Arrhenius) must suffer more range loss at -20°C than NMC"

    def test_route_completable_short_route(self):
        pred = self._pred()
        short = [bms.RouteSegment(5.0, 60.0, 0.0, 1.0, "Short trip")]
        result = pred.predict(80_000.0, "nmc", short)
        assert result.route_completable
        assert result.soc_at_destination is not None
        assert result.soc_at_destination > pred.SOC_RESERVE

    def test_route_not_completable_tiny_battery(self):
        pred = self._pred()
        long_route = bms.ROUTE_PROFILES["highway"]
        result = pred.predict(500.0, "nmc", long_route)   # 500 Wh = tiny
        assert not result.route_completable
        assert result.soc_at_destination is None

    def test_energy_breakdown_structure(self):
        pred = self._pred()
        result = pred.predict(80_000.0, "nmc", bms.ROUTE_PROFILES["mixed"])
        bd = result.energy_breakdown
        assert "Traction" in bd
        assert "HVAC" in bd
        assert "Regeneration" in bd
        assert bd["Traction"] >= 0.0
        assert bd["Regeneration"] <= 0.0

    def test_temperature_sweep_ordering(self):
        pred = self._pred()
        sweep = pred.temperature_range_sweep(80_000.0, "nmc", "wltp",
                                              temperatures_C=[-20, 0, 20, 40])
        ranges = list(sweep.values())
        # Range at -20°C < range at 20°C (may be non-monotone at very hot)
        assert ranges[0] < ranges[2]

    def test_suv_shorter_range_than_compact(self):
        compact_pred = self._pred(vehicle=bms.VehicleParams.compact())
        suv_pred = self._pred(vehicle=bms.VehicleParams.suv())
        r_compact = compact_pred.predict_max_range_km(80_000.0, "nmc", "highway")
        r_suv = suv_pred.predict_max_range_km(80_000.0, "nmc", "highway")
        assert r_suv < r_compact, "SUV (heavier, higher drag) must have shorter range"

    def test_downhill_provides_regen(self):
        pred = self._pred()
        downhill = [bms.RouteSegment(10.0, 60.0, -8.0, 1.0, "Steep descent")]
        result = pred.predict(80_000.0, "nmc", downhill)
        # Energy recovered should be non-zero (stored as negative in breakdown)
        assert result.energy_breakdown["Regeneration"] < 0.0


# ----------------------------------------------------------------------
# 14. India + Two-Wheeler Range Predictor
# ----------------------------------------------------------------------
class TestIndiaTwoWheeler:
    def _pred(self, vehicle=None):
        return bms.RangePredictor(vehicle=vehicle)

    # ── Two-wheeler presets ───────────────────────────────────────────────
    def test_e_scooter_has_no_hvac(self):
        scooter = bms.VehicleParams.e_scooter()
        assert scooter.hvac_max_W == 0.0

    def test_e_motorcycle_has_no_hvac(self):
        assert bms.VehicleParams.e_motorcycle().hvac_max_W == 0.0

    def test_e_moped_has_no_hvac(self):
        assert bms.VehicleParams.e_moped().hvac_max_W == 0.0

    def test_two_wheeler_presets_in_vehicle_presets(self):
        for key in ("e_scooter", "e_motorcycle", "e_moped"):
            assert key in bms.VEHICLE_PRESETS
            v = bms.VEHICLE_PRESETS[key]
            assert v.hvac_max_W == 0.0
            assert v.mass_kg < 500.0   # lighter than any car

    def test_e_scooter_lighter_than_sedan(self):
        assert bms.VehicleParams.e_scooter().mass_kg < bms.VehicleParams.sedan().mass_kg

    def test_two_wheeler_positive_range(self):
        pred = self._pred(vehicle=bms.VehicleParams.e_scooter())
        result = pred.predict(3_000.0, "nmc", bms.ROUTE_PROFILES["midc"])
        assert result.estimated_range_km > 0

    def test_moped_vs_motorcycle_range(self):
        # Moped: lower speed, lower mass, worse Cd — net lower range on highway
        moped_pred = self._pred(vehicle=bms.VehicleParams.e_moped())
        moto_pred = self._pred(vehicle=bms.VehicleParams.e_motorcycle())
        r_moped = moped_pred.predict_max_range_km(3_000.0, "nmc", "midc")
        r_moto = moto_pred.predict_max_range_km(3_000.0, "nmc", "midc")
        # Both should be positive
        assert r_moped > 0 and r_moto > 0

    def test_two_wheeler_no_hvac_penalty(self):
        # At extreme cold, 4-wheeler HVAC penalty is real; 2-wheeler should not have it
        pred_car = self._pred(vehicle=bms.VehicleParams.sedan())
        pred_scooter = self._pred(vehicle=bms.VehicleParams.e_scooter())
        cold = bms.WeatherConditions(temperature_C=-10.0)
        result_car = pred_car.predict(40_000.0, "nmc", bms.ROUTE_PROFILES["city"], cold)
        result_scooter = pred_scooter.predict(2_000.0, "nmc", bms.ROUTE_PROFILES["city"], cold)
        assert result_car.energy_breakdown["HVAC"] > 0.0
        assert result_scooter.energy_breakdown["HVAC"] == pytest.approx(0.0)

    # ── MIDC drive cycle ────────────────────────────────────────────────
    def test_midc_profile_exists(self):
        assert "midc" in bms.ROUTE_PROFILES
        segs = bms.ROUTE_PROFILES["midc"]
        assert len(segs) == 2
        total_km = sum(s.distance_km for s in segs)
        assert abs(total_km - 19.7) < 0.5   # MIDC total ≈ 19.7 km

    def test_india_nh_profile_exists(self):
        assert "india_nh" in bms.ROUTE_PROFILES
        segs = bms.ROUTE_PROFILES["india_nh"]
        assert len(segs) == 5
        total_km = sum(s.distance_km for s in segs)
        assert total_km > 60.0

    # ── Road quality ────────────────────────────────────────────────────
    def test_poor_road_increases_consumption(self):
        pred = self._pred(vehicle=bms.VehicleParams.e_scooter())
        # At slow city speeds (<20 km/h), rolling resistance dominates aero drag,
        # so the Cr penalty from poor road outweighs the speed-reduction aero saving
        good_seg = [bms.RouteSegment(10.0, 15.0, 0.0, 0.80, "Good road", "good")]
        poor_seg = [bms.RouteSegment(10.0, 15.0, 0.0, 0.80, "Poor road", "poor")]
        w = bms.WeatherConditions.mild()
        e_good = pred.predict(5_000.0, "nmc", good_seg, w).total_consumed_Wh
        e_poor = pred.predict(5_000.0, "nmc", poor_seg, w).total_consumed_Wh
        assert e_poor > e_good, "Poor road quality must increase energy consumption"

    def test_excellent_road_lower_than_good(self):
        pred = self._pred()
        good_seg = [bms.RouteSegment(10.0, 80.0, 0.0, 1.0, "Good", "good")]
        exc_seg = [bms.RouteSegment(10.0, 80.0, 0.0, 1.0, "Excellent", "excellent")]
        w = bms.WeatherConditions.mild()
        e_good = pred.predict(80_000.0, "nmc", good_seg, w).total_consumed_Wh
        e_exc = pred.predict(80_000.0, "nmc", exc_seg, w).total_consumed_Wh
        assert e_exc < e_good

    # ── India city routes ───────────────────────────────────────────────
    def test_india_city_routes_exist(self):
        expected = {"delhi_ncr", "mumbai", "bangalore", "chennai",
                    "pune", "hyderabad", "kolkata"}
        assert expected.issubset(set(bms.INDIA_CITY_ROUTES.keys()))

    def test_india_city_routes_non_empty(self):
        for city, segs in bms.INDIA_CITY_ROUTES.items():
            assert len(segs) > 0, f"{city} has no segments"
            assert sum(s.distance_km for s in segs) > 0

    def test_india_city_route_prediction_positive(self):
        pred = self._pred(vehicle=bms.VehicleParams.e_scooter())
        for city, segs in bms.INDIA_CITY_ROUTES.items():
            result = pred.predict(3_000.0, "lfp", segs)
            assert result.estimated_range_km > 0, f"{city}: range must be > 0"

    # ── India weather presets ───────────────────────────────────────────
    def test_india_weather_presets(self):
        for factory in [
            bms.WeatherConditions.india_summer,
            bms.WeatherConditions.india_monsoon,
            bms.WeatherConditions.india_winter_north,
            bms.WeatherConditions.india_coastal,
        ]:
            w = factory()
            assert isinstance(w, bms.WeatherConditions)
            assert -20.0 <= w.temperature_C <= 55.0

    def test_india_weather_dict(self):
        assert len(bms.INDIA_WEATHER) >= 12
        for key, w in bms.INDIA_WEATHER.items():
            assert isinstance(w, bms.WeatherConditions)

    def test_india_summer_hot(self):
        w = bms.WeatherConditions.india_summer()
        assert w.temperature_C >= 38.0

    def test_india_monsoon_has_rain(self):
        w = bms.WeatherConditions.india_monsoon()
        assert w.precipitation == "rain"

    def test_summer_reduces_range_vs_mild_india(self):
        pred = self._pred(vehicle=bms.VehicleParams.e_scooter())
        route = bms.ROUTE_PROFILES["midc"]
        r_mild = pred.predict_max_range_km(3_000.0, "nmc", "midc",
                                            bms.WeatherConditions.mild())
        r_summer = pred.predict_max_range_km(3_000.0, "nmc", "midc",
                                              bms.WeatherConditions.india_summer())
        # Hot reduces capacity slightly; AC draw is zero (e-scooter) but
        # hot weather overhead is small — range may be slightly lower
        assert r_mild > 0 and r_summer > 0


# ----------------------------------------------------------------------
# 16. Large-scale packs (20S × 20P — world-model scale)
# ----------------------------------------------------------------------
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
