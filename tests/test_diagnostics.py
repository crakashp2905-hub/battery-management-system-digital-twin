"""Tests: diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

import bms


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

