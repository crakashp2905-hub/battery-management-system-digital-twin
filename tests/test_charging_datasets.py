"""Tests: charging datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import bms


class TestChargingAging:
    _ARGS = dict(pack_energy_kWh=60.0, q_nom_Ah=2.3, r0_ohm=0.025, v_nom=3.7)

    def _charge(self, method, **kw):
        proto = bms.ChargeProtocol(method, soc_start=0.2, soc_end=0.9, **kw)
        return bms.ChargingModel().simulate(proto, **self._ARGS)

    def test_c_rate_scales_with_power(self):
        m = bms.ChargingModel()
        slow = m.effective_c_rate(bms.ChargeProtocol(bms.ChargeMethod.AC_LEVEL2), 60.0)
        fast = m.effective_c_rate(bms.ChargeProtocol(bms.ChargeMethod.DC_ULTRA), 60.0)
        assert 0.0 < slow < fast
        assert fast > 3.0                      # 250 kW into 60 kWh ≈ 4C

    def test_ultra_charging_faster_hotter_and_ages_more(self):
        ac = self._charge(bms.ChargeMethod.AC_LEVEL2)
        dc = self._charge(bms.ChargeMethod.DC_ULTRA)
        assert dc.duration_min < ac.duration_min           # fast to charge
        assert dc.peak_cell_temp_C > ac.peak_cell_temp_C    # but hotter
        assert dc.plating_risk > 0.0 and ac.plating_risk == 0.0
        assert dc.capacity_fade_pct > 5.0 * ac.capacity_fade_pct  # and ages far more
        assert 0.0 < ac.efficiency < 1.0 and 0.0 < dc.efficiency < 1.0

    def test_moderate_dc_is_gentle(self):
        # 50 kW DC (~0.8C) triggers no plating and ages like AC, unlike 250 kW.
        dc_fast = self._charge(bms.ChargeMethod.DC_FAST)
        ac = self._charge(bms.ChargeMethod.AC_LEVEL2)
        assert dc_fast.plating_risk == 0.0
        assert dc_fast.capacity_fade_pct < 3.0 * ac.capacity_fade_pct

    def test_cold_fast_charge_raises_plating_and_fade(self):
        warm = self._charge(bms.ChargeMethod.DC_ULTRA, ambient_C=25.0)
        cold = self._charge(bms.ChargeMethod.DC_ULTRA, ambient_C=0.0)
        assert cold.plating_risk > warm.plating_risk
        assert cold.capacity_fade_pct > warm.capacity_fade_pct

    def test_aging_cycle_reduces_soh_and_raises_resistance(self):
        model = bms.AgingModel()
        s = model.cycle(bms.AgingState(), throughput_efc=1.0, c_rate=3.0,
                        temperature_C=40.0, dod=1.0)
        assert s.soh_capacity < 1.0
        assert s.soh_resistance > 1.0
        assert s.equivalent_full_cycles == pytest.approx(1.0)

    def test_calendar_fade_follows_sqrt_time(self):
        model = bms.AgingModel()
        s = bms.AgingState()
        s1 = model.calendar(s, days=100.0, temperature_C=35.0, soc_avg=0.9)
        # A second equal interval adds less fade than the first (√-time law).
        first = s.soh_capacity - s1.soh_capacity
        s2 = model.calendar(s1, days=100.0, temperature_C=35.0, soc_avg=0.9)
        second = s1.soh_capacity - s2.soh_capacity
        assert first > second > 0.0
        assert s2.calendar_days == pytest.approx(200.0)

    def test_apply_to_pack_scales_capacity_and_resistance_idempotently(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=3))
        q0 = pack.capacities_Ah.copy()
        r0 = np.array([g.params.R0 for g in pack.groups])
        model = bms.AgingModel()
        state = bms.AgingState(soh_capacity=0.8, soh_resistance=1.5)
        model.apply_to_pack(pack, state)
        assert np.allclose(pack.capacities_Ah, 0.8 * q0, rtol=1e-6)
        assert np.allclose([g.params.R0 for g in pack.groups], 1.5 * r0, rtol=1e-6)
        model.apply_to_pack(pack, state)   # again → same, not compounded
        assert np.allclose(pack.capacities_Ah, 0.8 * q0, rtol=1e-6)

    def test_ultra_charging_lower_soh_over_life(self):
        model = bms.AgingModel()

        def soh_after(method, n):
            r = self._charge(method)
            s = bms.AgingState()
            for _ in range(n):
                s = model.cycle(s, r.throughput_efc, r.c_rate, r.peak_cell_temp_C,
                                r.throughput_efc, 0.5, r.plating_risk)
            return s.soh_capacity

        soh_ac = soh_after(bms.ChargeMethod.AC_LEVEL2, 300)
        soh_dc = soh_after(bms.ChargeMethod.DC_ULTRA, 300)
        assert soh_ac > 0.95            # slow charging barely ages over 300 charges
        assert soh_dc < soh_ac          # ultra-rapid degrades markedly more



class TestDatasets:
    def test_synthetic_drivecycle_consistent(self):
        d = bms.synthetic_drivecycle("nmc", duration_s=600, seed=3)
        assert d.n == len(d.soc_true) == len(d.voltage_V)
        assert np.all((d.soc_true >= 0.0) & (d.soc_true <= 1.0))
        assert 2.5 < float(np.median(d.voltage_V)) < 4.3
        assert d.dt == pytest.approx(1.0)

    def test_csv_round_trip(self, tmp_path):
        d = bms.synthetic_drivecycle("lfp", duration_s=300, seed=4)
        p = tmp_path / "dc.csv"
        bms.save_drivecycle_csv(d, p)
        d2 = bms.load_drivecycle_csv(p, chemistry="lfp")
        assert d2.n == d.n
        assert np.allclose(d2.soc_true, d.soc_true)
        assert np.allclose(d2.voltage_V, d.voltage_V)

    def test_leaderboard_kf_beats_coulomb_under_bias(self):
        d = bms.synthetic_drivecycle("nmc", duration_s=1200, seed=1, current_bias_A=0.4)
        lb = bms.estimator_leaderboard(d)
        assert {"rmse", "mae", "max_err", "runtime_s"} <= set(lb.columns)
        assert list(lb["rmse"]) == sorted(lb["rmse"])            # ranked by rmse
        assert lb.loc["ekf", "rmse"] < lb.loc["coulomb", "rmse"]  # KF beats drift

    def test_coulomb_counted_when_soc_column_absent(self, tmp_path):
        import pandas as pd
        d = bms.synthetic_drivecycle("nmc", duration_s=200, seed=2)
        pd.DataFrame({"time_s": d.time_s, "current_A": d.current_A,
                      "voltage_V": d.voltage_V}).to_csv(tmp_path / "no_soc.csv", index=False)
        d2 = bms.load_drivecycle_csv(tmp_path / "no_soc.csv", chemistry="nmc",
                                     soc0=float(d.soc_true[0]))
        assert d2.n == d.n and np.all((d2.soc_true >= 0.0) & (d2.soc_true <= 1.0))

    def test_committed_sample_loads_and_scores(self):
        sample = (Path(__file__).resolve().parents[1]
                  / "data" / "samples" / "synthetic_drivecycle_nmc.csv")
        assert sample.exists()
        d = bms.load_drivecycle_csv(sample, chemistry="nmc")
        lb = bms.estimator_leaderboard(d, estimators=["coulomb", "ekf"])
        assert bool(np.isfinite(lb["rmse"]).all())

    def test_capacity_fade_soh_and_rul(self, tmp_path):
        import pandas as pd
        cyc = np.arange(1, 201.0)
        cap = 2.3 * (1 - 0.0004 * cyc)
        pd.DataFrame({"cycle": cyc, "capacity_Ah": cap}).to_csv(tmp_path / "cap.csv", index=False)
        c2, cap2 = bms.load_capacity_fade_csv(tmp_path / "cap.csv")
        assert np.allclose(cap2, cap)
        soh, rul = bms.soh_curve(cap2)
        assert soh[0] == pytest.approx(1.0) and soh[-1] < 1.0
        assert rul > 0.0 and np.isfinite(rul)

    def test_nasa_mat_parser(self, tmp_path):
        from scipy.io import savemat
        p = tmp_path / "B0005.mat"
        savemat(str(p), {"B0005": {"cycle": [
            {"type": "discharge", "data": {"Capacity": 1.85}},
            {"type": "charge", "data": {}},
            {"type": "discharge", "data": {"Capacity": 1.74}}]}})
        nums, caps = bms.nasa_mat_to_capacity(str(p))
        assert nums.tolist() == [1.0, 2.0]
        assert np.allclose(caps, [1.85, 1.74])

