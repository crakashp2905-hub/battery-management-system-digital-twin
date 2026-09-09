"""Tests: calibration."""

from __future__ import annotations

import numpy as np
import pytest

import bms


class TestCalibration:
    @staticmethod
    def _pulse(true_R0=0.028):
        ocv = bms.OCVSOC()
        true = bms.ECMParameters(R0=true_R0, R1=0.014, C1=2200, R2=0.03, C2=9000,
                                 Q_nom_Ah=2.3)
        ecm = bms.SecondOrderECM(params=true, ocv_curve=ocv)
        ecm.reset(0.8)
        n = 1200
        i = np.zeros(n)
        i[100:250] = 2.3        # discharge pulse
        i[400:550] = -2.3       # charge pulse
        i[700:1000] = 1.15      # sustained load
        sim = ecm.simulate(i, 1.0)
        v = sim["v_terminal"] + np.random.default_rng(0).normal(0, 0.002, n)
        return i, v

    def test_fit_from_pulse_recovers_params(self):
        i, v = self._pulse(true_R0=0.028)
        res = bms.fit_from_pulse(i, v, 1.0, capacity_Ah=2.3)
        assert res.success
        assert abs(res.params.R0 - 0.028) < 0.005      # recovers R0 from a noisy pulse
        assert res.rmse_v < 0.01                        # voltage fit < 10 mV

    def test_fit_cell_distribution(self):
        rng = np.random.default_rng(1)
        cells = [self._pulse(true_R0=0.028 * (1 + rng.normal(0, 0.08))) for _ in range(4)]
        dist = bms.fit_cell_distribution(cells)
        assert dist["n_cells"] == 4
        assert abs(dist["distribution"]["R0"]["mean"] - 0.028) < 0.006
        assert dist["pack_scatter"]["r0_sigma"] > 0.0   # data-learned scatter

    def test_validation_report_bucketed_and_tagged(self):
        d = bms.synthetic_drivecycle("nmc", duration_s=1000, seed=2)
        rep = bms.validation_report(d, "ekf", buckets_by="c_rate", n_buckets=3)
        assert {"rmse", "mae", "max_err", "source", "chemistry", "estimator"} <= set(rep.columns)
        assert (rep["source"] == "synthetic").all()
        assert bool(np.isfinite(rep["rmse"]).all()) and len(rep) >= 2

    def test_source_tag_real_vs_synthetic(self, tmp_path):
        d = bms.synthetic_drivecycle("nmc", duration_s=200, seed=3)
        assert d.source == "synthetic"
        bms.save_drivecycle_csv(d, tmp_path / "dc.csv")
        loaded = bms.load_drivecycle_csv(tmp_path / "dc.csv", chemistry="nmc")
        assert loaded.source == "real"


class TestTemperatureBenchmark:
    def test_synthetic_drivecycle_honours_temperature(self):
        warm = bms.synthetic_drivecycle("nmc", duration_s=600, seed=1, temperature_C=25.0)
        cold = bms.synthetic_drivecycle("nmc", duration_s=600, seed=1, temperature_C=-15.0)
        assert np.allclose(warm.temperature_C, 25.0)
        assert np.allclose(cold.temperature_C, -15.0)
        # A genuinely colder plant → different terminal voltage (Arrhenius R + OCV).
        assert not np.allclose(warm.voltage_V, cold.voltage_V)
        # 25 °C is the reference, so the measured current (the input) is unchanged.
        assert np.allclose(warm.current_A, cold.current_A)

    def test_temperature_aware_beats_naive_in_the_cold(self):
        # Same cold trace scored two ways: filter told the temperature vs
        # assuming 25 °C.  Knowing the temperature must not hurt, and helps a lot.
        cold = bms.synthetic_drivecycle("nmc", duration_s=1200, seed=1,
                                        current_bias_A=0.05, temperature_C=-10.0)
        aware = bms.estimator_leaderboard(cold, ["ekf"], temperature_aware=True)
        naive = bms.estimator_leaderboard(cold, ["ekf"], temperature_aware=False)
        assert aware.loc["ekf", "rmse"] < naive.loc["ekf", "rmse"]
        assert aware.loc["ekf", "rmse"] < 0.02          # aware stays accurate (<2%)

    def test_temperature_aware_is_noop_at_reference(self):
        # At 25 °C the "naive" assumption is correct, so both agree.
        warm = bms.synthetic_drivecycle("nmc", duration_s=800, seed=4,
                                        current_bias_A=0.05, temperature_C=25.0)
        aware = bms.estimator_leaderboard(warm, ["ekf"], temperature_aware=True)
        naive = bms.estimator_leaderboard(warm, ["ekf"], temperature_aware=False)
        assert aware.loc["ekf", "rmse"] == pytest.approx(naive.loc["ekf", "rmse"], abs=1e-9)
