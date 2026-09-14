"""Tests: analog front-end (AFE) measurement model."""

from __future__ import annotations

import numpy as np

import bms
from bms.afe import AFE, AFEConfig


class TestAFE:
    def test_quantisation_is_on_a_grid(self):
        afe = AFE(AFEConfig(v_bits=12, v_range_V=5.0, v_noise_V=0.0,
                            v_gain_error=0.0, v_offset_V=0.0))
        step = 5.0 / 2 ** 12
        m = afe.measure_voltage(3.71234)
        assert abs(round(m / step) * step - m) < 1e-9         # lands on an ADC level
        assert abs(m - 3.71234) < step                         # within one LSB

    def test_gain_and_offset_bias_the_reading(self):
        afe = AFE(AFEConfig(v_noise_V=0.0, v_gain_error=0.01, v_offset_V=0.05,
                            v_bits=16))
        assert afe.measure_voltage(4.0) > 4.0                  # +1% gain, +50 mV offset

    def test_current_sensor_bandwidth_smears_a_step(self):
        afe = AFE(AFEConfig(i_bandwidth_hz=1.0, i_noise_A=0.0, i_offset_A=0.0,
                            i_bits=16, i_range_A=300.0))
        # A current step: the bandwidth-limited reading lags the true value.
        afe.measure_current(0.0, dt=1.0)
        first = afe.measure_current(100.0, dt=1.0)
        assert first < 100.0                                   # cannot jump instantly
        later = first
        for _ in range(20):
            later = afe.measure_current(100.0, dt=1.0)
        assert later > first                                   # settles upward

    def test_apply_to_drivecycle_changes_signals_and_tags(self):
        d = bms.synthetic_drivecycle("nmc", duration_s=600, seed=1)
        d_afe = AFE(AFEConfig()).apply_to_drivecycle(d)
        assert not np.allclose(d.voltage_V, d_afe.voltage_V)
        assert not np.allclose(d.current_A, d_afe.current_A)
        assert d_afe.name.endswith("+afe")
        assert len(d_afe.voltage_V) == len(d.voltage_V)

    def test_afe_degrades_coulomb_and_reshuffles_ranking(self):
        # Under a realistic AFE (current offset + bandwidth + quantisation) the
        # open-loop Coulomb counter degrades, so a voltage-feedback filter wins.
        d = bms.synthetic_drivecycle("nmc", duration_s=1800, seed=1)
        d_afe = AFE(AFEConfig(i_offset_A=0.05, i_bandwidth_hz=5.0,
                              v_bits=12)).apply_to_drivecycle(d)
        ideal = bms.estimator_leaderboard(d, ["coulomb", "ekf", "ukf"])
        afe = bms.estimator_leaderboard(d_afe, ["coulomb", "ekf", "ukf"])
        assert ideal.index[0] == "coulomb"                     # ideal: coulomb best
        assert afe.loc["coulomb", "rmse"] > ideal.loc["coulomb", "rmse"]  # AFE hurts it
        assert afe.index[0] != "coulomb"                       # a filter now wins

    def test_reset_makes_it_reproducible(self):
        cfg = AFEConfig(seed=7)
        a1 = AFE(cfg); a2 = AFE(cfg)
        assert a1.measure_voltage(3.7) == a2.measure_voltage(3.7)
