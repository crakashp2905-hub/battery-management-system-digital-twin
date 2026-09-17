"""Tests: Monte-Carlo fleet reliability / warranty curves."""

from __future__ import annotations

import numpy as np

import bms


class TestMonteCarloLife:
    def test_shapes_and_monotone_fade(self):
        mc = bms.monte_carlo_life(n_samples=300, years=12, seed=0)
        assert mc["years"].shape == (13,)
        assert mc["soh_p50"].shape == (13,)
        assert mc["soh_p50"][0] == 1.0
        assert mc["soh_p50"][-1] < mc["soh_p50"][0]           # fleet fades
        # Percentile band is ordered p05 <= p50 <= p95.
        assert np.all(mc["soh_p05"] <= mc["soh_p50"] + 1e-9)
        assert np.all(mc["soh_p50"] <= mc["soh_p95"] + 1e-9)

    def test_warranty_curve_is_a_cdf(self):
        mc = bms.monte_carlo_life(n_samples=500, years=14, seed=1)
        w = mc["warranty_curve"]
        assert w[0] == 0.0                                    # nobody failed at t=0
        assert np.all(np.diff(w) >= -1e-9)                    # non-decreasing
        assert 0.0 <= w[-1] <= 1.0

    def test_rul_interval_brackets_the_mean(self):
        mc = bms.monte_carlo_life(n_samples=500, years=16, seed=2)
        assert mc["rul_years_p05"] <= mc["rul_years_mean"] <= mc["rul_years_p95"]
        assert mc["rul_years_p05"] < mc["rul_years_p95"]      # non-trivial spread

    def test_heat_shortens_life(self):
        cool = bms.monte_carlo_life(n_samples=500, years=16, temperature_C=25, seed=3)
        hot = bms.monte_carlo_life(n_samples=500, years=16, temperature_C=42, seed=3)
        assert hot["b10_life_years"] < cool["b10_life_years"]
        assert hot["rul_years_mean"] < cool["rul_years_mean"]

    def test_more_scatter_widens_the_band(self):
        tight = bms.monte_carlo_life(n_samples=600, years=14, k_cycle_cov=0.05, seed=4)
        wide = bms.monte_carlo_life(n_samples=600, years=14, k_cycle_cov=0.30, seed=4)
        spread_t = tight["soh_p95"][-1] - tight["soh_p05"][-1]
        spread_w = wide["soh_p95"][-1] - wide["soh_p05"][-1]
        assert spread_w > spread_t

    def test_warranty_reserve_reads_the_curve(self):
        mc = bms.monte_carlo_life(n_samples=500, years=14, seed=5)
        r = bms.warranty_reserve(mc, 8.0)
        assert 0.0 <= r <= 1.0
        assert r == bms.warranty_reserve(mc, 8.0)             # deterministic read
