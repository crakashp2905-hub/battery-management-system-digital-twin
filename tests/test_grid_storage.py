"""Tests: stationary / grid storage dispatch."""

from __future__ import annotations

import numpy as np

import bms
from bms.aging import AgingModel, AgingParams


class TestStationaryStorage:
    def test_dispatch_respects_power_and_energy_limits(self):
        s = bms.StationaryStorage(capacity_kWh=100, max_power_kW=50,
                                  round_trip_efficiency=0.9)
        s.reset(0.5)
        assert s.dispatch(80.0, 1.0) <= 50.0 + 1e-9          # power cap
        s.reset(0.06)                                        # nearly empty
        assert s.dispatch(50.0, 1.0) < 5.0                   # energy-limited

    def test_round_trip_efficiency_loses_energy(self):
        s = bms.StationaryStorage(capacity_kWh=100, max_power_kW=100,
                                  round_trip_efficiency=0.81, soc_min=0.0, soc_max=1.0)
        s.reset(0.0)
        charged_in = -s.dispatch(-50.0, 1.0)                 # charge 50 kWh from grid
        delivered_out = sum(s.dispatch(100.0, 1.0) for _ in range(20))  # drain fully
        # A full round trip returns RTE (0.81) of what went in — strictly less.
        assert delivered_out < charged_in
        assert delivered_out == __import__("pytest").approx(charged_in * 0.81, rel=0.05)


class TestDispatchStrategies:
    def test_peak_shaving_caps_grid_load(self):
        load = np.array([20, 30, 80, 90, 40, 25, 70, 15.0] * 3)
        batt = bms.peak_shaving_dispatch(load, threshold_kW=50)
        s = bms.StationaryStorage(capacity_kWh=200, max_power_kW=60)
        out = bms.simulate_dispatch(s, batt, dt_h=0.25, soc0=0.8)
        grid = load - out["power_kW"]                        # net load on the grid
        assert grid.max() <= 50.0 + 5.0                      # peaks shaved to threshold
        assert out["energy_served_kWh"] > 0

    def test_arbitrage_buys_low_sells_high(self):
        price = np.array([0.05, 0.30, 0.04, 0.35])
        sch = bms.arbitrage_schedule(price, buy_below=0.08, sell_above=0.25, power_kW=40)
        assert sch[0] < 0 and sch[2] < 0                     # charge when cheap
        assert sch[1] > 0 and sch[3] > 0                     # discharge when dear

    def test_degradation_aware_reduces_throughput(self):
        load = np.array([20, 30, 80, 90, 40, 25, 70, 15.0] * 3)
        batt = bms.peak_shaving_dispatch(load, threshold_kW=50)
        ag = AgingModel(AgingParams())
        naive = bms.simulate_dispatch(bms.StationaryStorage(), batt, dt_h=0.25, soc0=0.6)
        aware = bms.simulate_dispatch(bms.StationaryStorage(), batt, dt_h=0.25, soc0=0.6,
                                      aging=ag, degradation_aware=True,
                                      value_per_kWh=0.01, fade_cost_per_pct=500.0)
        assert aware["throughput_kWh"] < naive["throughput_kWh"]


class TestOptimalStorageSoC:
    def test_lower_soc_minimises_calendar_fade(self):
        opt = bms.optimal_storage_soc(days=365, temperature_C=25.0)
        assert 0.05 <= opt["best_soc"] <= 0.5                # low-mid SoC is best
        assert opt["fade_at_best_pct"] == opt["fade_pct"].min()
        assert opt["fade_pct"].max() > opt["fade_at_best_pct"]   # SoC actually matters

    def test_hotter_storage_fades_more(self):
        cool = bms.optimal_storage_soc(days=365, temperature_C=15.0)
        hot = bms.optimal_storage_soc(days=365, temperature_C=40.0)
        assert hot["fade_pct"].mean() > cool["fade_pct"].mean()
