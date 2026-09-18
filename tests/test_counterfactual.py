"""Tests: the Counterfactual Battery Twin (what-if policy simulation)."""

from __future__ import annotations

import numpy as np

import bms

_P = bms.ECMParameters(R0=0.025, R1=0.015, C1=2000, R2=0.03, C2=8000, Q_nom_Ah=2.3)


def _charger():
    return bms.MPCCharger(params=_P, ocv_curve=bms.OCVSOC(),
                          limits=bms.ChargeLimits(soc_target=0.9, i_max_A=12.0))


class TestCounterfactualCharge:
    def test_faster_charge_costs_more_health(self):
        res = bms.counterfactual_charge(_charger(), soc0=0.2, c_rates=(1.0, 2.0, 3.0))
        by = {o.label: o for o in res["outcomes"]}
        # Higher C-rate: shorter time, hotter peak, more capacity fade.
        assert by["3C"].charge_time_s < by["1C"].charge_time_s
        assert by["3C"].peak_temperature_C > by["1C"].peak_temperature_C
        assert by["3C"].capacity_fade_pct > by["1C"].capacity_fade_pct

    def test_fastest_and_gentlest_are_the_expected_extremes(self):
        res = bms.counterfactual_charge(_charger(), soc0=0.2, c_rates=(1.0, 2.0, 3.0))
        assert res["fastest"] == "3C"        # highest C-rate reaches target soonest
        assert res["gentlest"] == "1C"       # lowest C-rate degrades least

    def test_outcomes_are_serialisable_and_complete(self):
        res = bms.counterfactual_charge(_charger(), soc0=0.3, c_rates=(1.0, 2.0))
        assert len(res["table"]) == 2
        row = res["table"][0]
        for key in ("label", "charge_time_s", "peak_temperature_C",
                    "capacity_fade_pct", "resistance_growth_pct"):
            assert key in row


class TestAlternateHistory:
    def test_harder_driving_ages_the_battery_more(self):
        i = np.zeros(1200)
        i[100:1100] = 1.0        # a 1C-ish discharge mission
        cmp = bms.alternate_history(i, 1.0, params=_P, soc0=0.9, current_scale=2.0)
        assert cmp.counterfactual_capacity_fade_pct > cmp.baseline_capacity_fade_pct
        assert cmp.worse == "counterfactual"

    def test_cooler_operation_ages_less(self):
        i = np.zeros(1000)
        i[100:900] = 1.5
        cmp = bms.alternate_history(i, 1.0, params=_P, soc0=0.9,
                                    temperatures=np.full(1000, 40.0),
                                    temperature_delta_C=-15.0, label="cooler")
        # Same mission 15 °C cooler degrades less than the hot baseline.
        assert cmp.counterfactual_capacity_fade_pct < cmp.baseline_capacity_fade_pct
        assert cmp.worse == "baseline"

    def test_wrapper_compare_charge(self):
        twin = bms.CounterfactualTwin(params=_P,
                                      limits=bms.ChargeLimits(soc_target=0.9, i_max_A=12.0))
        res = twin.compare_charge(soc0=0.2, c_rates=(1.0, 3.0))
        assert res["fastest"] == "3C" and res["gentlest"] == "1C"
