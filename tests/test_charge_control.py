"""Tests: MPC / optimal fast charging."""

from __future__ import annotations

import numpy as np

import bms
from bms.charge_control import ChargeLimits, MPCCharger, compare_charging


def _charger(**lim_kw):
    p = bms.ECMParameters(R0=0.025, R1=0.015, C1=2000, R2=0.03, C2=8000, Q_nom_Ah=2.3)
    limits = ChargeLimits(**{"v_max": 4.2, "t_max_C": 45.0, "i_max_A": 11.0,
                             "soc_target": 0.8, **lim_kw})
    return MPCCharger(params=p, ocv_curve=bms.OCVSOC(), limits=limits, ambient_C=25.0)


class TestMPCCharging:
    def test_reaches_target_and_respects_all_limits(self):
        out = _charger().charge(soc0=0.2, dt=1.0)
        assert out["reached_target"]
        assert out["final_soc"] >= 0.8 - 1e-6
        # Never violates voltage, temperature, or plating.
        assert out["voltage_V"].max() <= 4.2 + 1e-3
        assert out["peak_temperature_C"] <= 45.0 + 0.5
        assert out["min_plating_margin_A"] >= -1e-6      # stays at/under plating cap

    def test_beats_cccv_time_at_bounded_temperature(self):
        cmp = compare_charging(_charger(), soc0=0.2, c_rate=1.0, dt=1.0)
        assert cmp["mpc_faster"]
        assert cmp["time_saving_s"] > 0
        # Both respect the temperature ceiling.
        assert cmp["mpc"]["peak_temperature_C"] <= 45.0 + 0.5
        assert cmp["cccv"]["peak_temperature_C"] <= 45.0 + 0.5

    def test_plating_cap_binds_in_the_cold(self):
        # Cold start → low plating C-limit → current is plating-limited, not V/T.
        warm = _charger().charge(soc0=0.2, T0=25.0, dt=1.0)
        cold = _charger().charge(soc0=0.2, T0=0.0, dt=1.0)
        assert cold["current_A"].max() < warm["current_A"].max()   # colder → gentler

    def test_plating_unaware_charges_harder(self):
        ch = _charger()
        ch.plating_aware = False
        out = ch.charge(soc0=0.2, T0=5.0, dt=1.0)
        ch2 = _charger()                                           # plating-aware
        out2 = ch2.charge(soc0=0.2, T0=5.0, dt=1.0)
        assert out["current_A"].max() >= out2["current_A"].max()

    def test_current_is_monotone_feasible(self):
        out = _charger().charge(soc0=0.2, dt=1.0)
        assert np.all(out["current_A"] >= 0.0)
        assert np.all(out["current_A"] <= 11.0 + 1e-6)            # hardware cap
