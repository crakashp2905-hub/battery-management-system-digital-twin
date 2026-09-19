"""Tests: degradation-mode inference (why is it aging, not just how much)."""

from __future__ import annotations

import bms


class TestModeAttribution:
    def test_lli_dominant_ic_gives_lli_largest_capacity_mode(self):
        v, fresh, aged = bms.synthetic_degraded_ic(lli=0.25, lam=0.05)
        d = bms.infer_degradation_modes(r0_bol=0.025, r0_now=0.027,
                                        v_axis=v, ic_fresh=fresh, ic_aged=aged)
        assert d.mode_fractions["LLI"] > d.mode_fractions["LAM"]
        assert set(d.mode_fractions) == {"LLI", "LAM", "resistance"}
        assert abs(sum(d.mode_fractions.values()) - 1.0) < 1e-6

    def test_resistance_growth_shows_as_its_own_mode(self):
        v, fresh, aged = bms.synthetic_degraded_ic(lli=0.05, lam=0.05)
        d = bms.infer_degradation_modes(r0_bol=0.025, r0_now=0.055,   # R0 more than doubled
                                        v_axis=v, ic_fresh=fresh, ic_aged=aged)
        assert d.resistance_growth_pct > 100.0
        assert d.mode_fractions["resistance"] > 0.3   # a large share of the pie

    def test_without_ic_falls_back_to_two_modes(self):
        d = bms.infer_degradation_modes(soh_capacity=0.85, r0_bol=0.025, r0_now=0.030)
        assert set(d.mode_fractions) == {"capacity_loss", "resistance"}
        assert abs(d.capacity_loss_pct - 15.0) < 1e-6


class TestMechanismInference:
    def test_calendar_high_soc_history_points_to_sei(self):
        hist = bms.OperatingHistory(mean_soc=0.9, calendar_days=700,
                                    equivalent_full_cycles=100, mean_temperature_C=30)
        v, fresh, aged = bms.synthetic_degraded_ic(lli=0.2, lam=0.03)
        d = bms.infer_degradation_modes(r0_bol=0.025, r0_now=0.027, v_axis=v,
                                        ic_fresh=fresh, ic_aged=aged, history=hist)
        top = d.likely_mechanisms[0][0]
        assert top == "calendar_sei_high_soc"
        assert "SoH" in d.narrative and "mechanism" in d.narrative

    def test_cold_fast_charging_points_to_plating(self):
        hist = bms.OperatingHistory(cold_charge_events=20, fast_charge_events=20,
                                    mean_temperature_C=10)
        v, fresh, aged = bms.synthetic_degraded_ic(lli=0.15, lam=0.02)
        d = bms.infer_degradation_modes(r0_bol=0.025, r0_now=0.04, v_axis=v,
                                        ic_fresh=fresh, ic_aged=aged, history=hist)
        names = [m[0] for m in d.likely_mechanisms]
        assert names[0] == "lithium_plating"

    def test_high_rate_history_points_to_rate_stress(self):
        hist = bms.OperatingHistory(mean_c_rate=3.0, equivalent_full_cycles=1200)
        d = bms.infer_degradation_modes(soh_capacity=0.9, r0_bol=0.025, r0_now=0.05,
                                        history=hist)
        names = [m[0] for m in d.likely_mechanisms]
        assert "high_rate_stress" in names[:2]

    def test_wrapper_and_serialisation(self):
        eng = bms.DegradationInference(r0_bol=0.025,
                                       history=bms.OperatingHistory(mean_soc=0.85,
                                                                    calendar_days=600))
        v, fresh, aged = bms.synthetic_degraded_ic(lli=0.2, lam=0.05)
        d = eng.diagnose(r0_now=0.03, v_axis=v, ic_fresh=fresh, ic_aged=aged)
        out = d.to_dict()
        for key in ("mode_fractions", "dominant_mode", "likely_mechanisms", "narrative"):
            assert key in out
