"""Tests: probabilistic prognostics (runaway probability, RUL, failure lead time)."""

from __future__ import annotations

import numpy as np

import bms


class TestRunawayProbability:
    def test_probability_rises_with_horizon_when_heating(self):
        f = bms.thermal_runaway_probability(55.0, 0.02, onset_C=70.0)  # +0.02 °C/s
        p = [f.prob_within(h) for h in f.horizons_s]
        assert p[0] <= p[1] <= p[2]                 # monotone in horizon
        assert p[-1] > p[0]

    def test_cooling_cell_is_low_risk(self):
        f = bms.thermal_runaway_probability(40.0, -0.01, onset_C=70.0)
        assert all(v < 0.2 for v in f.probabilities.values())

    def test_imbalance_and_pressure_lower_the_effective_onset(self):
        calm = bms.thermal_runaway_probability(60.0, 0.01, onset_C=70.0)
        stressed = bms.thermal_runaway_probability(60.0, 0.01, onset_C=70.0,
                                                   imbalance=0.3, pressure_kPa=300.0)
        assert stressed.effective_onset_C < calm.effective_onset_C
        assert stressed.prob_within(300.0) > calm.prob_within(300.0)


class TestRULDistribution:
    def test_distribution_brackets_the_point_estimate(self):
        d = bms.rul_distribution(soh_now=0.92, fade_per_cycle=2e-4, eol=0.8)
        point = (0.92 - 0.8) / 2e-4                  # 600 cycles
        assert d.p05_cycles < point < d.p95_cycles
        assert d.p05_cycles < d.median_cycles < d.p95_cycles
        assert d.std_cycles > 0.0

    def test_more_uncertain_fade_widens_the_interval(self):
        tight = bms.rul_distribution(0.9, 2e-4, fade_sigma=1e-5)
        loose = bms.rul_distribution(0.9, 2e-4, fade_sigma=8e-5)
        assert (loose.p95_cycles - loose.p05_cycles) > (tight.p95_cycles - tight.p05_cycles)

    def test_already_at_eol_is_zero(self):
        d = bms.rul_distribution(0.80, 2e-4, eol=0.8)
        assert d.mean_cycles == 0.0


class TestPredictiveFailure:
    def test_more_precursors_raise_failure_probability(self):
        quiet = bms.predict_failure({"pressure_rise": 0.1})
        loud = bms.predict_failure({"pressure_rise": 0.9, "gas_detected": 0.8,
                                    "resistance_jump": 0.7})
        assert loud.failure_probability > quiet.failure_probability
        assert 0.0 <= loud.failure_probability <= 1.0

    def test_dominant_precursor_and_lead_time(self):
        pred = bms.predict_failure({"pressure_rise": 0.9, "temperature_spread": 0.2},
                                   lead_times_s={"pressure_rise": 600.0})
        assert pred.dominant_precursor == "pressure_rise"
        assert pred.lead_time_s < 600.0             # high severity → less time left
        assert np.isfinite(pred.lead_time_s)

    def test_no_signals_is_low_risk(self):
        pred = bms.predict_failure({})
        assert pred.failure_probability < 0.15
        assert pred.dominant_precursor == "none"
