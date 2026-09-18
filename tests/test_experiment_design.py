"""Tests: the Autonomous Experiment Designer (optimal, safety-gated excitation)."""

from __future__ import annotations

import numpy as np

import bms

_P = bms.ECMParameters(R0=0.025, R1=0.015, C1=2000, R2=0.03, C2=8000, Q_nom_Ah=2.3)


class TestExperimentDesigner:
    def test_library_scales_current_to_capacity(self):
        lib = bms.experiment_library(capacity_Ah=2.3, dt=1.0)
        names = {e.name for e in lib}
        assert {"rest", "cc_discharge_1C", "hppc"} <= names
        one_c = next(e for e in lib if e.name == "cc_discharge_1C")
        assert np.isclose(np.max(one_c.currents), 2.3)     # 1C of a 2.3 Ah cell

    def test_designer_prefers_an_informative_excitation_over_rest(self):
        design = bms.design_next_experiment(_P, soc0=0.7, chemistry="nmc")
        assert design.best is not None
        assert design.best.name != "rest"                  # rest carries least info
        # ranking is sorted best-first and rest is not on top
        assert design.ranking[0][1] >= design.ranking[-1][1]

    def test_targeting_capacity_returns_a_current_moving_experiment(self):
        design = bms.design_next_experiment(_P, soc0=0.7, target="Q_Ah")
        assert design.best is not None and design.criterion == "target:Q_Ah"
        assert design.best.name != "rest"                  # capacity needs SoC to move

    def test_safety_filter_rejects_discharge_at_low_soc(self):
        # Near-empty cell: aggressive discharges would cross v_min and are rejected;
        # a charge experiment survives and should be chosen.
        design = bms.design_next_experiment(_P, soc0=0.03, chemistry="nmc")
        assert "cc_discharge_2C" in design.rejected_unsafe
        assert design.best is not None
        assert design.best.currents.min() < 0              # a charge pulse won

    def test_current_limit_filters_high_c_experiments(self):
        design = bms.design_next_experiment(_P, soc0=0.7, i_max_A=1.0)   # ≤ ~0.43C
        assert "cc_discharge_2C" in design.rejected_unsafe
        assert "cc_discharge_1C" in design.rejected_unsafe

    def test_information_gain_orders_gentle_below_strong(self):
        cap = _P.Q_nom_Ah
        gentle = bms.Experiment("g", np.full(600, 0.3 * cap), 1.0, "")
        strong = bms.Experiment("s", np.full(600, 2.0 * cap), 1.0, "")
        g = bms.expected_information_gain(gentle, _P, 0.8)
        s = bms.expected_information_gain(strong, _P, 0.8)
        assert s > g
