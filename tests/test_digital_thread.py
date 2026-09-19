"""Tests: the manufacturing → field digital thread."""

from __future__ import annotations

import numpy as np

import bms

_USAGE = bms.FieldUsage(c_rate=1.0, temperature_C=35.0, dod=0.8, soc_avg=0.6,
                        cycles_per_day=1.0)


def _nominal(cell_id="A"):
    return bms.ManufacturingRecord(cell_id=cell_id, formation_efficiency=0.93,
                                   initial_capacity_Ah=2.30, nominal_capacity_Ah=2.30,
                                   initial_r0_ohm=0.025, nominal_r0_ohm=0.025)


def _poor(cell_id="B"):
    return bms.ManufacturingRecord(cell_id=cell_id, formation_efficiency=0.85,
                                   initial_capacity_Ah=2.20, nominal_capacity_Ah=2.30,
                                   initial_r0_ohm=0.040, nominal_r0_ohm=0.025)


class TestFormationCoupling:
    def test_poor_formation_raises_aging_multipliers(self):
        good = bms.link_formation_to_aging(_nominal())
        bad = bms.link_formation_to_aging(_poor())
        assert good.capacity_fade_mult == 1.0 or abs(good.capacity_fade_mult - 1.0) < 1e-6
        assert bad.capacity_fade_mult > good.capacity_fade_mult    # low CE → faster fade
        assert bad.resistance_growth_mult > good.resistance_growth_mult  # high R0 → faster

    def test_poor_cell_reaches_eol_sooner(self):
        good = bms.project_field_trajectory(_nominal(), _USAGE)
        bad = bms.project_field_trajectory(_poor(), _USAGE)
        assert bad.cycles_to_eol < good.cycles_to_eol
        assert bad.start_soh < good.start_soh          # lower capacity grade at birth
        assert bad.days_to_eol < good.days_to_eol


class TestDigitalThread:
    def test_rank_puts_worst_formed_cell_first(self):
        thread = bms.DigitalThread(records=[_nominal("good"), _poor("bad"),
                                            _nominal("good2")])
        ranked = thread.rank_field_risk(_USAGE)
        assert ranked[0].cell_id == "bad"             # highest field risk first

    def test_life_distribution_reports_b10_and_worst(self):
        rng = np.random.default_rng(0)
        records = [bms.ManufacturingRecord(
            cell_id=f"c{i}",
            formation_efficiency=float(np.clip(rng.normal(0.92, 0.02), 0.80, 0.95)),
            initial_capacity_Ah=float(rng.normal(2.28, 0.03)),
            nominal_capacity_Ah=2.30,
            initial_r0_ohm=float(np.clip(rng.normal(0.026, 0.004), 0.02, 0.05)),
            nominal_r0_ohm=0.025) for i in range(30)]
        dist = bms.DigitalThread(records=records).life_distribution(_USAGE)
        assert dist["n"] == 30
        assert dist["b10_cycles"] <= dist["mean_cycles"]
        assert dist["min_cycles"] <= dist["b10_cycles"]
        assert "worst_cell" in dist

    def test_formation_predicts_life_positive_correlation(self):
        rng = np.random.default_rng(1)
        records = [bms.ManufacturingRecord(
            cell_id=f"c{i}",
            formation_efficiency=float(np.clip(rng.uniform(0.83, 0.94), 0.80, 0.95)),
            initial_capacity_Ah=2.30, nominal_capacity_Ah=2.30,
            initial_r0_ohm=0.025, nominal_r0_ohm=0.025) for i in range(40)]
        corr = bms.DigitalThread(records=records).formation_life_correlation(_USAGE)
        # Better formation efficiency → longer projected life: the thread's claim.
        assert corr > 0.8

    def test_record_and_projection_serialise(self):
        proj = bms.project_field_trajectory(_poor(), _USAGE)
        d = proj.to_dict()
        for key in ("cell_id", "cycles_to_eol", "days_to_eol",
                    "capacity_fade_mult", "start_soh"):
            assert key in d
        assert abs(_poor().capacity_grade - 2.20 / 2.30) < 1e-9
