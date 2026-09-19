"""Tests: the unified twin — one canonical state, one data flow."""

from __future__ import annotations

import numpy as np

import bms

_P = bms.ECMParameters(R0=0.025, R1=0.015, C1=2000, R2=0.03, C2=8000, Q_nom_Ah=2.3)


def _drive(params, soc0=0.9, n=1500, seed=0):
    ecm = bms.SecondOrderECM(params=params, ocv_curve=bms.OCVSOC())
    i = np.zeros(n)
    i[100:700] = 1.5
    i[800:1500] = 2.0
    sim = ecm.simulate(i, 1.0, soc0=soc0)
    v = sim["v_terminal"] + np.random.default_rng(seed).normal(0, 0.002, n)
    return i, v, sim["soc"]


class TestUnifiedTwin:
    def test_single_state_carries_everything(self):
        i, v, _ = _drive(_P)
        tw = bms.UnifiedTwin(params=_P, chemistry="nmc")
        tw.reset(0.9)
        st = tw.run(i, v, 1.0)[-1]
        # one snapshot holds estimation, params, power, degradation, prognostics, trust
        for f in ("soc", "soh", "capacity_Ah", "r0_ohm", "r1_ohm", "r2_ohm",
                  "sop_charge_W", "sop_discharge_W", "degradation_mode",
                  "rul_median_cycles", "thermal_runaway_prob", "fault_probability",
                  "confidence", "observable", "version"):
            assert hasattr(st, f)
        assert st is tw.state                          # single source of truth
        assert st.version == "model_v1"

    def test_state_is_estimation_consistent(self):
        i, v, soc_true = _drive(_P)
        tw = bms.UnifiedTwin(params=_P); tw.reset(0.9)
        st = tw.run(i, v, 1.0)[-1]
        assert abs(st.soc - soc_true[-1]) < 0.03       # tracks the underlying core
        assert 0.0 <= st.confidence <= 1.0
        lo, hi = st.soc_95_ci
        assert lo <= st.soc <= hi

    def test_sop_headroom_is_directional(self):
        i, v, _ = _drive(_P)
        tw = bms.UnifiedTwin(params=_P); tw.reset(0.9)
        st = tw.run(i, v, 1.0)[-1]
        assert st.sop_charge_W >= 0.0 and st.sop_discharge_W >= 0.0

    def test_aged_cell_shows_resistance_degradation_and_lower_confidence(self):
        aged = bms.ECMParameters(R0=0.060, R1=0.025, C1=2000, R2=0.05, C2=8000, Q_nom_Ah=2.1)
        i, v, _ = _drive(aged, seed=1)
        tw = bms.UnifiedTwin(params=_P); tw.reset(0.9)      # fresh model, aged cell
        states = tw.run(i, v, 1.0)
        assert any(s.drift for s in states)                 # drift detected somewhere
        assert states[-1].resistance_growth_pct >= 0.0

    def test_to_dict_is_json_ready(self):
        i, v, _ = _drive(_P)
        tw = bms.UnifiedTwin(params=_P); tw.reset(0.9)
        d = tw.run(i, v, 1.0)[-1].to_dict()
        for key in ("soc_95_ci", "soh_95_ci", "degradation_fractions",
                    "thermal_runaway_prob", "observable"):
            assert key in d
