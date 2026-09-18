"""Tests: the central BatteryDigitalTwin — one uncertainty-aware state."""

from __future__ import annotations

import numpy as np

import bms

_TRUE = bms.ECMParameters(R0=0.025, R1=0.015, C1=2000, R2=0.03, C2=8000, Q_nom_Ah=2.3)
_AGED = bms.ECMParameters(R0=0.060, R1=0.025, C1=2000, R2=0.05, C2=8000, Q_nom_Ah=2.1)


def _drive(params, soc0=0.9, n=1500, seed=0, noise=0.002):
    """A trace with real SoC excitation (≈ 28 % swing) held under load to the end
    so capacity stays observable at the final sample."""
    ecm = bms.SecondOrderECM(params=params, ocv_curve=bms.OCVSOC())
    ecm.reset(soc0)
    i = np.zeros(n)
    i[100:700] = 1.5
    i[800:1500] = 2.0
    sim = ecm.simulate(i, 1.0)
    v = sim["v_terminal"] + np.random.default_rng(seed).normal(0.0, noise, n)
    return i, v, sim["soc"]


def _rest(params, soc0=0.7, n=400, seed=0, noise=0.002):
    """A rested trace (no current) — no excitation, capacity unobservable."""
    ecm = bms.SecondOrderECM(params=params, ocv_curve=bms.OCVSOC())
    ecm.reset(soc0)
    i = np.zeros(n)
    sim = ecm.simulate(i, 1.0)
    v = sim["v_terminal"] + np.random.default_rng(seed).normal(0.0, noise, n)
    return i, v, sim["soc"]


class TestBatteryDigitalTwin:
    def test_matched_twin_tracks_and_is_confident(self):
        i, v, soc_true = _drive(_TRUE)
        tw = bms.BatteryDigitalTwin(params=_TRUE)
        tw.reset(0.9)
        states = tw.run(i, v, 1.0)
        end = states[-1]
        assert abs(end.soc - soc_true[-1]) < 0.03          # tracks truth
        assert end.drift is False                           # model matches
        assert end.capacity_observable is True              # 28 % swing excites capacity
        assert end.soc_confidence > 0.6                     # SoC is well identified
        assert end.confidence > 0.4                         # overall: a trusted twin
        assert 0.85 < end.soh < 1.10                        # fresh cell ≈ full SoH

    def test_run_returns_snapshots_and_tracks_latest_state(self):
        i, v, _ = _drive(_TRUE)
        tw = bms.BatteryDigitalTwin(params=_TRUE)
        tw.reset(0.9)
        states = tw.run(i, v, 1.0)
        assert len(states) == len(i)
        assert states[-1] is tw.state
        assert states[-1].n_updates == len(i)

    def test_mismatched_cell_flags_drift_and_lowers_confidence(self):
        im, vm, _ = _drive(_TRUE)
        tw_ok = bms.BatteryDigitalTwin(params=_TRUE); tw_ok.reset(0.9)
        ok = tw_ok.run(im, vm, 1.0)

        ia, va, _ = _drive(_AGED, seed=1)              # aged cell, fresh-model twin
        tw_bad = bms.BatteryDigitalTwin(params=_TRUE); tw_bad.reset(0.9)
        bad = tw_bad.run(ia, va, 1.0)

        assert not any(s.drift for s in ok)             # matched model never drifts
        assert any(s.drift for s in bad)                # mismatched cell trips drift
        # The residual carries the unmodelled resistance growth, and mean trust falls.
        resid_ok = float(np.mean([s.residual_rms_V for s in ok]))
        resid_bad = float(np.mean([s.residual_rms_V for s in bad]))
        assert resid_bad > 2.0 * resid_ok
        assert (np.mean([s.confidence for s in bad])
                < np.mean([s.confidence for s in ok]))

    def test_cold_start_is_low_confidence(self):
        i, v, _ = _drive(_TRUE)
        tw = bms.BatteryDigitalTwin(params=_TRUE, warmup_steps=20)
        tw.reset(0.9)
        first = tw.update(float(v[0]), float(i[0]), 1.0)
        assert first.confidence < 0.2                   # not warmed up yet
        assert first.n_updates == 1

    def test_rest_makes_capacity_unobservable(self):
        i, v, _ = _rest(_TRUE)
        tw = bms.BatteryDigitalTwin(params=_TRUE)
        tw.reset(0.7)
        end = tw.run(i, v, 1.0)[-1]
        assert end.capacity_observable is False         # no SoC movement
        assert end.soh_confidence < 0.05                # capacity not identifiable
        assert end.excitation < tw.excitation_ref

    def test_state_credible_intervals_and_to_dict(self):
        i, v, _ = _drive(_TRUE)
        tw = bms.BatteryDigitalTwin(params=_TRUE)
        tw.reset(0.9)
        end = tw.run(i, v, 1.0)[-1]
        lo, hi = end.soc_95_ci
        assert 0.0 <= lo <= end.soc <= hi <= 1.0        # bracketed and clipped
        slo, shi = end.soh_95_ci
        assert slo <= end.soh <= shi
        d = end.to_dict()
        for key in ("soc", "soc_95_ci", "soh", "soh_95_ci", "confidence",
                    "r0_ohm", "drift", "capacity_observable"):
            assert key in d
        assert d["soc_95_ci"] == list(end.soc_95_ci)

    def test_online_resistance_is_tracked(self):
        i, v, _ = _drive(_TRUE)
        tw = bms.BatteryDigitalTwin(params=_TRUE)
        tw.reset(0.9)
        end = tw.run(i, v, 1.0)[-1]
        assert 0.0 < end.r0_ohm < 0.2                   # a plausible ohmic resistance
