"""Tests: online RLS parameter identification."""

from __future__ import annotations

import numpy as np

import bms


def _stepped_trace(true_R0: float, seed: int = 0):
    p = bms.ECMParameters(R0=true_R0, R1=0.012, C1=2500, R2=0.02, C2=9000,
                          Q_nom_Ah=2.3)
    ecm = bms.SecondOrderECM(params=p, ocv_curve=bms.OCVSOC())
    ecm.reset(0.7)
    n = 1500
    i = np.zeros(n)
    i[100:300] = 2.3          # discharge step
    i[500:700] = -2.3         # charge step
    i[900:1200] = 1.15        # sustained
    sim = ecm.simulate(i, 1.0)
    v = sim["v_terminal"] + np.random.default_rng(seed).normal(0.0, 0.002, n)
    return i, v


class TestRLSIdentifier:
    def test_recovers_r0_from_steps(self):
        i, v = _stepped_trace(0.028)
        rls = bms.RLSIdentifier(r0_init=0.015)     # start deliberately off
        rls.run(i, v)
        assert abs(rls.r0 - 0.028) < 0.004          # recovers R0 to a few mΩ

    def test_tracks_a_change_in_r0(self):
        # Cold (high R0) first half, warm (low R0) second half.
        i1, v1 = _stepped_trace(0.040, seed=1)
        i2, v2 = _stepped_trace(0.020, seed=2)
        rls = bms.RLSIdentifier(r0_init=0.040, forgetting=0.99)
        rls.run(i1, v1)
        assert abs(rls.r0 - 0.040) < 0.006
        before = rls.r0
        rls.run(i2, v2)
        assert rls.r0 < before - 0.005               # followed R0 down materially

    def test_coasts_when_current_is_flat(self):
        # No current steps → no information → estimate must not wander.
        rls = bms.RLSIdentifier(r0_init=0.025)
        v = 3.7 + np.random.default_rng(0).normal(0.0, 0.002, 500)
        rls.run(np.full(500, 1.0), v)               # constant current
        assert abs(rls.r0 - 0.025) < 1e-3            # essentially unchanged

    def test_resistance_stays_positive(self):
        i, v = _stepped_trace(0.03)
        rls = bms.RLSIdentifier(r0_init=0.03)
        traj = rls.run(i, v)
        assert np.all(traj > 0.0)
