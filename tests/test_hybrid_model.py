"""Tests: the physics + ML hybrid residual model."""

from __future__ import annotations

import numpy as np

import bms

_P = bms.ECMParameters(Q_nom_Ah=2.3)


def _rich_current(seed, n=2000):
    """A random piecewise current that covers the operating regime."""
    rng = np.random.default_rng(seed)
    seg = rng.choice([-1.5, -0.8, 0.0, 0.8, 1.5, 2.0], size=n // 40)
    return np.repeat(seg, 40)[:n]


def _truth(seed, currents, soc0=0.7):
    """A cell whose voltage carries a smooth residual the 2-RC ECM cannot fit."""
    sim = bms.SecondOrderECM(params=_P, ocv_curve=bms.OCVSOC()).simulate(
        currents, 1.0, soc0=soc0)
    extra = 0.02 * (sim["soc"] - 0.5) + 0.008 * np.tanh(2.0 * currents)
    v = sim["v_terminal"] + extra + np.random.default_rng(seed + 99).normal(0, 0.001, len(currents))
    return v


class TestHybridResidualModel:
    def test_unfitted_hybrid_is_pure_physics(self):
        i = _rich_current(1)
        h = bms.HybridResidualModel(params=_P)
        v_phys = bms.SecondOrderECM(params=_P, ocv_curve=bms.OCVSOC()).simulate(
            i, 1.0, soc0=0.7)["v_terminal"]
        assert np.allclose(h.predict(i, 1.0, soc0=0.7), v_phys)

    def test_hybrid_beats_physics_in_sample(self):
        i = _rich_current(1); v = _truth(1, i)
        h = bms.HybridResidualModel(params=_P).fit(i, v, 1.0, soc0=0.7)
        s = h.score(i, v, 1.0, soc0=0.7)
        assert s["hybrid_rmse_V"] < s["physics_rmse_V"]
        assert s["improvement_pct"] > 40.0                # learns the missed residual

    def test_hybrid_generalises_to_a_holdout_trace(self):
        iA = _rich_current(1); vA = _truth(1, iA)
        iB = _rich_current(2); vB = _truth(2, iB)          # independent, same regime
        h = bms.HybridResidualModel(params=_P).fit(iA, vA, 1.0, soc0=0.7)
        s = h.score(iB, vB, 1.0, soc0=0.7)
        assert s["hybrid_rmse_V"] < s["physics_rmse_V"]    # not just memorising A
        assert s["improvement_pct"] > 30.0

    def test_accepts_a_custom_regressor(self):
        from sklearn.linear_model import Ridge
        i = _rich_current(3); v = _truth(3, i)
        h = bms.HybridResidualModel(params=_P, regressor=Ridge(alpha=1.0))
        h.fit(i, v, 1.0, soc0=0.7)
        pred = h.predict(i, 1.0, soc0=0.7)
        assert pred.shape == i.shape and np.all(np.isfinite(pred))
