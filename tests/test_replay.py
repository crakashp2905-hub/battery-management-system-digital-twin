"""Tests: the digital-twin replay engine."""

from __future__ import annotations

import numpy as np

import bms

_P = bms.ECMParameters(Q_nom_Ah=2.3)


def _trace(seed=0):
    i = np.zeros(1500)
    i[100:500] = 1.5
    i[700:1100] = 2.0
    sim = bms.SecondOrderECM(params=_P, ocv_curve=bms.OCVSOC()).simulate(i, 1.0, soc0=0.9)
    v = sim["v_terminal"] + np.random.default_rng(seed).normal(0, 0.002, len(i))
    return bms.DriveTrace(currents=i, voltages=v, dt=1.0, soc_true=sim["soc"],
                          soc0=0.9, name="drive")


def _estimators():
    return {name: bms.make_soc_estimator(name, params=_P, ocv_curve=bms.OCVSOC(),
                                         capacity_Ah=2.3)
            for name in ("coulomb", "ekf", "ukf")}


class TestReplayEngine:
    def test_replay_scores_a_config(self):
        eng = bms.ReplayEngine(_trace())
        res = eng.replay("ekf", bms.make_soc_estimator("ekf", params=_P,
                                                       ocv_curve=bms.OCVSOC(), capacity_Ah=2.3))
        assert res.config == "ekf"
        assert res.soc_rmse >= 0.0 and np.isfinite(res.soc_rmse)
        assert res.runtime_s > 0.0 and res.n_samples == 1500

    def test_compare_returns_ranked_leaderboard(self):
        board = bms.ReplayEngine(_trace()).compare(_estimators())
        assert len(board) == 3
        rmses = [r.soc_rmse for r in board]
        assert rmses == sorted(rmses)                 # best (lowest RMSE) first
        assert board[0].soc_rmse <= board[-1].soc_rmse

    def test_same_trace_gives_reproducible_ranking(self):
        b1 = bms.ReplayEngine(_trace(1)).compare(_estimators())
        b2 = bms.ReplayEngine(_trace(1)).compare(_estimators())
        assert [r.config for r in b1] == [r.config for r in b2]

    def test_trace_without_truth_still_runs(self):
        t = _trace()
        t.soc_true = None
        res = bms.ReplayEngine(t).replay(
            "coulomb", bms.make_soc_estimator("coulomb", params=_P,
                                              ocv_curve=bms.OCVSOC(), capacity_Ah=2.3))
        assert np.isnan(res.soc_rmse)                 # no ground truth → RMSE undefined
        d = res.to_dict()
        assert d["config"] == "coulomb"
