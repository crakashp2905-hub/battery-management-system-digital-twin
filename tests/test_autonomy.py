"""Tests: the Autonomous Battery Intelligence closed learning loop."""

from __future__ import annotations

import numpy as np

import bms

_FRESH = bms.ECMParameters(R0=0.025, R1=0.015, C1=2000, R2=0.03, C2=8000, Q_nom_Ah=2.3)
_AGED = bms.ECMParameters(R0=0.055, R1=0.024, C1=2000, R2=0.048, C2=8000, Q_nom_Ah=2.02)


def _validation():
    i = np.zeros(1000)
    i[100:900] = 1.5
    return (i, 0.85)


def _twin(seed=3):
    return bms.AutonomousBatteryTwin(
        params=bms.ECMParameters(R0=0.025, R1=0.015, C1=2000, R2=0.03, C2=8000,
                                 Q_nom_Ah=2.3), seed=seed)


class TestAutonomousLoop:
    def test_loop_improves_fidelity_over_rounds(self):
        twin = _twin(seed=3)
        rounds = twin.learn(_AGED, n_rounds=8, validation=_validation())
        assert len(rounds) == 8
        # Held-out fidelity improves substantially from the cold start.
        assert rounds[-1].prediction_rmse_V < rounds[0].prediction_rmse_V
        assert rounds[-1].param_error_after < rounds[0].param_error_before

    def test_validation_gating_is_monotone_nonincreasing(self):
        # The cross-validation gate guarantees the loop never regresses / diverges.
        twin = _twin(seed=7)
        rounds = twin.learn(_AGED, n_rounds=8, validation=_validation())
        vals = [r.prediction_rmse_V for r in rounds]
        assert all(vals[i + 1] <= vals[i] + 1e-9 for i in range(len(vals) - 1))

    def test_recovers_the_identifiable_parameter(self):
        twin = _twin(seed=3)
        twin.learn(_AGED, n_rounds=8, validation=_validation())
        # R0 (well-excited by every current step) is pulled essentially onto truth.
        assert abs(twin.params.R0 - _AGED.R0) < 0.3 * abs(_FRESH.R0 - _AGED.R0)

    def test_loop_targets_its_own_uncertainty(self):
        twin = _twin(seed=3)
        rounds = twin.learn(_AGED, n_rounds=8, validation=_validation())
        # After the first (D-optimal) round the twin steers experiments at the
        # parameters it is least sure of — at least one round has a named target.
        assert any(r.target is not None for r in rounds)
        assert rounds[0].target is None                    # round 0 maximises total info

    def test_degradation_mode_inference(self):
        twin = _twin(seed=3)
        twin.learn(_AGED, n_rounds=8, validation=_validation())
        rep = twin.degradation_report()
        # R0 more than doubled while capacity barely moved → resistance-driven aging.
        assert rep["dominant_mode"] == "resistance_growth"
        assert rep["resistance_growth_pct"] > 50.0
        assert 0.0 <= rep["soh_capacity"] <= 1.05

    def test_round_record_is_serialisable(self):
        twin = _twin(seed=1)
        rnd = twin.step_once(_AGED, validation=_validation())
        d = rnd.to_dict()
        for key in ("round", "target", "experiment", "param_error_after",
                    "prediction_rmse_V"):
            assert key in d

    def test_runs_without_validation(self):
        twin = _twin(seed=2)
        rounds = twin.learn(_AGED, n_rounds=3)                # no held-out gate
        assert len(rounds) == 3
        assert all(np.isnan(r.prediction_rmse_V) for r in rounds)
