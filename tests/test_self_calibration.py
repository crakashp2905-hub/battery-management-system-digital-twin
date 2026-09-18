"""Tests: the observability-gated Self-Calibrating Twin."""

from __future__ import annotations

import numpy as np

import bms

_TRUE = bms.ECMParameters(R0=0.025, R1=0.015, C1=2000, R2=0.03, C2=8000, Q_nom_Ah=2.3)
_AGED = bms.ECMParameters(R0=0.055, R1=0.022, C1=2000, R2=0.045, C2=8000, Q_nom_Ah=2.05)


def _measure(params, soc0=0.85, n=1500, seed=0, noise=0.002):
    """A dynamic (identifiable) trace measured from `params`, model-agnostic."""
    ecm = bms.SecondOrderECM(params=params, ocv_curve=bms.OCVSOC())
    i = np.zeros(n)
    i[100:400] = 2.0
    i[500:700] = -1.5
    i[900:1400] = 1.5
    sim = ecm.simulate(i, 1.0, soc0=soc0)
    v = sim["v_terminal"] + np.random.default_rng(seed).normal(0.0, noise, n)
    return i, v


class TestSelfCalibratingTwin:
    def test_matched_model_needs_no_calibration(self):
        i, v = _measure(_TRUE)
        twin = bms.SelfCalibratingTwin(params=_TRUE)
        assert twin.needs_calibration(i, v, 1.0, soc0=0.85) is False

    def test_recovers_aged_parameters_from_dynamic_trace(self):
        i, v = _measure(_AGED)                              # aged cell measured
        twin = bms.SelfCalibratingTwin(params=_TRUE)        # fresh-parameter twin
        assert twin.needs_calibration(i, v, 1.0, soc0=0.85) is True
        res = twin.calibrate(i, v, 1.0, soc0=0.85)
        assert res.improved is True
        assert res.rmse_after_V < res.rmse_before_V
        # R0 and capacity are pulled toward the aged truth.
        assert abs(twin.params.R0 - _AGED.R0) < abs(_TRUE.R0 - _AGED.R0)
        assert abs(twin.params.Q_nom_Ah - _AGED.Q_nom_Ah) < abs(_TRUE.Q_nom_Ah - _AGED.Q_nom_Ah)
        assert "R0" in res.calibrated and "Q_Ah" in res.calibrated

    def test_rest_trace_refuses_to_recalibrate_unobservable_params(self):
        # A rest carries no information about R0/R1/R2/Q — the gate must skip them.
        i = np.zeros(400)
        ecm = bms.SecondOrderECM(params=_AGED, ocv_curve=bms.OCVSOC())
        v = ecm.simulate(i, 1.0, soc0=0.6)["v_terminal"] + 0.001
        twin = bms.SelfCalibratingTwin(params=_TRUE)
        res = twin.calibrate(i, v, 1.0, soc0=0.6)
        for p in ("R0", "R1", "R2", "Q_Ah"):
            assert p in res.skipped_unidentifiable          # not fixed from a rest
            assert p not in res.calibrated
        # R0 must remain untouched — the twin did not invent a value it cannot see.
        assert twin.params.R0 == _TRUE.R0

    def test_calibration_reports_deltas_for_changed_params(self):
        i, v = _measure(_AGED)
        twin = bms.SelfCalibratingTwin(params=_TRUE)
        res = twin.calibrate(i, v, 1.0, soc0=0.85)
        assert set(res.deltas) == set(res.calibrated)
        assert res.deltas["R0"] > 0.0                       # R0 grew toward the aged value
        d = res.to_dict()
        assert d["improved"] is True and "deltas" in d
