"""Tests: online data assimilation / digital-twin sync."""

from __future__ import annotations

import numpy as np

import bms


def _trace(params, soc0=0.9, seed=0, noise=0.003):
    ecm = bms.SecondOrderECM(params=params, ocv_curve=bms.OCVSOC())
    ecm.reset(soc0)
    n = 1500
    i = np.zeros(n)
    i[100:500] = 1.5
    i[700:1100] = 2.0
    sim = ecm.simulate(i, 1.0)
    v = sim["v_terminal"] + np.random.default_rng(seed).normal(0.0, noise, n)
    return i, v, sim["soc"]


_TRUE = bms.ECMParameters(R0=0.025, R1=0.015, C1=2000, R2=0.03, C2=8000, Q_nom_Ah=2.3)
_AGED = bms.ECMParameters(R0=0.060, R1=0.025, C1=2000, R2=0.05, C2=8000, Q_nom_Ah=2.1)


class TestTwinSync:
    def test_matched_twin_tracks_and_does_not_drift(self):
        i, v, soc_true = _trace(_TRUE)
        tw = bms.TwinSync(params=_TRUE, ocv_curve=bms.OCVSOC())
        tw.reset(0.90)
        out = tw.run(i, v, 1.0)
        assert abs(out["soc"][-1] - soc_true[-1]) < 0.01     # tracks truth
        assert out["drift"] is False                          # model matches → no drift

    def test_corrects_a_wrong_initial_soc(self):
        i, v, soc_true = _trace(_TRUE)
        tw = bms.TwinSync(params=_TRUE, ocv_curve=bms.OCVSOC())
        tw.reset(0.75)                                        # 15% off truth (0.90)
        out = tw.run(i, v, 1.0)
        # The assimilation pulls SoC back toward truth by the end.
        assert abs(out["soc"][-1] - soc_true[-1]) < abs(0.75 - soc_true[0])

    def test_mismatched_model_is_flagged_as_drift(self):
        # Feed an aged cell (R0 more than doubled) to a fresh-model twin.
        i, v, _ = _trace(_AGED, seed=1)
        tw = bms.TwinSync(params=_TRUE, ocv_curve=bms.OCVSOC())
        tw.reset(0.90)
        assert tw.run(i, v, 1.0)["drift"] is True             # model no longer matches

    def test_zero_gain_is_open_loop_coulomb(self):
        i, v, _ = _trace(_TRUE)
        tw = bms.TwinSync(params=_TRUE, ocv_curve=bms.OCVSOC(), correction_gain=0.0)
        tw.reset(0.9)
        out = tw.run(i, v, 1.0)
        expected = np.clip(0.9 - np.cumsum(i) / (_TRUE.Q_nom_Ah * 3600.0), 0.0, 1.0)
        assert np.allclose(out["soc"], expected, atol=1e-6)   # no correction applied

    def test_residual_and_prediction_exposed(self):
        i, v, _ = _trace(_TRUE)
        tw = bms.TwinSync(params=_TRUE, ocv_curve=bms.OCVSOC())
        tw.reset(0.9)
        out = tw.assimilate(1.5, float(v[120]), 1.0)
        assert {"soc", "predicted_voltage_V", "residual_V", "residual_rms_V",
                "drift"} <= set(out)
        assert 2.5 < out["predicted_voltage_V"] < 4.3
