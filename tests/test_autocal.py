"""Tests: universal auto-calibration (identify any cell from data)."""

from __future__ import annotations

import numpy as np

import bms


def _synth_cell(chem, R0=0.03, n=1400, seed=0):
    """A synthetic 'unknown' cell of a given chemistry with a dynamic trace."""
    p = bms.ECMParameters(R0=R0, R1=0.02, C1=2500, R2=0.04, C2=9000,
                          Q_nom_Ah=2.3, chemistry=chem)
    ocv = bms.OCVSOC.from_chemistry(chem)
    i = np.zeros(n)
    i[50:400] = 1.5; i[500:800] = -1.0; i[900:1350] = 1.2
    sim = bms.SecondOrderECM(params=p, ocv_curve=ocv).simulate(i, 1.0, soc0=0.9)
    v = sim["v_terminal"] + np.random.default_rng(seed).normal(0, 0.002, n)
    return i, v, sim["soc"], p


class TestAutoCalibrate:
    def test_identifies_the_correct_chemistry_template(self):
        # An unknown LFP cell (flat OCV) should be matched to the LFP template.
        i, v, _, _ = _synth_cell("lfp")
        cal = bms.auto_calibrate(i, v, 1.0, capacity_Ah=2.3)
        assert cal.chemistry == "lfp"
        assert cal.voltage_rmse_V < 0.02
        assert cal.candidates["lfp"] <= min(cal.candidates.values()) + 1e-9

    def test_calibrated_cell_reconstructs_voltage(self):
        i, v, _, _ = _synth_cell("nmc")
        cal = bms.auto_calibrate(i, v, 1.0, capacity_Ah=2.3)
        v_pred = cal.predicted_voltage(i, 1.0)
        assert np.sqrt(np.mean((v_pred - v) ** 2)) < 0.02

    def test_calibrated_estimator_tracks_soc(self):
        i, v, soc_true, _ = _synth_cell("nca")
        cal = bms.auto_calibrate(i, v, 1.0, capacity_Ah=2.3)
        est = cal.make_estimator("ekf")
        est.reset(0.9)
        soc = np.asarray(est.run(i, v, 1.0), float)
        assert np.sqrt(np.mean((soc - soc_true) ** 2)) < 0.05

    def test_estimate_capacity_from_full_discharge(self):
        # 2 A for 3600 s from full to empty ≈ 2.0 Ah.
        i = np.full(3600, 2.0)
        q = bms.estimate_capacity(i, 1.0, soc0=1.0, soc_end=0.0)
        assert abs(q - 2.0) < 0.05

    def test_serialises_and_reports_candidates(self):
        i, v, _, _ = _synth_cell("lmo")
        cal = bms.auto_calibrate(i, v, 1.0, capacity_Ah=2.3)
        d = cal.to_dict()
        assert set(("chemistry", "capacity_Ah", "voltage_rmse_V", "candidates")) <= set(d)
        assert len(cal.candidates) == 7            # all templates were tried
