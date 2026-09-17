"""Tests: particle-filter SoC estimator + EIS/DRT analysis."""

from __future__ import annotations

import numpy as np

import bms


class TestParticleFilter:
    def test_registered_and_protocol(self):
        assert "pf" in bms.available_soc_estimators()
        est = bms.make_soc_estimator("pf", params=bms.ECMParameters(), ocv_curve=bms.OCVSOC())
        assert isinstance(est, bms.RecursiveSocEstimator)

    def test_tracks_soc_and_reports_uncertainty(self):
        p = bms.ECMParameters(R0=0.025, R1=0.012, C1=2500, R2=0.02, C2=9000, Q_nom_Ah=2.3)
        ecm = bms.SecondOrderECM(params=p, ocv_curve=bms.OCVSOC()); ecm.reset(0.85)
        i = np.zeros(1200); i[100:600] = 1.5; i[800:1100] = 2.0
        sim = ecm.simulate(i, 1.0)
        v = sim["v_terminal"] + np.random.default_rng(0).normal(0, 0.003, 1200)
        est = bms.ParticleFilterEstimator(params=p, ocv_curve=bms.OCVSOC()); est.reset(0.85)
        soc = est.run(i, v, 1.0)
        assert np.all((soc >= 0) & (soc <= 1))
        assert abs(soc[-1] - sim["soc"][-1]) < 0.02
        assert est.soc_uncertainty_1sigma >= 0.0

    def test_beats_ekf_on_flat_ocv_lfp(self):
        # The PF's non-Gaussian posterior helps most where the OCV is flat (LFP).
        d = bms.synthetic_drivecycle("lfp", duration_s=1500, seed=1)
        b = bms.estimator_leaderboard(d, ["ekf", "pf"])
        assert b.loc["pf", "rmse"] < b.loc["ekf", "rmse"]


class TestEISAnalysis:
    def _spectrum(self, R0=0.02, R1=0.015, C1=2000):
        p = bms.ECMParameters(R0=R0, R1=R1, C1=C1, R2=1e-9, C2=1.0, Q_nom_Ah=2.3)
        return bms.simulate_eis(p, frequencies_Hz=np.logspace(-2, 4, 120))

    def test_drt_recovers_the_time_constant_order(self):
        freq, zre, zneg = self._spectrum(R1=0.015, C1=2000)   # τ = 30 s
        tau, gamma = bms.compute_drt(freq, zneg, n_tau=80, reg=1e-2)
        assert np.all(gamma >= 0.0)                            # non-negative DRT
        tau_peak = tau[int(np.argmax(gamma))]
        # DRT peaks are broadened; require the right order of magnitude.
        assert 30.0 / 5 < tau_peak < 30.0 * 5

    def test_eis_resistances_recovers_ohmic_intercept(self):
        freq, zre, zneg = self._spectrum(R0=0.02)
        r = bms.eis_resistances(freq, zre, zneg)
        assert abs(r["R0"] - 0.02) < 2e-3          # ohmic intercept recovered
        assert r["R_ct"] >= 0.0

    def test_eis_soh_drops_with_resistance_growth(self):
        freq, zre, zneg = self._spectrum(R0=0.02, R1=0.015)
        _, zra, zna = self._spectrum(R0=0.035, R1=0.030)       # aged: R grown
        soh = bms.eis_soh(freq, zre, zneg, zra, zna)
        assert soh["resistance_ratio"] > 1.0                   # resistance grew
        assert soh["R0_growth"] > 1.0
        assert 0.0 <= soh["soh_impedance"] < 1.0               # SoH dropped
