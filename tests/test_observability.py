"""Tests: the Battery Observability Engine (Fisher information / CRLB)."""

from __future__ import annotations

import numpy as np

import bms

_P = bms.ECMParameters(R0=0.025, R1=0.015, C1=2000, R2=0.03, C2=8000, Q_nom_Ah=2.3)


def _pulse_train(n=1500):
    """A dynamic HPPC-like profile: strong current steps + rest, big SoC swing."""
    i = np.zeros(n)
    i[100:400] = 2.0        # discharge pulse
    i[500:700] = -1.5       # charge pulse
    i[900:1400] = 1.5       # long discharge (moves SoC → excites capacity)
    return i


class TestObservabilityEngine:
    def test_dynamic_trajectory_identifies_r0_and_capacity(self):
        i = _pulse_train()
        rep = bms.analyze_observability(_P, i, 1.0, soc0=0.8)
        assert rep.identifiable["R0"] is True         # current steps reveal R0
        assert rep.identifiable["Q_Ah"] is True       # SoC swing reveals capacity
        assert rep.crlb["R0"] < 0.1
        assert np.isfinite(rep.d_opt)

    def test_rest_makes_resistance_and_capacity_unobservable(self):
        i = np.zeros(400)                              # a pure rest — no current
        rep = bms.analyze_observability(_P, i, 1.0, soc0=0.6)
        # No current → no ohmic drop → R0/R1/R2 carry no information.
        assert rep.identifiable["R0"] is False
        assert rep.identifiable["Q_Ah"] is False       # SoC never moves
        assert not np.isfinite(rep.crlb["R0"])
        assert rep.condition_number == float("inf") or rep.e_opt <= 1e-9

    def test_larger_current_carries_more_information(self):
        eng = bms.ObservabilityEngine(params=_P)
        gentle = np.zeros(1500); gentle[100:1400] = 0.3
        strong = np.zeros(1500); strong[100:1400] = 2.5
        d_gentle = eng.information(gentle, 1.0, soc0=0.8)
        d_strong = eng.information(strong, 1.0, soc0=0.8)
        assert d_strong > d_gentle                     # more excitation → more info

    def test_lower_sensor_noise_tightens_crlb(self):
        i = _pulse_train()
        loud = bms.analyze_observability(_P, i, 1.0, soc0=0.8, sigma_v=0.02)
        quiet = bms.analyze_observability(_P, i, 1.0, soc0=0.8, sigma_v=0.002)
        assert quiet.crlb["R0"] < loud.crlb["R0"]      # better sensor → tighter bound

    def test_report_surfaces_worst_identified_parameter(self):
        i = _pulse_train()
        rep = bms.analyze_observability(_P, i, 1.0, soc0=0.8)
        worst = rep.most_uncertain()
        assert worst in rep.params
        # The worst CRLB really is the maximum over the set.
        assert rep.crlb[worst] == max(rep.crlb.values())
        d = rep.to_dict()
        assert d["most_uncertain"] == worst and "identifiable" in d

    def test_sensitivities_shape_and_zero_at_rest(self):
        i = np.zeros(300)
        S, names = bms.voltage_sensitivities(_P, i, 1.0, soc0=0.5)
        assert S.shape == (300, len(names))
        j_r0 = names.index("R0")
        assert np.allclose(S[:, j_r0], 0.0)            # R0 has no effect at I=0
        j_soc = names.index("soc0")
        assert np.any(np.abs(S[:, j_soc]) > 0.0)       # initial SoC still shows in OCV
