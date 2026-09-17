"""Tests: Single Particle Model (electrochemical fidelity)."""

from __future__ import annotations

import numpy as np

from bms.spm import SingleParticleModel, SPMParams


class TestSPM:
    def test_full_discharge_monotonic(self):
        m = SingleParticleModel(SPMParams(Q_nom_Ah=2.3))
        out = m.simulate(np.full(3000, 2.3), 1.0, soc0=1.0)
        assert out["soc"][0] > 0.98 and out["soc"][-1] < 0.1     # discharges fully
        # Terminal voltage decreases overall and stays in a physical window.
        assert out["v_terminal"][0] > out["v_terminal"][-1]
        assert np.all((out["v_terminal"] > 2.0) & (out["v_terminal"] < 4.4))

    def test_diffusion_limitation_surface_below_bulk(self):
        # Under load the particle *surface* is more depleted than the bulk, and
        # more so at higher C-rate — the physics an ECM cannot express.
        def gap(c_rate):
            m = SingleParticleModel(SPMParams(Q_nom_Ah=2.3))
            o = m.simulate(np.full(400, c_rate * 2.3), 1.0, soc0=0.9)
            return o["soc"][-1] - o["surface_soc"][-1]
        gap_slow, gap_fast = gap(0.5), gap(3.0)
        assert gap_fast > gap_slow > 0.0
        assert gap_fast > 0.1                                     # pronounced at 3C

    def test_voltage_relaxes_on_rest(self):
        m = SingleParticleModel(SPMParams(Q_nom_Ah=2.3))
        m.reset(0.8)
        m.simulate(np.full(200, 3 * 2.3), 1.0)                    # 3C pulse
        v_loaded = m.terminal_voltage(0.0)
        v_rested = m.simulate(np.zeros(600), 1.0)["v_terminal"][-1]
        assert v_rested > v_loaded + 0.05                         # gradient relaxes upward

    def test_lower_voltage_at_higher_rate(self):
        def v_after(c_rate):
            m = SingleParticleModel(SPMParams(Q_nom_Ah=2.3))
            return m.simulate(np.full(300, c_rate * 2.3), 1.0, soc0=0.9)["v_terminal"][-1]
        assert v_after(3.0) < v_after(0.5)                        # rate capability

    def test_soc_conserved_by_coulomb_counting(self):
        # The bulk SoC change must equal the ampere-second integral (conservation).
        m = SingleParticleModel(SPMParams(Q_nom_Ah=2.3))
        i = np.full(500, 1.15)                                    # 0.5C
        out = m.simulate(i, 1.0, soc0=0.9)
        expected = 0.9 - i.sum() * 1.0 / (2.3 * 3600.0)
        assert abs(out["soc"][-1] - expected) < 0.01
