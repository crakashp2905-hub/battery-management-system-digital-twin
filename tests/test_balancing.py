"""Tests: balancing."""

from __future__ import annotations

import numpy as np
import pytest

import bms


class TestBalancers:
    def _factory(self, seed=7):
        return lambda: bms.BatteryPack(bms.PackConfig(
            n_cells=4, initial_soc_sigma=0.10, seed=seed))

    def test_inductor_converges(self):
        pack = self._factory()()
        initial_imbalance = pack.soc_imbalance()
        bal = bms.InductorBalancer(max_current_A=2.0, gain=10.0)
        for _ in range(2_000):
            cur = bal.step(pack, 1.0)
            pack.step(0.0, 1.0, balancing_currents=cur)
        # With 3% capacity scatter, the steady-state isn't perfect SOC
        # equality but a much-reduced spread. The system should bring
        # imbalance to <1% from initial ~13%.
        assert pack.soc_imbalance() < 0.01
        assert pack.soc_imbalance() < 0.1 * initial_imbalance

    def test_passive_dissipates_energy(self):
        pack = self._factory()()
        bal = bms.PassiveBalancer()
        for _ in range(60):
            cur = bal.step(pack, 1.0)
            pack.step(0.0, 1.0, balancing_currents=cur)
        # Imbalance must shrink and dissipative loss must accumulate.
        assert bal.energy_loss_J > 0

    def test_compare_returns_dataframe(self):
        df = bms.compare_balancers(self._factory(), duration_s=600, dt=1.0)
        assert {"final_imbalance", "energy_loss_J"}.issubset(df.columns)
        assert len(df) == 3

    def test_inductor_more_efficient_than_passive(self):
        df = bms.compare_balancers(self._factory(), duration_s=3600, dt=2.0)
        assert (df.loc["inductor_active", "energy_loss_J"]
                < df.loc["passive_resistive", "energy_loss_J"])


# ----------------------------------------------------------------------
# 5. SOC estimators
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def truth_trace():
    oc = bms.OCVSOC()
    p = bms.ECMParameters(R0=0.025, R1=0.012, C1=2500, R2=0.025,
                          C2=10_000, Q_nom_Ah=2.3)
    m = bms.SecondOrderECM(params=p, ocv_curve=oc)
    n = 1800
    i = np.zeros(n)
    i[100:600] = 1.5
    i[800:1400] = 2.5
    out = m.simulate(i, 1.0, soc0=0.95)
    return p, oc, i, out["v_terminal"], out["soc"]

