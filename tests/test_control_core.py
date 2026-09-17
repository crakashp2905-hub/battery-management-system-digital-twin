"""Tests: portable control core + software-in-the-loop."""

from __future__ import annotations

import numpy as np

import bms
from bms.control_core import (
    FLAG_OVERCURRENT,
    FLAG_OVERTEMP,
    FLAG_OVERVOLTAGE,
    CoreParams,
    core_reset,
    core_step,
    ocv_lut,
    run_sil,
)


def _plant_trace(seed=0, n=1800):
    p = bms.ECMParameters(R0=0.025, R1=0.015, C1=2000, R2=0.03, C2=8000, Q_nom_Ah=2.3)
    ecm = bms.SecondOrderECM(params=p, ocv_curve=bms.OCVSOC.from_chemistry("nmc", hysteresis_v=0.0))
    ecm.reset(0.9)
    i = np.zeros(n)
    i[100:600] = 1.5
    i[800:1400] = 2.3
    sim = ecm.simulate(i, 1.0)
    v = sim["v_terminal"] + np.random.default_rng(seed).normal(0, 0.003, n)
    return i, v, sim["soc"]


class TestControlCore:
    _CORE = CoreParams(r0=0.025, r1=0.015, tau1=30, r2=0.03, tau2=267, q_nom_as=2.3 * 3600)

    def test_ocv_lut_is_monotone(self):
        socs = np.linspace(0, 1, 21)
        vs = [ocv_lut(s) for s in socs]
        assert all(b >= a - 1e-9 for a, b in zip(vs, vs[1:]))
        assert ocv_lut(0.0) < ocv_lut(1.0)

    def test_sil_tracks_the_twin_plant(self):
        i, v, soc = _plant_trace()
        res = run_sil(self._CORE, i, v, np.full(len(i), 25.0), 1.0, soc_truth=soc)
        assert res["soc_rmse"] < 0.01          # C-portable core tracks the twin
        assert res["soc_max_err"] < 0.02

    def test_safety_flags_fire(self):
        st = core_reset(0.5)
        assert core_step(st, self._CORE, 3.7, 1.0, 80.0, 1.0)["flags"] & FLAG_OVERTEMP
        assert core_step(st, self._CORE, 4.5, 1.0, 25.0, 1.0)["flags"] & FLAG_OVERVOLTAGE
        assert core_step(st, self._CORE, 3.7, 99.0, 25.0, 1.0)["flags"] & FLAG_OVERCURRENT
        assert core_step(st, self._CORE, 4.5, 1.0, 80.0, 1.0)["contactor_open"] is True

    def test_no_flags_when_nominal(self):
        st = core_reset(0.6)
        out = core_step(st, self._CORE, 3.8, 1.0, 25.0, 1.0)
        assert out["flags"] == 0
        assert out["contactor_open"] is False
        assert out["power_discharge_W"] > 0 and out["power_charge_W"] > 0

    def test_is_dependency_light(self):
        # The core must not import numpy/scipy/pandas (it targets firmware).
        import bms.control_core as cc
        src = open(cc.__file__, encoding="utf-8").read()
        assert "import numpy" not in src
        assert "import scipy" not in src and "import pandas" not in src

    def test_state_is_carried_between_steps(self):
        st = core_reset(0.9)
        s0 = st.soc
        for _ in range(50):
            core_step(st, self._CORE, 3.8, 2.3, 25.0, 1.0)   # sustained discharge
        assert st.soc < s0                                    # SoC advanced with load
