"""Tests: dynamic (Plett) OCV hysteresis."""

from __future__ import annotations

import numpy as np

import bms
from bms.hysteresis import PlettHysteresis


class TestPlettHysteresis:
    def test_saturates_to_branches(self):
        h = PlettHysteresis(max_hysteresis_V=0.02, rate_per_Ah=8.0)
        h.reset(0.0)
        for _ in range(3000):
            h.update(2.3, 1.0, 2.3)                # sustained discharge → h → −1
        assert h.h < -0.95
        assert h.voltage < -0.018
        h.reset(0.0)
        for _ in range(3000):
            h.update(-2.3, 1.0, 2.3)               # sustained charge → h → +1
        assert h.h > 0.95
        assert h.voltage > 0.018

    def test_transition_is_smooth_not_instant(self):
        h = PlettHysteresis(max_hysteresis_V=0.02, rate_per_Ah=8.0)
        h.reset(1.0)                               # on the charge branch
        h.update(2.3, 1.0, 2.3)                    # one discharge step
        assert h.h > 0.9                           # has NOT jumped to −1 (static would)

    def test_voltage_is_bounded_by_M(self):
        h = PlettHysteresis(max_hysteresis_V=0.03)
        for cur in (2.3, -2.3, 5.0, -1.0):
            for _ in range(100):
                h.update(cur, 1.0, 2.3)
            assert abs(h.voltage) <= 0.03 + 1e-9

    def test_holds_state_at_zero_current(self):
        h = PlettHysteresis()
        h.reset(0.5)
        h.update(0.0, 1.0, 2.3)
        assert h.h == 0.5                          # no throughput → no change

    def test_wired_into_ecm_gives_branch_separation(self):
        def _final_v(current_sign, soc0):
            ecm = bms.SecondOrderECM(
                params=bms.ECMParameters(Q_nom_Ah=2.3), ocv_curve=bms.OCVSOC(),
                hysteresis=PlettHysteresis(0.02, 8.0))
            return ecm.simulate(np.full(200, current_sign * 2.3), 1.0,
                                soc0=soc0)["v_terminal"][-1]
        v_charge = _final_v(-1, 0.4)               # charge up to ~0.5
        v_discharge = _final_v(+1, 0.6)            # discharge down to ~0.5
        assert v_charge > v_discharge              # charge branch sits higher

    def test_reset_clears_hysteresis_via_ecm(self):
        ecm = bms.SecondOrderECM(hysteresis=PlettHysteresis(0.02, 8.0))
        ecm.simulate(np.full(100, 2.3), 1.0, soc0=0.8)
        ecm.reset(0.8)
        assert ecm.hysteresis.h == 0.0
