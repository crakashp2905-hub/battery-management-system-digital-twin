"""Property-based invariant tests (hypothesis).

These assert *laws* the twin must obey for **any** input — charge conservation,
OCV monotonicity, non-negative irreversible heat, bounded SoC / safety score —
catching whole classes of bugs that example-based tests miss.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import bms

_finite = dict(allow_nan=False, allow_infinity=False)


class TestChargeConservation:
    @given(soc0=st.floats(0.35, 0.65, **_finite),
           cap=st.floats(2.0, 5.0, **_finite),
           currents=st.lists(st.floats(-1.0, 1.0, **_finite), min_size=1, max_size=120))
    @settings(max_examples=150, deadline=None)
    def test_coulomb_counter_integrates_exactly(self, soc0, cap, currents):
        # Small currents/short horizon keep SoC inside [0,1] → no clamping, so
        # the counter must equal the exact ampere-second integral.
        i = np.asarray(currents, float)
        cc = bms.CoulombCounter(cap, soc0=soc0)
        out = cc.run(i, np.zeros_like(i), 1.0)
        expected = np.clip(soc0 - np.cumsum(i) * 1.0 / (cap * 3600.0), 0.0, 1.0)
        assert np.allclose(out, expected, atol=1e-9)


class TestOCVMonotonic:
    @given(a=st.floats(0.02, 0.98, **_finite), b=st.floats(0.02, 0.98, **_finite))
    @settings(max_examples=200, deadline=None)
    def test_ocv_is_nondecreasing_in_soc(self, a, b):
        ocv = bms.OCVSOC()
        lo, hi = min(a, b), max(a, b)
        # Rested cell (no hysteresis/current), same temperature.
        assert float(ocv.ocv(hi)) >= float(ocv.ocv(lo)) - 1e-9


class TestHeatNonNegative:
    @given(current=st.floats(-300.0, 300.0, **_finite),
           r0=st.floats(1e-4, 0.1, **_finite),
           over=st.floats(-0.5, 0.5, **_finite))
    @settings(max_examples=200, deadline=None)
    def test_irreversible_heat_is_never_negative(self, current, r0, over):
        # OCV − V_terminal = `over`; irreversible heat = |I·over| floored at I²R0.
        q = bms.ThermalModel.heat_generation(
            np.array([current]), np.array([r0]),
            np.array([3.7 - over]), np.array([3.7]))
        assert q[0] >= 0.0
        assert q[0] >= current ** 2 * r0 - 1e-12       # ohmic floor holds


class TestSafetyScoreBounded:
    @given(t_max=st.floats(-20.0, 120.0, **_finite),
           imbalance=st.floats(0.0, 0.5, **_finite),
           soh=st.floats(0.4, 1.0, **_finite))
    @settings(max_examples=200, deadline=None)
    def test_state_of_safety_in_unit_interval(self, t_max, imbalance, soh):
        result = {"T_cells": np.array([25.0, 25.0, t_max, 25.0]),
                  "v_cells": np.full(4, 3.7), "imbalance": imbalance}
        sos = bms.state_of_safety(result, chemistry="nmc", soh=soh)
        assert 0.0 <= sos["sos"] <= 1.0


class TestEstimatorBounds:
    @given(currents=st.lists(st.floats(-5.0, 5.0, **_finite), min_size=5, max_size=60))
    @settings(max_examples=60, deadline=None)
    def test_recursive_estimators_report_soc_in_unit_interval(self, currents):
        p = bms.ECMParameters()
        ocv = bms.OCVSOC()
        ecm = bms.SecondOrderECM(params=p, ocv_curve=ocv)
        ecm.reset(0.7)
        i = np.asarray(currents, float)
        v = ecm.simulate(i, 1.0)["v_terminal"]
        for name in ("coulomb", "ekf", "bias_ekf"):
            est = bms.make_soc_estimator(name, params=p, ocv_curve=ocv, capacity_Ah=2.3)
            est.reset(0.7)
            out = est.run(i, v, 1.0)
            assert np.all((out >= 0.0) & (out <= 1.0)), name


class TestChecksumRange:
    @given(data=st.binary(min_size=8, max_size=8))
    @settings(max_examples=200, deadline=None)
    def test_can_checksum_is_a_byte(self, data):
        assert 0 <= bms.can_checksum(data) <= 255
