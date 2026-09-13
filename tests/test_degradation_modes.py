"""Tests: LLI vs LAM degradation-mode diagnosis."""

from __future__ import annotations

import numpy as np
import pytest

import bms


class TestDegradationModes:
    @pytest.mark.parametrize("lli,lam", [(0.15, 0.05), (0.05, 0.20), (0.12, 0.08)])
    def test_recovers_injected_modes(self, lli, lam):
        v, fresh, aged = bms.synthetic_degraded_ic(lli=lli, lam=lam)
        d = bms.diagnose_degradation_modes(v, fresh, aged)
        assert abs(d["lli"] - lli) < 0.02
        assert abs(d["lam"] - lam) < 0.02

    def test_dominant_mode_is_identified(self):
        v, fresh, aged = bms.synthetic_degraded_ic(lli=0.20, lam=0.03)
        assert bms.diagnose_degradation_modes(v, fresh, aged)["dominant_mode"] == "LLI"
        v, fresh, aged = bms.synthetic_degraded_ic(lli=0.03, lam=0.20)
        assert bms.diagnose_degradation_modes(v, fresh, aged)["dominant_mode"] == "LAM"

    def test_no_degradation_is_zero(self):
        v, fresh, aged = bms.synthetic_degraded_ic(lli=0.0, lam=0.0)
        d = bms.diagnose_degradation_modes(v, fresh, aged)
        assert d["lli"] == pytest.approx(0.0, abs=1e-6)
        assert d["lam"] == pytest.approx(0.0, abs=1e-6)
        assert d["total_capacity_loss"] == pytest.approx(0.0, abs=1e-6)

    def test_pure_lam_lowers_peak_pure_lli_narrows(self):
        v, fresh, lam_aged = bms.synthetic_degraded_ic(lli=0.0, lam=0.25)
        v2, fresh2, lli_aged = bms.synthetic_degraded_ic(lli=0.25, lam=0.0)
        # LAM drops the peak height; LLI keeps it but cuts the area.
        assert lam_aged.max() < fresh.max() - 1e-6
        assert lli_aged.max() == pytest.approx(fresh2.max(), abs=1e-3)   # height ~unchanged
        area = lambda ic, ax: float(np.trapezoid(ic, ax)) if hasattr(np, "trapezoid") \
            else float(np.trapz(ic, ax))
        assert area(lli_aged, v2) < area(fresh2, v2) - 1e-6

    def test_total_loss_exceeds_each_mode(self):
        v, fresh, aged = bms.synthetic_degraded_ic(lli=0.1, lam=0.1)
        d = bms.diagnose_degradation_modes(v, fresh, aged)
        assert d["total_capacity_loss"] >= max(d["lli"], d["lam"]) - 1e-9
