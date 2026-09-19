"""Validation on a committed slice of REAL data (NASA PCoE B0005, public domain).

This runs the estimator pipeline on a genuine discharge cycle from the NASA Ames
battery aging dataset (shipped as a small ~20 KB fixture with attribution), so a
real-data result is exercised in CI without the multi-MB .mat files.
"""

from __future__ import annotations

from pathlib import Path

import bms

_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "samples" / "nasa_b0005_discharge_real.csv"
_CAP_AH = 1.8353          # the real discharge capacity of this cycle (see file header)


class TestRealNASAValidation:
    def test_fixture_exists_and_loads_skipping_attribution_header(self):
        assert _FIXTURE.exists()
        data = bms.load_drivecycle_csv(str(_FIXTURE), chemistry="nmc", capacity_Ah=_CAP_AH)
        assert data.source == "real"                    # tagged real, not synthetic
        assert data.n > 150                             # the discharge cycle
        assert abs(float(data.time_s[0])) < 1e-6        # '#' header lines were skipped

    def test_coulomb_counting_is_accurate_on_real_data(self):
        data = bms.load_drivecycle_csv(str(_FIXTURE), chemistry="nmc", capacity_Ah=_CAP_AH)
        board = bms.estimator_leaderboard(data)
        # Coulomb counting tracks the real coulomb-counted SoC to well under 2 %.
        assert board.loc["coulomb", "rmse"] < 0.02

    def test_voltage_filters_need_a_cell_specific_ocv(self):
        # The documented real-data finding: with the generic NMC OCV (this is an
        # LiCoO2 cell) the voltage filters are far worse than coulomb-counting —
        # real data makes the case for cell-specific calibration.
        data = bms.load_drivecycle_csv(str(_FIXTURE), chemistry="nmc", capacity_Ah=_CAP_AH)
        board = bms.estimator_leaderboard(data)
        assert board.loc["ekf", "rmse"] > board.loc["coulomb", "rmse"]

    def test_loader_skips_comment_headers(self):
        # Real public datasets carry licence/attribution headers; the loader must
        # tolerate '#' comment lines.
        import numpy as np
        raw = _FIXTURE.read_text().splitlines()
        assert raw[0].startswith("#") and raw[1].startswith("#")
        data = bms.load_drivecycle_csv(str(_FIXTURE), chemistry="nmc", capacity_Ah=_CAP_AH)
        assert np.all(np.isfinite(data.voltage_V))
