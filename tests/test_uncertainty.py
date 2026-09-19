"""Tests: uncertainty calibration metrics (is a 95% interval really 95%?)."""

from __future__ import annotations

import numpy as np

import bms


def _gaussian(n=20000, sigma=0.02, scale=1.0, seed=0):
    """Truth = mean + N(0, sigma); `scale` multiplies the *reported* sigma."""
    rng = np.random.default_rng(seed)
    mean = rng.uniform(0.2, 0.8, n)
    y = mean + rng.normal(0.0, sigma, n)
    reported_sigma = np.full(n, sigma * scale)
    return y, mean, reported_sigma


class TestCalibrationMetrics:
    def test_well_calibrated_covers_at_nominal(self):
        y, m, s = _gaussian(scale=1.0)
        assert abs(bms.coverage(y, m, s, 0.95) - 0.95) < 0.02
        assert abs(bms.coverage(y, m, s, 0.50) - 0.50) < 0.02
        assert bms.expected_calibration_error(y, m, s) < 0.02

    def test_overconfident_undercovers(self):
        # Reported sigma half the truth → intervals too tight → under-coverage.
        y, m, s = _gaussian(scale=0.5)
        assert bms.coverage(y, m, s, 0.95) < 0.90
        rep = bms.assess_calibration(y, m, s)
        assert rep.verdict == "overconfident"

    def test_underconfident_overcovers(self):
        y, m, s = _gaussian(scale=2.0)                # sigma too large
        assert bms.coverage(y, m, s, 0.95) > 0.98
        assert bms.assess_calibration(y, m, s).verdict == "underconfident"

    def test_reliability_curve_is_monotone_and_bracketed(self):
        y, m, s = _gaussian(scale=1.0)
        levels, obs = bms.reliability_curve(y, m, s)
        assert np.all(np.diff(obs) >= -0.02)          # coverage rises with level
        assert obs[0] < obs[-1]
        # well-calibrated → observed ≈ nominal
        assert np.mean(np.abs(levels - obs)) < 0.03

    def test_picp_and_sharpness_tradeoff(self):
        y, m, s = _gaussian(scale=1.0)
        lo, hi = m - 1.96 * s, m + 1.96 * s
        assert abs(bms.picp(y, lo, hi) - 0.95) < 0.02
        assert bms.mpiw(lo, hi) > 0
        # a sharper (smaller-sigma) forecast has a smaller 95% width
        assert bms.sharpness(s * 0.5) < bms.sharpness(s)

    def test_calibration_scale_fixes_miscalibration(self):
        # An overconfident forecast (σ half of truth) is corrected by the scale.
        y, m, s = _gaussian(scale=0.5)
        c = bms.calibration_scale(y, m, s, 0.95)
        assert c > 1.3                                 # needs widening
        assert abs(bms.coverage(y, m, c * s, 0.95) - 0.95) < 0.02
        # An underconfident forecast (σ too large) is shrunk.
        y2, m2, s2 = _gaussian(scale=2.0)
        c2 = bms.calibration_scale(y2, m2, s2, 0.95)
        assert c2 < 0.7
        assert abs(bms.coverage(y2, m2, c2 * s2, 0.95) - 0.95) < 0.02

    def test_report_serialises(self):
        y, m, s = _gaussian()
        d = bms.assess_calibration(y, m, s).to_dict()
        for k in ("coverage_95", "ece", "sharpness_95", "verdict"):
            assert k in d
