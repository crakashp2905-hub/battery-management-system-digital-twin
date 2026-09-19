"""Uncertainty calibration — is a reported 95 % interval actually 95 %?

A twin that reports ``SoC = 0.71 ± 0.02`` is only trustworthy if that interval is
**calibrated**: over many predictions, the true value should fall inside the 95 %
interval about 95 % of the time.  A confidence number nobody has checked is
decoration.  This module provides the standard probabilistic-forecast metrics to
validate (and, when they fail, expose) the twin's uncertainty:

* :func:`coverage` — the observed hit-rate of a nominal ``level`` interval;
* :func:`picp` / :func:`mpiw` — prediction-interval coverage and mean width;
* :func:`reliability_curve` — observed vs nominal coverage across levels (the
  data behind a reliability diagram);
* :func:`expected_calibration_error` — the average gap between nominal and
  observed coverage (0 = perfectly calibrated);
* :func:`sharpness` — average interval width (tight is good, *given* calibration);
* :func:`assess_calibration` — a one-call :class:`CalibrationReport`.

An over-confident twin (σ too small) under-covers; an under-confident one
over-covers.  Both are reported honestly rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _z(level: float) -> float:
    """Half-width multiplier for a central Gaussian interval of probability ``level``."""
    from scipy.stats import norm
    return float(norm.ppf(0.5 + 0.5 * float(np.clip(level, 0.0, 1.0 - 1e-9))))


def coverage(y_true: np.ndarray, mean: np.ndarray, sigma: np.ndarray,
             level: float = 0.95) -> float:
    """Observed fraction of truths inside the ``level`` Gaussian interval."""
    y = np.asarray(y_true, float)
    m = np.asarray(mean, float)
    s = np.abs(np.asarray(sigma, float))
    z = _z(level)
    inside = np.abs(y - m) <= z * s
    return float(np.mean(inside))


def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction-interval coverage probability for explicit [lower, upper]."""
    y = np.asarray(y_true, float)
    return float(np.mean((y >= np.asarray(lower, float)) & (y <= np.asarray(upper, float))))


def mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean prediction-interval width."""
    return float(np.mean(np.asarray(upper, float) - np.asarray(lower, float)))


def sharpness(sigma: np.ndarray, level: float = 0.95) -> float:
    """Mean width of the ``level`` interval — smaller is sharper (given calibration)."""
    return float(2.0 * _z(level) * np.mean(np.abs(np.asarray(sigma, float))))


def reliability_curve(y_true: np.ndarray, mean: np.ndarray, sigma: np.ndarray,
                      levels: np.ndarray | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Observed vs nominal coverage across ``levels`` (reliability-diagram data)."""
    if levels is None:
        levels = np.linspace(0.1, 0.95, 10)
    levels = np.asarray(levels, float)
    obs = np.array([coverage(y_true, mean, sigma, float(p)) for p in levels])
    return levels, obs


def expected_calibration_error(y_true: np.ndarray, mean: np.ndarray,
                               sigma: np.ndarray, n_levels: int = 10) -> float:
    """Mean |nominal − observed| coverage over a sweep of levels (0 = calibrated)."""
    levels = np.linspace(0.05, 0.95, n_levels)
    nominal, obs = reliability_curve(y_true, mean, sigma, levels)
    return float(np.mean(np.abs(nominal - obs)))


def calibration_scale(y_true: np.ndarray, mean: np.ndarray, sigma: np.ndarray,
                      level: float = 0.95) -> float:
    """Multiplicative σ-correction that makes the ``level`` interval nominal.

    Returns ``c`` such that ``coverage(y, mean, c·sigma, level) ≈ level``: the
    empirical ``level``-quantile of the standardised residuals divided by the
    Gaussian ``z``.  ``c < 1`` shrinks an over-cautious (underconfident) σ; ``c > 1``
    widens an overconfident one — the standard fix once calibration is measured.
    """
    y = np.asarray(y_true, float)
    m = np.asarray(mean, float)
    s = np.abs(np.asarray(sigma, float))
    ok = s > 1e-12
    if not np.any(ok):
        return 1.0
    z = np.abs((y[ok] - m[ok]) / s[ok])
    return float(np.quantile(z, level) / max(_z(level), 1e-9))


@dataclass(frozen=True)
class CalibrationReport:
    """A verdict on how trustworthy a twin's uncertainty is."""

    coverage_95: float
    ece: float
    sharpness_95: float
    n: int
    verdict: str            # "calibrated" / "overconfident" / "underconfident"

    def to_dict(self) -> dict:
        return {"coverage_95": self.coverage_95, "ece": self.ece,
                "sharpness_95": self.sharpness_95, "n": self.n, "verdict": self.verdict}


def assess_calibration(y_true: np.ndarray, mean: np.ndarray, sigma: np.ndarray, *,
                       tol: float = 0.05) -> CalibrationReport:
    """Summarise calibration quality with a plain-language verdict.

    The verdict uses the **signed** miscalibration averaged over the whole
    reliability curve (observed − nominal coverage): near zero is "calibrated",
    consistently below nominal is "overconfident" (intervals too tight), above is
    "underconfident".  This is robust where 95 % coverage alone saturates at 1.0.
    """
    cov = coverage(y_true, mean, sigma, 0.95)
    ece = expected_calibration_error(y_true, mean, sigma)
    sharp = sharpness(sigma, 0.95)
    nominal, obs = reliability_curve(y_true, mean, sigma)
    signed_bias = float(np.mean(obs - nominal))
    if signed_bias < -tol:
        verdict = "overconfident"
    elif signed_bias > tol:
        verdict = "underconfident"
    else:
        verdict = "calibrated"
    return CalibrationReport(coverage_95=cov, ece=ece, sharpness_95=sharp,
                             n=int(np.asarray(y_true).size), verdict=verdict)
