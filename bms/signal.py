"""Signal conditioning — clean raw BMS sensor data before it reaches a model.

Real battery telemetry is noisy and glitchy: dropped samples, single-sample
current spikes, ADC quantisation, thermal noise on voltage.  Feeding that to an
estimator or a parameter fit pollutes the result.  This module provides the small
set of robust cleaners that genuinely matter for BMS data, drawn from the
standard signal-processing toolkit:

* :func:`hampel` — a robust median/MAD outlier filter that removes sensor spikes
  without smearing real edges (the workhorse for despiking current/voltage);
* :func:`median_filter` — classic impulse-noise rejection;
* :func:`savitzky_golay` — polynomial smoothing that preserves peak shape, so
  it is safe ahead of DVA/ICA where peak height/area carry the diagnosis;
* :func:`ewma` / :func:`moving_average` — light low-pass smoothing;
* :func:`clean_signal` — the recommended pipeline (Hampel despike → gentle
  Savitzky-Golay), returning the cleaned series and where it acted.

Everything is pure NumPy/SciPy (both core dependencies) and edge-safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MAD_TO_SIGMA = 1.4826      # MAD → σ for a normal distribution


def hampel(x: np.ndarray, window: int = 7, n_sigma: float = 3.0
           ) -> tuple[np.ndarray, np.ndarray]:
    """Robust outlier removal via a sliding median / MAD.

    A sample more than ``n_sigma`` robust-σ from its local median is replaced by
    that median.  Returns ``(filtered, outlier_mask)``.  ``window`` is the full
    window length (odd; the half-width is ``window // 2``).
    """
    x = np.asarray(x, float)
    n = x.size
    k = max(1, window // 2)
    out = x.copy()
    mask = np.zeros(n, bool)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        win = x[lo:hi]
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        sigma = _MAD_TO_SIGMA * mad
        if sigma > 0 and abs(x[i] - med) > n_sigma * sigma:
            out[i] = med
            mask[i] = True
    return out, mask


def median_filter(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Sliding-window median filter (impulse-noise rejection), edge-safe."""
    x = np.asarray(x, float)
    n = x.size
    k = max(1, window // 2)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(x[max(0, i - k):min(n, i + k + 1)])
    return out


def savitzky_golay(x: np.ndarray, window: int = 11, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing (preserves peak shape). Falls back gracefully
    when the series is shorter than the window."""
    from scipy.signal import savgol_filter
    x = np.asarray(x, float)
    w = min(window, x.size if x.size % 2 == 1 else x.size - 1)
    if w < 3 or w <= poly:
        return x.copy()
    if w % 2 == 0:
        w -= 1
    return savgol_filter(x, w, poly)


def ewma(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Exponentially weighted moving average (causal low-pass)."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    if x.size == 0:
        return out
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred moving average, edge-safe (shrinking window at the ends)."""
    x = np.asarray(x, float)
    n = x.size
    k = max(1, window // 2)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.mean(x[max(0, i - k):min(n, i + k + 1)])
    return out


@dataclass(frozen=True)
class CleanResult:
    """A cleaned signal plus what was changed."""

    signal: np.ndarray
    n_outliers: int
    outlier_mask: np.ndarray

    def to_dict(self) -> dict:
        return {"n_outliers": int(self.n_outliers)}


def clean_signal(x: np.ndarray, *, hampel_window: int = 7, n_sigma: float = 3.0,
                 smooth_window: int = 7, smooth_poly: int = 2,
                 smooth: bool = True) -> CleanResult:
    """Recommended BMS cleaning pipeline: Hampel despike, then light SG smoothing.

    Returns a :class:`CleanResult` with the cleaned series, the number of spikes
    removed, and their mask.
    """
    despiked, mask = hampel(x, window=hampel_window, n_sigma=n_sigma)
    out = savitzky_golay(despiked, window=smooth_window, poly=smooth_poly) if smooth else despiked
    return CleanResult(signal=out, n_outliers=int(mask.sum()), outlier_mask=mask)
