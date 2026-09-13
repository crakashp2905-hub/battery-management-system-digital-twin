"""Degradation-mode diagnosis: loss of lithium inventory (LLI) vs active
material (LAM) from incremental-capacity (IC / dQ/dV) curves.

Capacity fade has *causes*, and the incremental-capacity curve carries their
fingerprints (Dubarry-style mechanistic analysis):

* **LAM** — loss of active material shrinks the electrode capacity, so the IC
  **peak height** drops.
* **LLI** — loss of cyclable lithium narrows the accessible SoC window, so the
  IC **area** shrinks *at a given peak height* (the peaks move together / the
  curve narrows) without necessarily lowering the peaks.

That gives a clean, invertible two-mode split from two measurable ratios — peak
**height** and total **area** between a fresh and an aged IC curve:

    LAM = 1 − height_aged / height_fresh
    LLI = 1 − (area_aged / area_fresh) / (height_aged / height_fresh)

:func:`synthetic_degraded_ic` builds a fresh/aged pair with *known* LLI/LAM so
the diagnosis can be verified end to end; :func:`diagnose_degradation_modes`
recovers them from any fresh/aged IC pair (e.g. from :func:`bms.compute_ica`).
"""

from __future__ import annotations

import numpy as np

# ``np.trapz`` was removed in NumPy 2.0 (renamed ``np.trapezoid``); pick whichever
# exists without eagerly evaluating the missing one.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _gaussian_ic(v_axis: np.ndarray, peak_v: float, height: float,
                 width: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((v_axis - peak_v) / max(width, 1e-9)) ** 2)


def synthetic_degraded_ic(lli: float = 0.0, lam: float = 0.0,
                          peak_v: float = 3.7, height: float = 1.0,
                          width: float = 0.05, n: int = 400):
    """Build a (v_axis, ic_fresh, ic_aged) triple with known LLI/LAM fractions.

    LAM scales the peak **height** by ``(1 − lam)``; LLI narrows the peak
    **width** by ``(1 − lli)`` (reducing area at fixed height).  So the aged
    curve encodes exactly those two modes, and the diagnosis must recover them.
    """
    v = np.linspace(peak_v - 6 * width, peak_v + 6 * width, n)
    fresh = _gaussian_ic(v, peak_v, height, width)
    aged = _gaussian_ic(v, peak_v, height * (1.0 - lam), width * (1.0 - lli))
    return v, fresh, aged


def diagnose_degradation_modes(v_axis: np.ndarray, ic_fresh: np.ndarray,
                               ic_aged: np.ndarray) -> dict:
    """Decompose fade into LLI / LAM fractions from a fresh vs aged IC curve.

    Returns ``lli``, ``lam`` (each ≥ 0), ``total_capacity_loss`` (1 − area
    ratio), and the raw ``height_ratio`` / ``area_ratio`` for transparency.
    """
    h_fresh = float(np.max(ic_fresh))
    h_aged = float(np.max(ic_aged))
    a_fresh = float(_trapz(ic_fresh, v_axis))
    a_aged = float(_trapz(ic_aged, v_axis))
    height_ratio = h_aged / max(h_fresh, 1e-12)
    area_ratio = a_aged / max(a_fresh, 1e-12)
    lam = max(0.0, 1.0 - height_ratio)
    lli = max(0.0, 1.0 - area_ratio / max(height_ratio, 1e-12))
    return {
        "lli": float(lli),
        "lam": float(lam),
        "total_capacity_loss": float(max(0.0, 1.0 - area_ratio)),
        "height_ratio": float(height_ratio),
        "area_ratio": float(area_ratio),
        "dominant_mode": "LLI" if lli > lam else ("LAM" if lam > lli else "balanced"),
    }
