"""
Differential Voltage Analysis (DVA) and Incremental Capacity Analysis (ICA).

Theory
------
DVA (dV/dQ)
    Peaks and valleys in the differential voltage fingerprint the boundaries
    between two-phase and single-phase regions in the electrode.  Under aging:
    * Peaks shift in Q (capacity loss)
    * Peak heights diminish (loss of active lithium or active material)
    Most diagnostic for NMC / NCA where peaks are pronounced.

ICA (dQ/dV)
    Peaks in incremental capacity correspond to voltage plateaus (flat OCV
    regions where much charge is stored in a narrow voltage window).  Well
    suited to LFP / LMFP where the flat plateau makes absolute SOC hard to
    estimate; the peak area ∝ active lithium inventory.

Both curves are obtained from a slow (C/20) galvanostatic discharge or the
OCV curve itself.  Savitzky-Golay smoothing is applied to suppress noise.

Usage
-----
::

    q, v = synthetic_discharge_for_dva("lfp", capacity_Ah=3.2)
    q_ax, dva_curve = compute_dva(q, v)
    v_ax, ica_curve = compute_ica(q, v)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


# ── DVA ────────────────────────────────────────────────────────────────────
def compute_dva(
    q_Ah: np.ndarray,
    v_V: np.ndarray,
    smooth_window: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Differential voltage analysis: dV/dQ [V/Ah].

    Parameters
    ----------
    q_Ah : array_like
        Cumulative charge throughput (monotonically increasing) [Ah].
    v_V : array_like
        Corresponding terminal (or OCV) voltage [V].
    smooth_window : int
        Savitzky-Golay filter window length (must be odd, ≥ 3).

    Returns
    -------
    q : ndarray
        Charge axis [Ah].
    dva : ndarray
        Smoothed dV/dQ [V/Ah].
    """
    q = np.asarray(q_Ah, float)
    v = np.asarray(v_V, float)

    # Sort by charge and remove duplicates
    idx = np.argsort(q)
    q, v = q[idx], v[idx]
    _, uniq = np.unique(q, return_index=True)
    q, v = q[uniq], v[uniq]

    dva = np.gradient(v, q)

    win = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
    if win >= 3 and len(dva) >= win:
        dva = savgol_filter(dva, win, polyorder=2)

    return q, dva


# ── ICA ────────────────────────────────────────────────────────────────────
def compute_ica(
    q_Ah: np.ndarray,
    v_V: np.ndarray,
    smooth_window: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """Incremental capacity analysis: dQ/dV [Ah/V].

    Parameters
    ----------
    q_Ah : array_like
        Cumulative charge throughput [Ah].
    v_V : array_like
        Corresponding voltage [V].
    smooth_window : int
        Savitzky-Golay filter window length (must be odd, ≥ 3).

    Returns
    -------
    v : ndarray
        Voltage axis [V].
    ica : ndarray
        Smoothed dQ/dV [Ah/V] (non-negative).
    """
    q = np.asarray(q_Ah, float)
    v = np.asarray(v_V, float)

    # Sort by voltage and remove duplicates
    idx = np.argsort(v)
    q, v = q[idx], v[idx]
    _, uniq = np.unique(v, return_index=True)
    q, v = q[uniq], v[uniq]

    # Discharge convention: q increases as v decreases → dq/dv < 0.
    # Standard ICA = |dQ_remaining/dV| = -dq_discharge/dV (always ≥ 0).
    ica = -np.gradient(q, v)
    ica = np.maximum(ica, 0.0)

    win = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
    if win >= 3 and len(ica) >= win:
        try:
            ica = savgol_filter(ica, win, polyorder=2)
            ica = np.maximum(ica, 0.0)
        except ValueError:
            pass

    return v, ica


# ── Synthetic reference trace ───────────────────────────────────────────────
def synthetic_discharge_for_dva(
    chemistry: str = "nmc",
    capacity_Ah: float | None = None,
    n_points: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic OCV discharge trace for DVA/ICA demonstration.

    Evaluates the OCV curve from SOC=1 down to SOC=0 (ideal slow C/20
    discharge where V_terminal ≈ OCV).  Use the returned arrays directly
    as inputs to :func:`compute_dva` or :func:`compute_ica`.

    Parameters
    ----------
    chemistry : str
        Cell chemistry (``"nmc"``, ``"lfp"``, ``"lmfp"``, ``"lto"``,
        ``"nca"``, ``"lmo"``).
    capacity_Ah : float, optional
        Cell capacity [Ah].  Defaults to the chemistry default.
    n_points : int
        Number of sample points along the SOC axis.

    Returns
    -------
    q_Ah : ndarray
        Cumulative discharge [Ah] from 0 → capacity_Ah.
    v_V : ndarray
        Open-circuit voltage [V] at each point.
    """
    from .chemistry import get_chemistry_props
    from .ocv_soc import OCVSOC

    if capacity_Ah is None:
        capacity_Ah = get_chemistry_props(chemistry)["default_capacity_Ah"]

    soc = np.linspace(1.0, 0.0, n_points)
    oc = OCVSOC.from_chemistry(chemistry)
    v = oc.ocv(soc).astype(float)
    q = (1.0 - soc) * float(capacity_Ah)
    return q, v
