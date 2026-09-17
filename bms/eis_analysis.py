"""EIS analysis: Distribution of Relaxation Times (DRT) and impedance-based SoH.

:func:`bms.simulate_eis` produces a Nyquist spectrum; this reads it back:

* :func:`compute_drt` deconvolves the impedance into a **distribution of
  relaxation times** γ(τ) — the state-of-the-art way to separate the overlapping
  physical processes (SEI, charge-transfer, diffusion) that a raw Nyquist arc
  blends together.  It solves the ill-posed inversion
  ``−Im Z(ω) = ∫ γ(τ)·ωτ/(1+(ωτ)²) dlnτ`` with Tikhonov (2nd-difference)
  smoothing and a non-negativity constraint.
* :func:`eis_resistances` extracts the ohmic ``R0`` (high-frequency intercept)
  and the charge-transfer ``R_ct`` (semicircle width); :func:`eis_soh` turns
  their growth into an **impedance-based SoH**, a fade signal independent of
  capacity (it tracks power fade, which capacity alone misses).
"""

from __future__ import annotations

import numpy as np


def _second_difference(n: int) -> np.ndarray:
    L = np.zeros((max(n - 2, 0), n))
    for i in range(n - 2):
        L[i, i], L[i, i + 1], L[i, i + 2] = 1.0, -2.0, 1.0
    return L


def compute_drt(freq_Hz: np.ndarray, Z_neg_imag: np.ndarray, n_tau: int = 80,
                reg: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
    """Distribution of relaxation times from the imaginary impedance.

    Returns ``(tau, gamma)`` — the relaxation-time grid [s] and the (non-negative)
    DRT amplitude at each.  Peaks in γ(τ) mark the cell's characteristic time
    constants; their area is the associated polarisation resistance.
    """
    from scipy.optimize import nnls

    f = np.asarray(freq_Hz, float)
    b = np.asarray(Z_neg_imag, float)
    omega = 2.0 * np.pi * f
    tau = np.logspace(np.log10(1.0 / omega.max()) - 0.5,
                      np.log10(1.0 / omega.min()) + 0.5, n_tau)
    wt = np.outer(omega, tau)
    A = wt / (1.0 + wt ** 2)                      # kernel: −Im Z contribution
    L = _second_difference(n_tau)
    A_aug = np.vstack([A, np.sqrt(reg) * L])
    b_aug = np.concatenate([b, np.zeros(L.shape[0])])
    gamma, _ = nnls(A_aug, b_aug)
    return tau, gamma


def eis_resistances(freq_Hz: np.ndarray, Z_real: np.ndarray,
                    Z_neg_imag: np.ndarray) -> dict:
    """Extract ohmic ``R0`` and charge-transfer ``R_ct`` from a Nyquist spectrum.

    ``R0`` is the high-frequency real-axis intercept; ``R_ct`` is the real-axis
    span of the semicircle up to its apex (peak −Im Z).
    """
    f = np.asarray(freq_Hz, float)
    zre = np.asarray(Z_real, float)
    zim = np.asarray(Z_neg_imag, float)
    order = np.argsort(f)                          # ascending frequency
    r0 = float(zre[order][-1])                     # highest frequency → ohmic
    apex = int(np.argmax(zim))                     # semicircle apex
    r_ct = float(max(0.0, zre[apex] - r0))
    return {"R0": r0, "R_ct": r_ct}


def eis_soh(freq_Hz, Z_real_fresh, Z_neg_imag_fresh,
            Z_real_aged, Z_neg_imag_aged, r_growth_at_eol: float = 2.0) -> dict:
    """Impedance-based SoH from resistance growth (fresh vs aged spectra).

    Ageing raises ``R0`` and ``R_ct``; total resistance roughly doubling by
    end-of-life is a common rule of thumb (``r_growth_at_eol``).  Returns the
    resistance ratio and a 1→0 ``soh_impedance`` that reaches 0 at that growth.
    """
    fr = eis_resistances(freq_Hz, Z_real_fresh, Z_neg_imag_fresh)
    ag = eis_resistances(freq_Hz, Z_real_aged, Z_neg_imag_aged)
    r_fresh = fr["R0"] + fr["R_ct"]
    r_aged = ag["R0"] + ag["R_ct"]
    ratio = r_aged / max(r_fresh, 1e-12)
    soh = float(np.clip(1.0 - (ratio - 1.0) / (r_growth_at_eol - 1.0), 0.0, 1.0))
    return {"resistance_ratio": float(ratio), "soh_impedance": soh,
            "R0_growth": ag["R0"] / max(fr["R0"], 1e-12),
            "Rct_growth": ag["R_ct"] / max(fr["R_ct"], 1e-12)}
