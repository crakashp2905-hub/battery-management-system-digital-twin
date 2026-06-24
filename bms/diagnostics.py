"""
Electrochemical diagnostics: EIS simulation and C-rate capability map.

EIS Simulation
--------------
The second-order RC ECM maps directly onto the impedance spectroscopy model:

    Z(ω) = R0 + R1/(1+jωτ₁) + R2/(1+jωτ₂) + Z_W(ω)

where Z_W is a semi-infinite Warburg element modelling solid-state diffusion:

    Z_W(ω) = σ/√ω · (1 − j)

The Nyquist plot (−Im Z vs Re Z) shows:
  * High-frequency intercept at R0 (pure ohmic resistance)
  * Two depressed semicircles (RC loops at τ₁ and τ₂)
  * 45-degree diffusion tail at low frequency (Warburg)

C-Rate Capability Map
---------------------
The maximum continuous discharge C-rate at each (SOC, temperature) point is
derived from the terminal-voltage constraint V_t ≥ v_min:

    I_max(SOC, T) = (V_OC(SOC, T) − v_min) / R0_eff(T)

Plotted as a 2-D heatmap, this reveals the operational envelope — where cold
temperatures or low SOC limit power delivery.

Usage
-----
::

    from bms import ECMParameters
    from bms.diagnostics import simulate_eis, compute_crate_map

    params = ECMParameters.for_nmc()
    freqs, Z_re, Z_neg_im = simulate_eis(params, temperature_C=25.0)

    soc_ax, T_ax, cmap = compute_crate_map(params, chemistry="nmc")
"""

from __future__ import annotations

import numpy as np

from .ecm import ECMParameters
from .ocv_soc import OCVSOC


# ── EIS simulation ─────────────────────────────────────────────────────────
def simulate_eis(
    params: ECMParameters,
    frequencies_Hz: np.ndarray | None = None,
    temperature_C: float = 25.0,
    warburg_sigma: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate an EIS spectrum from second-order RC ECM parameters.

    Parameters
    ----------
    params : ECMParameters
        Cell ECM parameters at 25 °C.  Arrhenius-scaled internally when
        *temperature_C* ≠ 25.
    frequencies_Hz : ndarray, optional
        Frequency sweep [Hz].  Defaults to 200 log-spaced points from
        10 kHz down to 1 mHz.
    temperature_C : float
        Operating temperature [°C].
    warburg_sigma : float
        Warburg pre-factor σ [Ω·s⁻⁰·⁵].  Controls low-frequency tail slope.

    Returns
    -------
    freq_Hz : ndarray, shape (N,)
        Frequency axis [Hz].
    Z_real : ndarray, shape (N,)
        Real part of impedance [Ω].
    Z_neg_imag : ndarray, shape (N,)
        Negative imaginary part [Ω] (Nyquist convention: −Im Z ≥ 0 for
        inductive-free cells).
    """
    if frequencies_Hz is None:
        frequencies_Hz = np.logspace(-3, 4, 200)

    p = params.at_temperature(temperature_C) if temperature_C != 25.0 else params
    f = np.asarray(frequencies_Hz, float)
    omega = 2.0 * np.pi * f

    Z_R1C1 = p.R1 / (1.0 + 1j * omega * p.tau1)
    Z_R2C2 = p.R2 / (1.0 + 1j * omega * p.tau2)
    Z_w = warburg_sigma / np.sqrt(omega) * (1.0 - 1j)

    Z = p.R0 + Z_R1C1 + Z_R2C2 + Z_w

    return f, Z.real.copy(), (-Z.imag).copy()


# ── C-rate capability map ───────────────────────────────────────────────────
def compute_crate_map(
    params: ECMParameters,
    chemistry: str = "nmc",
    soc_points: int = 20,
    temp_points: int = 17,
    T_min_C: float = -20.0,
    T_max_C: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute maximum discharge C-rate on a SOC × temperature grid.

    At each (SOC, T) node the maximum discharge current limited by the
    terminal-voltage constraint V_t ≥ v_min is:

        I_max = (OCV(SOC, T) − v_min) / R0_eff(T)

    Then C-rate = I_max / Q_nom.

    Parameters
    ----------
    params : ECMParameters
        Reference cell ECM at 25 °C.
    chemistry : str
        Cell chemistry string — used to look up ``v_min`` and the OCV table.
    soc_points : int
        Grid resolution along the SOC axis.
    temp_points : int
        Grid resolution along the temperature axis.
    T_min_C, T_max_C : float
        Temperature range for the map [°C].

    Returns
    -------
    soc_grid : ndarray, shape (soc_points,)
        SOC axis [0–1].
    T_grid : ndarray, shape (temp_points,)
        Temperature axis [°C].
    crate_map : ndarray, shape (soc_points, temp_points)
        Maximum discharge C-rate at each grid point.
    """
    from .chemistry import get_chemistry_props

    props = get_chemistry_props(chemistry)
    v_min = props["v_min"]
    ocv_curve = OCVSOC.from_chemistry(chemistry)

    soc_grid = np.linspace(0.05, 1.0, soc_points)
    T_grid = np.linspace(T_min_C, T_max_C, temp_points)
    crate_map = np.zeros((soc_points, temp_points))

    for i, soc in enumerate(soc_grid):
        for j, T in enumerate(T_grid):
            p_T = params.at_temperature(T)
            ocv = float(ocv_curve.ocv(float(soc), T_C=T))
            i_max_A = max(0.0, (ocv - v_min) / max(p_T.R0, 1e-9))
            crate_map[i, j] = i_max_A / max(params.Q_nom_Ah, 1e-9)

    return soc_grid, T_grid, crate_map
