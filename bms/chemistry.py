"""
Cell chemistry definitions for all supported Li-ion / solid-state chemistries.

Supported chemistries
---------------------
NMC  — LiNiMnCoO₂       High energy density, automotive/consumer
LFP  — LiFePO₄           Long cycle life, thermally stable, grid/EV
LMFP — LiMnFePO₄         Dual-plateau (Fe + Mn), higher voltage than LFP
LTO  — Li₄Ti₅O₁₂         Ultra-safe, zero-strain anode, wide temperature range
NCA  — LiNiCoAlO₂         Highest energy density, EV (Tesla), thermally sensitive
LMO  — LiMn₂O₄            Spinel, low cost, double plateau, higher self-discharge
SSB  — Solid-State Battery  Li-metal anode, oxide/sulfide SE, highest safety & voltage

Each entry in CHEMISTRY_PROPS contains:
  ocv_table                  — (N×2) array of [SOC, OCV_V]
  arrhenius_K                — Ea/R [K] for resistance temperature scaling
  temp_coeff_V_per_K         — dOCV/dT [V/K] at reference temperature
  v_min, v_max               — operating voltage window [V]
  v_overcharge               — rule-based overcharge threshold [V]
  v_dropout                  — rule-based sensor-dropout threshold [V]
  T_runaway_C                — thermal-runaway onset temperature [°C]
  nominal_voltage_V          — cell nominal voltage [V]
  default_capacity_Ah        — typical 18650/21700/prismatic capacity [Ah]
  self_discharge_pct_per_month — self-discharge rate at 25 °C [%/month]
  default_ecm                — default ECM parameters dict
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class CellChemistry(str, Enum):
    """Supported cell chemistries."""
    NMC  = "nmc"   # LiNiMnCoO₂  — high energy density
    LFP  = "lfp"   # LiFePO₄     — long cycle life, thermally stable
    LMFP = "lmfp"  # LiMnFePO₄   — dual plateau, higher voltage than LFP
    LTO  = "lto"   # Li₄Ti₅O₁₂   — ultra-safe, wide temperature
    NCA  = "nca"   # LiNiCoAlO₂  — highest energy density, thermally sensitive
    LMO  = "lmo"   # LiMn₂O₄     — spinel, low cost, double plateau
    SSB  = "ssb"   # Solid-State  — Li-metal anode, highest safety, poor cold perf


# ── OCV-SOC tables ─────────────────────────────────────────────────────────

# NMC 18650 — 3.0 V (0%) → 4.20 V (100%)
_NMC_SOC_OCV = np.array([
    [0.00, 3.000],
    [0.05, 3.350],
    [0.10, 3.520],
    [0.20, 3.610],
    [0.30, 3.660],
    [0.40, 3.710],
    [0.50, 3.760],
    [0.60, 3.820],
    [0.70, 3.890],
    [0.80, 3.960],
    [0.90, 4.060],
    [0.95, 4.130],
    [1.00, 4.200],
])

# LFP prismatic — 2.50 V (0%) → 3.65 V (100%), characteristic flat plateau
_LFP_SOC_OCV = np.array([
    [0.00, 2.500],
    [0.05, 3.000],
    [0.10, 3.150],
    [0.15, 3.210],
    [0.20, 3.240],
    [0.30, 3.270],
    [0.40, 3.290],
    [0.50, 3.310],
    [0.60, 3.320],
    [0.70, 3.330],
    [0.80, 3.340],
    [0.90, 3.360],
    [0.95, 3.400],
    [1.00, 3.650],
])

# LMFP (LiMn₀.₇Fe₀.₃PO₄) — 2.80 V → 4.15 V
# Two plateaus: Fe²⁺/³⁺ ~3.44 V (SOC 0–0.30) and Mn²⁺/³⁺ ~3.95 V (SOC 0.40–0.90)
_LMFP_SOC_OCV = np.array([
    [0.00, 2.800],
    [0.05, 3.200],
    [0.10, 3.380],
    [0.15, 3.420],
    [0.20, 3.445],
    [0.25, 3.460],
    [0.30, 3.490],
    [0.35, 3.650],
    [0.40, 3.860],
    [0.50, 3.940],
    [0.60, 3.970],
    [0.70, 3.990],
    [0.80, 4.020],
    [0.90, 4.060],
    [0.95, 4.100],
    [1.00, 4.150],
])

# LTO (Li₄Ti₅O₁₂ anode / NMC cathode) — 1.50 V → 2.80 V
# Extremely flat plateau ~2.35–2.45 V (zero-strain insertion)
_LTO_SOC_OCV = np.array([
    [0.00, 1.500],
    [0.05, 1.800],
    [0.10, 2.050],
    [0.15, 2.250],
    [0.20, 2.320],
    [0.30, 2.360],
    [0.40, 2.380],
    [0.50, 2.400],
    [0.60, 2.420],
    [0.70, 2.450],
    [0.80, 2.490],
    [0.90, 2.560],
    [0.95, 2.660],
    [1.00, 2.800],
])

# NCA (LiNi₀.₈Co₀.₁₅Al₀.₀₅O₂) — 3.00 V → 4.20 V
# Similar to NMC but higher energy density and more pronounced 90–100% rise
_NCA_SOC_OCV = np.array([
    [0.00, 3.000],
    [0.05, 3.400],
    [0.10, 3.550],
    [0.20, 3.650],
    [0.30, 3.700],
    [0.40, 3.745],
    [0.50, 3.790],
    [0.60, 3.840],
    [0.70, 3.910],
    [0.80, 3.980],
    [0.90, 4.080],
    [0.95, 4.150],
    [1.00, 4.200],
])

# LMO (LiMn₂O₄ spinel) — 3.00 V → 4.20 V
# Characteristic double-plateau at ~4.00 V and ~4.06 V (two Mn phase transitions)
_LMO_SOC_OCV = np.array([
    [0.00, 3.000],
    [0.05, 3.550],
    [0.10, 3.700],
    [0.20, 3.900],
    [0.30, 3.950],
    [0.40, 3.980],
    [0.45, 4.000],
    [0.50, 4.010],
    [0.55, 4.035],
    [0.60, 4.060],
    [0.70, 4.090],
    [0.80, 4.120],
    [0.90, 4.150],
    [0.95, 4.170],
    [1.00, 4.200],
])


# SSB — Li-metal anode (or Si anode) + solid electrolyte + NMC-like cathode
# Voltage range 3.0 V → 4.35 V (Li-metal ~0.15 V lower potential than graphite)
# Solid electrolyte gives very high T_runaway but steep Arrhenius at low T.
_SSB_SOC_OCV = np.array([
    [0.00, 3.000],
    [0.05, 3.380],
    [0.10, 3.600],
    [0.20, 3.740],
    [0.30, 3.830],
    [0.40, 3.920],
    [0.50, 4.000],
    [0.60, 4.080],
    [0.70, 4.140],
    [0.80, 4.200],
    [0.90, 4.270],
    [0.95, 4.310],
    [1.00, 4.350],
])


# ── Per-chemistry property tables ───────────────────────────────────────────
CHEMISTRY_PROPS: dict[str, dict] = {
    "nmc": {
        "ocv_table": _NMC_SOC_OCV,
        "arrhenius_K": 4000.0,
        "temp_coeff_V_per_K": -5e-4,
        "v_min": 3.0,
        "v_max": 4.2,
        "v_overcharge": 4.25,
        "v_dropout": 0.5,
        "T_runaway_C": 70.0,
        "nominal_voltage_V": 3.7,
        "default_capacity_Ah": 2.3,
        "self_discharge_pct_per_month": 1.5,
        "default_ecm": {
            "R0": 0.025, "R1": 0.015, "C1": 2000.0,
            "R2": 0.030, "C2": 8000.0,
        },
    },
    "lfp": {
        "ocv_table": _LFP_SOC_OCV,
        "arrhenius_K": 3500.0,
        "temp_coeff_V_per_K": -3e-4,
        "v_min": 2.5,
        "v_max": 3.65,
        "v_overcharge": 3.70,
        "v_dropout": 1.5,
        "T_runaway_C": 90.0,
        "nominal_voltage_V": 3.2,
        "default_capacity_Ah": 3.2,
        "self_discharge_pct_per_month": 3.0,
        "default_ecm": {
            "R0": 0.020, "R1": 0.012, "C1": 3000.0,
            "R2": 0.025, "C2": 10000.0,
        },
    },
    "lmfp": {
        "ocv_table": _LMFP_SOC_OCV,
        "arrhenius_K": 3700.0,          # between LFP (3500) and NMC (4000)
        "temp_coeff_V_per_K": -4e-4,
        "v_min": 2.8,
        "v_max": 4.15,
        "v_overcharge": 4.20,
        "v_dropout": 1.2,
        "T_runaway_C": 85.0,            # more stable than NMC, less than LFP
        "nominal_voltage_V": 3.7,
        "default_capacity_Ah": 3.0,
        "self_discharge_pct_per_month": 2.5,
        "default_ecm": {
            "R0": 0.018, "R1": 0.011, "C1": 2500.0,
            "R2": 0.022, "C2": 9000.0,
        },
    },
    "lto": {
        "ocv_table": _LTO_SOC_OCV,
        "arrhenius_K": 3000.0,          # least temperature-sensitive anode
        "temp_coeff_V_per_K": -2e-4,    # very low OCV temp sensitivity
        "v_min": 1.5,
        "v_max": 2.8,
        "v_overcharge": 2.85,
        "v_dropout": 0.5,
        "T_runaway_C": 95.0,            # extremely thermally stable (zero-strain)
        "nominal_voltage_V": 2.3,
        "default_capacity_Ah": 3.5,
        "self_discharge_pct_per_month": 1.0,
        "default_ecm": {
            "R0": 0.008, "R1": 0.005, "C1": 5000.0,   # very low R0 → superb power
            "R2": 0.010, "C2": 15000.0,
        },
    },
    "nca": {
        "ocv_table": _NCA_SOC_OCV,
        "arrhenius_K": 4500.0,          # most temperature-sensitive cathode
        "temp_coeff_V_per_K": -5e-4,
        "v_min": 3.0,
        "v_max": 4.2,
        "v_overcharge": 4.25,
        "v_dropout": 0.5,
        "T_runaway_C": 65.0,            # less stable than NMC — lower onset temp
        "nominal_voltage_V": 3.65,
        "default_capacity_Ah": 3.0,
        "self_discharge_pct_per_month": 1.5,
        "default_ecm": {
            "R0": 0.022, "R1": 0.013, "C1": 1800.0,
            "R2": 0.028, "C2": 7500.0,
        },
    },
    "lmo": {
        "ocv_table": _LMO_SOC_OCV,
        "arrhenius_K": 3800.0,
        "temp_coeff_V_per_K": -4.5e-4,
        "v_min": 3.0,
        "v_max": 4.2,
        "v_overcharge": 4.25,
        "v_dropout": 0.5,
        "T_runaway_C": 55.0,            # Mn dissolution accelerates at moderate T
        "nominal_voltage_V": 3.8,
        "default_capacity_Ah": 2.0,
        "self_discharge_pct_per_month": 5.5,   # highest — Mn dissolution effect
        "default_ecm": {
            "R0": 0.030, "R1": 0.018, "C1": 1500.0,
            "R2": 0.035, "C2": 6000.0,
        },
    },
    "ssb": {
        "ocv_table": _SSB_SOC_OCV,
        # Solid electrolyte ionic conductivity drops steeply below 0 °C.
        # Arrhenius_K ≈ 5500 K — highest of all chemistries.
        "arrhenius_K": 5500.0,
        "temp_coeff_V_per_K": -4e-4,
        "v_min": 3.0,
        "v_max": 4.35,
        "v_overcharge": 4.40,
        "v_dropout": 0.5,
        # Solid electrolyte is non-flammable → thermal runaway onset >>150 °C.
        "T_runaway_C": 150.0,
        "nominal_voltage_V": 3.85,
        "default_capacity_Ah": 4.0,     # higher energy density vs. liquid Li-ion
        "self_discharge_pct_per_month": 0.3,   # lowest — no liquid electrolyte shuttle
        "default_ecm": {
            "R0": 0.030, "R1": 0.018, "C1": 1500.0,  # R0 higher but very T-sensitive
            "R2": 0.025, "C2": 5000.0,
        },
    },
}


def get_chemistry_props(chemistry: str | CellChemistry) -> dict:
    """Return the property dict for *chemistry* (case-insensitive string or enum).

    Parameters
    ----------
    chemistry : str or CellChemistry
        ``"nmc"``, ``"lfp"``, ``"lmfp"``, ``"lto"``, ``"nca"``, ``"lmo"``,
        or a :class:`CellChemistry` enum member.

    Returns
    -------
    dict
        Keys: ``ocv_table``, ``arrhenius_K``, ``temp_coeff_V_per_K``,
        ``v_min``, ``v_max``, ``v_overcharge``, ``v_dropout``,
        ``T_runaway_C``, ``nominal_voltage_V``, ``default_capacity_Ah``,
        ``self_discharge_pct_per_month``, ``default_ecm``.
    """
    key = chemistry.value if isinstance(chemistry, CellChemistry) else str(chemistry).lower()
    if key not in CHEMISTRY_PROPS:

        raise ValueError(
            f"Unknown chemistry {chemistry!r}. "
            f"Valid options: {sorted(CHEMISTRY_PROPS)}"
        )
    return CHEMISTRY_PROPS[key]
