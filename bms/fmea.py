"""
Failure Modes and Effects Analysis (FMEA) and remaining-useful-life (RUL).

The FMEA table follows AIAG-VDA conventions: each row scores a failure
mode on Severity (S), Occurrence (O) and Detection (D), each on a 1-10
scale, and computes the Risk Priority Number RPN = S × O × D.

`estimate_rul` implements a simple square-root-of-time capacity-fade model
used widely in lithium-ion ageing studies:

    Q(N) = Q0 · (1 - α · √N)

where N is the equivalent full-cycle count and α is calibrated against the
historical capacity trace. The RUL is the additional cycle count before
capacity drops below `eol_threshold` × Q0 (typically 0.8).

`estimate_rul_with_resistance` extends this by fitting a dual fade model:

    Q(N) = Q0 · (1 - α · √N)          # capacity fade
    R(N) = R0 · (1 + β · √N)          # resistance growth

RUL is the minimum of the remaining cycles to either EoL threshold.  This
is a better predictor of actual end-of-life than capacity alone, because
rising internal resistance degrades power capability (peak power ∝ 1/R)
even before capacity reaches 80%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# FMEA — default table for a Li-ion BMS
# ----------------------------------------------------------------------
_DEFAULT_FMEA = [
    # mode, effect, cause, S, O, D, controls
    ("Overcharge",            "Cell venting / fire",           "Charger fault, cell delta",      10, 3, 3, "Voltage limit + redundant cutoff"),
    ("Over-discharge",        "Capacity fade, copper plating", "Excess load, low-SOC drive",      7, 4, 4, "Cell minimum voltage cutoff"),
    ("External short circuit","Massive current, fire",         "Pack abuse, contamination",      10, 2, 2, "Fuse, contactor opening"),
    ("Internal short circuit","Thermal runaway",               "Manuf. defect, dendrites",       10, 2, 7, "Voltage drop trend monitor"),
    ("Thermal runaway",       "Pack fire, propagation",        "Cell short, overcharge",         10, 2, 6, "Thermal sensors + propagation barriers"),
    ("Cell imbalance",        "Reduced usable capacity",       "Manufacturing scatter, ageing",   4, 7, 3, "Active balancing strategy"),
    ("Voltage sensor failure","Loss of state observability",   "Wire breakage, ADC fault",        7, 4, 5, "Plausibility checks vs neighbours"),
    ("Voltage sensor bias",   "Drifting SOC estimation",       "ADC drift, reference drift",      6, 5, 7, "Periodic calibration / OCV anchor"),
    ("Temperature sensor failure","Thermal blind spot",        "Wire breakage, NTC failure",      8, 3, 4, "Redundant sensors + range checks"),
    ("Coolant pump failure",  "Pack overheating",              "Pump fault, leak",                9, 3, 3, "Flow sensor + secondary path"),
    ("BMS firmware fault",    "Loss of supervision",           "Memory corruption, bug",          9, 2, 6, "Watchdog + redundant MCU"),
    ("Capacity fade (ageing)","RUL exhaustion",                "Cycling and calendar ageing",     5, 9, 4, "RUL estimator + maintenance"),
]


def build_fmea_table(rows: list[tuple] | None = None) -> pd.DataFrame:
    """Return a pandas DataFrame with an extra `RPN` column."""
    cols = ["failure_mode", "effect", "cause", "S", "O", "D", "controls"]
    rows = rows or _DEFAULT_FMEA
    df = pd.DataFrame(rows, columns=cols)
    df["RPN"] = df["S"] * df["O"] * df["D"]
    return df.sort_values("RPN", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# RUL — square-root-of-time capacity-fade model
# ----------------------------------------------------------------------
def estimate_rul(cycle_counts: np.ndarray,
                 capacity_history_Ah: np.ndarray,
                 nominal_capacity_Ah: float,
                 eol_threshold: float = 0.8) -> dict:
    """Fit Q(N) = Q0 (1 - α √N) and return RUL in equivalent full cycles."""
    cycle_counts = np.asarray(cycle_counts, float)
    capacity_history_Ah = np.asarray(capacity_history_Ah, float)
    if len(cycle_counts) < 2 or len(cycle_counts) != len(capacity_history_Ah):
        raise ValueError("cycle_counts and capacity_history_Ah must be same length, ≥ 2")

    Q0 = float(nominal_capacity_Ah)
    # y = 1 - Q/Q0 = α √N  ⇒  α = ⟨y · √N⟩ / ⟨N⟩
    y = 1.0 - capacity_history_Ah / Q0
    sqrtN = np.sqrt(np.maximum(cycle_counts, 1e-9))
    alpha = float(np.dot(y, sqrtN) / max(np.dot(sqrtN, sqrtN), 1e-12))

    eol_y = 1.0 - eol_threshold
    N_eol = (eol_y / alpha) ** 2 if alpha > 0 else np.inf
    N_now = float(cycle_counts[-1])
    rul = max(N_eol - N_now, 0.0)

    soh = float(capacity_history_Ah[-1] / Q0)
    fit_rmse = float(np.sqrt(np.mean((Q0 * (1 - alpha * sqrtN) - capacity_history_Ah) ** 2)))
    return {
        "alpha": alpha,
        "soh": soh,
        "cycles_now": N_now,
        "cycles_to_eol": float(N_eol),
        "rul_cycles": float(rul),
        "fit_rmse_Ah": fit_rmse,
    }


# ----------------------------------------------------------------------
# Dual fade model: capacity + resistance
# ----------------------------------------------------------------------
def estimate_rul_with_resistance(
    cycle_counts: np.ndarray,
    capacity_history_Ah: np.ndarray,
    resistance_history_mOhm: np.ndarray,
    nominal_capacity_Ah: float,
    nominal_resistance_mOhm: float,
    eol_capacity_threshold: float = 0.8,
    eol_resistance_threshold: float = 1.5,
) -> dict:
    """Fit capacity-fade AND resistance-growth models; RUL is the limiting.

    Models
    ------
    * Capacity : Q(N) = Q0 · (1 − α · √N)
    * Resistance: R(N) = R0 · (1 + β · √N)

    End-of-life is the earlier of:
    * Capacity falls below ``eol_capacity_threshold × Q0``   (typically 80%)
    * Resistance rises above ``eol_resistance_threshold × R0``  (typically 150%)

    Parameters
    ----------
    cycle_counts : (N,) array
        Sorted array of equivalent full-cycle counts.
    capacity_history_Ah : (N,) array
        Measured discharge capacity at each cycle.
    resistance_history_mOhm : (N,) array
        Measured DC internal resistance [mΩ] at each cycle.
    nominal_capacity_Ah : float
        Fresh-cell rated capacity Q0.
    nominal_resistance_mOhm : float
        Fresh-cell internal resistance R0 [mΩ].
    eol_capacity_threshold : float
        Fraction of Q0 that defines capacity EoL (default 0.8).
    eol_resistance_threshold : float
        Multiple of R0 that defines resistance EoL (default 1.5 = 50% rise).

    Returns
    -------
    dict with keys:
        alpha, beta, soh_capacity, soh_resistance,
        cycles_now, cycles_to_cap_eol, cycles_to_res_eol,
        cycles_to_eol (minimum), rul_cycles,
        limiting_mode ("capacity" | "resistance"),
        fit_rmse_cap_Ah, fit_rmse_res_mOhm.
    """
    cycles = np.asarray(cycle_counts, float)
    cap = np.asarray(capacity_history_Ah, float)
    res = np.asarray(resistance_history_mOhm, float)
    if len(cycles) < 2:
        raise ValueError("Need at least 2 data points")

    Q0 = float(nominal_capacity_Ah)
    R0 = float(nominal_resistance_mOhm)
    sqrtN = np.sqrt(np.maximum(cycles, 1e-9))

    # --- Capacity fade fit: y = 1 − Q/Q0 = α·√N ---
    y_cap = 1.0 - cap / Q0
    alpha = float(np.dot(y_cap, sqrtN) / max(np.dot(sqrtN, sqrtN), 1e-12))
    eol_cap_y = 1.0 - eol_capacity_threshold
    N_cap_eol = (eol_cap_y / alpha) ** 2 if alpha > 0 else np.inf

    # --- Resistance growth fit: z = R/R0 − 1 = β·√N ---
    z_res = res / R0 - 1.0
    beta = float(np.dot(z_res, sqrtN) / max(np.dot(sqrtN, sqrtN), 1e-12))
    eol_res_z = eol_resistance_threshold - 1.0
    N_res_eol = (eol_res_z / beta) ** 2 if beta > 0 else np.inf

    N_now = float(cycles[-1])
    N_eol = min(N_cap_eol, N_res_eol)
    rul = max(N_eol - N_now, 0.0)
    limiting = "capacity" if N_cap_eol <= N_res_eol else "resistance"

    soh_cap = float(cap[-1] / Q0)
    soh_res = float(R0 / max(res[-1], 1e-9))   # power-delivery SoH proxy

    rmse_cap = float(np.sqrt(np.mean((Q0 * (1 - alpha * sqrtN) - cap) ** 2)))
    rmse_res = float(np.sqrt(np.mean((R0 * (1 + beta * sqrtN) - res) ** 2)))

    return {
        "alpha": alpha,
        "beta": beta,
        "soh_capacity": soh_cap,
        "soh_resistance": soh_res,
        "cycles_now": N_now,
        "cycles_to_cap_eol": float(N_cap_eol),
        "cycles_to_res_eol": float(N_res_eol),
        "cycles_to_eol": float(N_eol),
        "rul_cycles": float(rul),
        "limiting_mode": limiting,
        "fit_rmse_cap_Ah": rmse_cap,
        "fit_rmse_res_mOhm": rmse_res,
    }
