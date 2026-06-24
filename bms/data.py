"""
Load profile generation and dataset adapters.

`generate_load_profile`
    Produces a synthetic but realistic discharge / regen current trace
    suitable for SOC-estimation benchmarking. Three pre-built modes:

    * "constant"  — flat C-rate discharge.
    * "pulse"     — square pulses, rest periods (HPPC-like).
    * "drive"     — UDDS-like piecewise-stochastic profile with regen.

`load_nasa_like_dataset`
    Loads a NASA PCoE Battery Aging dataset CSV if a local path is given,
    otherwise synthesises a schema-equivalent trace so the rest of the
    pipeline can run end-to-end without internet access. The synthetic
    trace now includes both ``capacity_Ah`` and ``resistance_mOhm`` columns
    to support the dual-fade RUL model in `fmea.estimate_rul_with_resistance`.

`generate_aging_profile`
    Generates a synthetic multi-temperature aging profile with realistic
    capacity fade and resistance growth, suitable for SoH tracking and
    predictive-maintenance demonstrations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
def generate_load_profile(duration_s: float, dt: float = 1.0,
                          mode: str = "drive", c_rate: float = 1.0,
                          capacity_Ah: float = 2.3, seed: int = 0) -> np.ndarray:
    """Return a current vector (positive = discharge, A)."""
    n = int(duration_s / dt)
    t = np.arange(n) * dt
    rng = np.random.default_rng(seed)
    I_nom = c_rate * capacity_Ah

    if mode == "constant":
        return np.full(n, I_nom)

    if mode == "pulse":
        i = np.zeros(n)
        period = max(60, int(120 / dt))
        on = max(30, int(60 / dt))
        for k in range(0, n, period):
            i[k:k + on] = I_nom * (0.5 + rng.uniform(0.0, 1.5))
        return i

    if mode == "drive":
        i = np.zeros(n)
        k = 0
        while k < n:
            seg_len = max(1, int(rng.integers(20, 200) / dt))
            kind = rng.choice(["cruise", "accel", "regen", "idle"], p=[0.45, 0.25, 0.10, 0.20])
            if kind == "cruise":
                level = I_nom * rng.uniform(0.4, 0.9)
            elif kind == "accel":
                level = I_nom * rng.uniform(1.2, 2.5)
            elif kind == "regen":
                level = -I_nom * rng.uniform(0.3, 0.8)
            else:
                level = 0.0
            i[k:k + seg_len] = level
            k += seg_len
        i += rng.normal(0, 0.05 * I_nom, n)
        return i

    raise ValueError(f"unknown mode {mode!r}")


# ----------------------------------------------------------------------
def generate_cccv_profile(
    Q_nom_Ah: float = 2.3,
    soc_start: float = 0.20,
    v_max: float | None = None,
    i_charge_C: float = 0.5,
    i_taper_C: float = 0.05,
    r0_ohm: float = 0.025,
    dt: float = 1.0,
    max_duration_s: float = 14_400.0,
    chemistry: str = "nmc",
) -> np.ndarray:
    """Simulate a CC-CV charge current profile.

    Implements the standard constant-current / constant-voltage lithium-ion
    charge algorithm:

    1. **CC phase** — charge at ``i_charge_C × Q_nom_Ah`` until the estimated
       terminal voltage reaches ``v_max``.
    2. **CV phase** — taper current while holding V = ``v_max`` until the
       current falls below ``i_taper_C × Q_nom_Ah``.

    Parameters
    ----------
    Q_nom_Ah : float
        Cell nominal capacity [Ah].
    soc_start : float
        Initial SOC for the charging session (0–1).
    v_max : float, optional
        Charge cut-off voltage [V].  Defaults to the chemistry ``v_max``.
    i_charge_C : float
        CC-phase charge rate [C].  0.5 = C/2 charge.
    i_taper_C : float
        CV-phase termination current rate [C].  0.05 = C/20.
    r0_ohm : float
        DC internal resistance used for terminal-voltage estimation [Ω].
    dt : float
        Time step [s].
    max_duration_s : float
        Hard maximum duration; profile is truncated here if not terminated.
    chemistry : str
        Cell chemistry, used to look up ``v_max`` when not specified.

    Returns
    -------
    np.ndarray
        Current profile [A], shape ``(N,)``.  Convention: **negative = charge**.
    """
    from .ocv_soc import OCVSOC
    from .chemistry import get_chemistry_props

    if v_max is None:
        v_max = get_chemistry_props(chemistry)["v_max"]

    n_max = int(max_duration_s / dt)
    oc = OCVSOC.from_chemistry(chemistry)
    I_cc = -i_charge_C * Q_nom_Ah       # negative = charge
    I_taper = -i_taper_C * Q_nom_Ah     # negative

    i_out: list[float] = []
    soc = float(np.clip(soc_start, 0.0, 1.0))
    phase = "cc"

    for _ in range(n_max):
        ocv = float(oc.ocv(soc))

        if phase == "cc":
            # V_terminal = OCV − I·R0 ; with I < 0, V_t > OCV
            v_t = ocv - I_cc * r0_ohm
            if v_t >= v_max or soc >= 0.999:
                phase = "cv"

        if phase == "cc":
            i_out.append(I_cc)
            soc = min(soc - I_cc * dt / (Q_nom_Ah * 3600.0), 1.0)
        else:
            # CV: hold V = v_max → I = (OCV − v_max) / R0 (negative during charge)
            I_now = (ocv - v_max) / max(r0_ohm, 1e-6)
            if I_now >= I_taper or soc >= 0.9999:
                break
            i_out.append(I_now)
            soc = min(soc - I_now * dt / (Q_nom_Ah * 3600.0), 1.0)

    return np.array(i_out) if i_out else np.array([0.0])


# ----------------------------------------------------------------------
def generate_power_profile(duration_s: float, dt: float = 1.0,
                           mode: str = "drive", p_rate: float = 1.0,
                           nominal_capacity_Ah: float = 2.3,
                           nominal_voltage_V: float = 3.7,
                           seed: int = 0) -> np.ndarray:
    """Return a power profile in Watts (positive = discharge).

    Generates a load profile in terms of *power* rather than current.
    Internally calls :func:`generate_load_profile` with the corresponding
    C-rate and scales by ``nominal_voltage_V``.

    Parameters
    ----------
    duration_s : float
        Profile duration [s].
    dt : float
        Time step [s].
    mode : str
        ``"constant"``, ``"pulse"``, or ``"drive"`` (same as
        :func:`generate_load_profile`).
    p_rate : float
        Power rate multiplier (analogous to C-rate but in power).
        ``p_rate = 1`` → ``P_nom = capacity_Ah × nominal_voltage_V`` W.
    nominal_capacity_Ah : float
        Cell capacity [Ah] used to scale the nominal power.
    nominal_voltage_V : float
        Nominal cell voltage [V] used to convert current → power.
    seed : int
        RNG seed.

    Returns
    -------
    np.ndarray
        Power profile, shape (N,) [W].
    """
    i_profile = generate_load_profile(
        duration_s, dt=dt, mode=mode, c_rate=p_rate,
        capacity_Ah=nominal_capacity_Ah, seed=seed,
    )
    return i_profile * nominal_voltage_V


# ----------------------------------------------------------------------
def load_nasa_like_dataset(path: str | Path | None = None,
                           cycles: int = 50, capacity_Ah: float = 2.3,
                           seed: int = 0) -> pd.DataFrame:
    """Return a per-cycle ageing record.

    Columns: ``cycle``, ``capacity_Ah``, ``resistance_mOhm``,
    ``temperature_C``, ``source``.

    If a CSV ``path`` is provided it is read directly (must contain the
    above columns, except ``resistance_mOhm`` which is back-filled if
    absent). Otherwise a synthetic trace is generated with realistic
    square-root-of-cycles capacity fade and linear resistance growth.

    Parameters
    ----------
    path : str or Path, optional
        CSV file path.  If None, synthetic data is generated.
    cycles : int
        Number of cycles to synthesise (ignored when path is given).
    capacity_Ah : float
        Nominal fresh-cell capacity [Ah].
    seed : int
        RNG seed for reproducibility.
    """
    if path is not None:
        df = pd.read_csv(path)
        df["source"] = df.get("source", "csv")
        if "resistance_mOhm" not in df.columns:
            # Back-fill with a nominal value so downstream code always has it.
            df["resistance_mOhm"] = 25.0
        return df

    rng = np.random.default_rng(seed)
    n_arr = np.arange(1, cycles + 1, dtype=float)
    sqrtN = np.sqrt(n_arr)

    # Capacity fade: Q(N) = Q0 * (1 - 0.02*sqrt(N)) + noise
    alpha = 0.02
    cap = capacity_Ah * (1 - alpha * sqrtN) + rng.normal(0, 0.005, cycles)
    cap = np.maximum(cap, 0.5 * capacity_Ah)

    # Resistance growth: R(N) = R0 * (1 + 0.015*sqrt(N)) + noise [mΩ]
    R0_mOhm = 25.0
    beta = 0.015
    res = R0_mOhm * (1 + beta * sqrtN) + rng.normal(0, 0.4, cycles)
    res = np.maximum(res, R0_mOhm)

    T = 25.0 + rng.normal(0, 1.5, cycles)

    return pd.DataFrame({
        "cycle": n_arr.astype(int),
        "capacity_Ah": cap,
        "resistance_mOhm": res,
        "temperature_C": T,
        "source": "synthetic",
    })


# ----------------------------------------------------------------------
def generate_aging_profile(cycles: int = 100,
                           capacity_Ah: float = 2.3,
                           r0_mOhm: float = 25.0,
                           temperatures_C: list[float] | None = None,
                           seed: int = 0) -> pd.DataFrame:
    """Generate a multi-temperature synthetic aging profile.

    Simulates capacity fade and resistance growth across a sequence of
    temperature windows. Temperature accelerates both degradation mechanisms
    via an Arrhenius term (higher T → faster degradation).

    Parameters
    ----------
    cycles : int
        Total cycle count to generate.
    capacity_Ah : float
        Fresh-cell rated capacity [Ah].
    r0_mOhm : float
        Fresh-cell DC internal resistance [mΩ].
    temperatures_C : list of float, optional
        List of test temperatures cycled through.  Defaults to [25, 35, 45].
    seed : int
        RNG seed.

    Returns
    -------
    DataFrame with columns: ``cycle``, ``capacity_Ah``, ``resistance_mOhm``,
    ``temperature_C``, ``source``, ``soh_capacity``, ``soh_resistance``.
    """
    if temperatures_C is None:
        temperatures_C = [25.0, 35.0, 45.0]

    rng = np.random.default_rng(seed)
    n_arr = np.arange(1, cycles + 1, dtype=float)
    sqrtN = np.sqrt(n_arr)

    # Assign a temperature to each cycle (cycle through the list).
    T_per_cycle = np.array([temperatures_C[i % len(temperatures_C)]
                             for i in range(cycles)])

    # Arrhenius acceleration relative to 25 °C (Ea/R ≈ 4000 K).
    T_ref_K = 298.15
    T_K = T_per_cycle + 273.15
    arrh = np.exp(4000.0 * (1.0 / T_ref_K - 1.0 / T_K))  # <1 at low T, >1 at high T

    # Cumulative "equivalent cycles at 25 °C" for the fade models.
    equiv_N = np.cumsum(arrh)
    sqrtN_eff = np.sqrt(equiv_N)

    alpha = 0.018   # capacity fade coefficient at 25 °C reference
    beta = 0.012    # resistance growth coefficient at 25 °C reference

    cap = capacity_Ah * (1 - alpha * sqrtN_eff) + rng.normal(0, 0.004, cycles)
    cap = np.maximum(cap, 0.5 * capacity_Ah)

    res = r0_mOhm * (1 + beta * sqrtN_eff) + rng.normal(0, 0.3, cycles)
    res = np.maximum(res, r0_mOhm)

    soh_cap = cap / capacity_Ah
    soh_res = r0_mOhm / res

    return pd.DataFrame({
        "cycle": n_arr.astype(int),
        "capacity_Ah": cap,
        "resistance_mOhm": res,
        "temperature_C": T_per_cycle,
        "source": "synthetic_multi_temp",
        "soh_capacity": soh_cap,
        "soh_resistance": soh_res,
    })
