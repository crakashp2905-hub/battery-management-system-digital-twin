"""
Real-dataset validation: loaders, a synthetic fixture, and an estimator
leaderboard built on the model-agnostic registry.

The library's accuracy claims are otherwise model-vs-model; this module makes it
trivial to validate against **real** cycling data.  Two recommended sources:

* **SoC** — LG 18650 drive cycles (Kollmeyer / McMaster, Mendeley): UDDS/US06/…
  at several temperatures, with coulomb-counted SoC ground truth → `load_drivecycle_csv`.
* **SoH** — NASA PCoE (18650 cycled to failure, `.mat`) → `nasa_mat_to_capacity`
  → `load_capacity_fade_csv`; MIT-Stanford (124 LFP fast-charged) for fast-charge aging.

See :data:`DATASET_SOURCES` for URLs.  Because the raw datasets are large, none
are committed — drop the files in and the loaders + `estimator_leaderboard` run
unchanged.  A committed synthetic fixture (`data/samples/`) plus
:func:`synthetic_drivecycle` keep the tests green without them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

DATASET_SOURCES: dict[str, dict] = {
    "lg_18650": {
        "use": "SoC estimator leaderboard",
        "cells": "LG 18650HG2 — UDDS / US06 / HWFET / LA92 drive cycles at several temperatures",
        "ground_truth": "coulomb-counted SoC",
        "format": "CSV",
        "url": "https://data.mendeley.com/datasets/cp3473x7xv",
        "loader": "load_drivecycle_csv(path, columns=LG_COLUMN_MAP)",
    },
    "nasa_pcoe": {
        "use": "SoH / capacity fade + RUL",
        "cells": "18650 cycled to failure with periodic capacity & impedance",
        "format": "MATLAB .mat",
        "url": "https://www.nasa.gov/intelligent-systems-division (PCoE Battery Data Set)",
        "loader": "nasa_mat_to_capacity(path)",
    },
    "mit_stanford": {
        "use": "fast-charge aging (LFP)",
        "cells": "124 LFP cells fast-charged to failure (Severson et al. 2019)",
        "format": "MATLAB .mat / batch",
        "url": "https://data.matr.io/1",
        "loader": "(custom) — capacity per cycle → load_capacity_fade_csv",
    },
}

# Example column mapping for the LG 18650 (Kollmeyer) CSVs — override as needed.
LG_COLUMN_MAP = {
    "time_s": "Time Stamp",
    "current_A": "Current",
    "voltage_V": "Voltage",
    "temperature_C": "Temperature",
}


# ======================================================================
@dataclass
class DriveCycleData:
    """A drive-cycle time series with SoC ground truth for estimator validation.

    Sign convention: ``current_A`` positive = discharge.  ``soc_true`` is in
    [0, 1].
    """

    time_s: np.ndarray
    current_A: np.ndarray
    voltage_V: np.ndarray
    temperature_C: np.ndarray
    soc_true: np.ndarray
    capacity_Ah: float
    chemistry: str = "nmc"
    name: str = ""
    source: str = "synthetic"     # "synthetic" | "real" — keep results separate

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.time_s))) if len(self.time_s) > 1 else 1.0

    @property
    def n(self) -> int:
        return len(self.current_A)


# ======================================================================
def synthetic_drivecycle(chemistry: str = "nmc", duration_s: float = 1800.0,
                         dt: float = 1.0, soc0: float = 0.9, seed: int = 0,
                         noise_v: float = 0.005,
                         current_bias_A: float = 0.0,
                         temperature_C: float = 25.0) -> DriveCycleData:
    """Generate a physically-consistent drive cycle from the ECM (the fixture).

    The plant ECM's SoC is the ground truth, so any estimator can be scored
    against it exactly.  ``current_bias_A`` adds a sensor bias to the *measured*
    current (voltage and true SoC come from the unbiased current) — the classic
    case where a Coulomb counter drifts but a voltage-feedback filter does not.

    ``temperature_C`` sets the (isothermal) cell temperature the plant runs at:
    the ECM resistances follow their Arrhenius shift and the OCV its temperature
    coefficient, so the generated voltage is genuinely cold/hot.  It is written
    into :attr:`DriveCycleData.temperature_C`, so a temperature-aware estimator
    can be scored against a cold or hot trace.  Defaults to 25 °C (the reference
    temperature), which reproduces the previous behaviour exactly.
    """
    from .chemistry import get_chemistry_props
    from .data import generate_load_profile
    from .ecm import ECMParameters, SecondOrderECM
    from .ocv_soc import OCVSOC

    props = get_chemistry_props(chemistry)
    cap = float(props["default_capacity_Ah"])
    d = props["default_ecm"]
    params = ECMParameters(R0=d["R0"], R1=d["R1"], C1=d["C1"], R2=d["R2"], C2=d["C2"],
                           Q_nom_Ah=cap, chemistry=chemistry)
    ocv = OCVSOC.from_chemistry(chemistry)
    ecm = SecondOrderECM(params=params, ocv_curve=ocv)
    ecm.reset(soc0)

    current = np.asarray(generate_load_profile(
        duration_s, dt=dt, mode="drive", c_rate=1.0, capacity_Ah=cap, seed=seed), float)
    temps = np.full(len(current), float(temperature_C))
    sim = ecm.simulate(current, dt=dt, temperatures=temps)
    rng = np.random.default_rng(seed)
    voltage = sim["v_terminal"] + rng.normal(0.0, noise_v, len(current))
    measured_current = current + float(current_bias_A)   # sensor bias, if any
    t = np.arange(len(current)) * dt
    return DriveCycleData(
        time_s=t, current_A=measured_current, voltage_V=voltage,
        temperature_C=temps, soc_true=sim["soc"],
        capacity_Ah=cap, chemistry=chemistry, name=f"synthetic_{chemistry}")


def save_drivecycle_csv(data: DriveCycleData, path) -> None:
    """Write a :class:`DriveCycleData` to CSV (round-trips with load_drivecycle_csv)."""
    import pandas as pd
    pd.DataFrame({
        "time_s": data.time_s, "current_A": data.current_A,
        "voltage_V": data.voltage_V, "temperature_C": data.temperature_C,
        "soc_true": data.soc_true,
    }).to_csv(path, index=False)


def load_drivecycle_csv(path, chemistry: str = "nmc", capacity_Ah: float | None = None,
                        columns: dict | None = None, soc0: float = 1.0,
                        name: str = "") -> DriveCycleData:
    """Load a drive cycle from CSV (real LG data or the synthetic fixture).

    ``columns`` remaps source column names onto
    ``time_s`` / ``current_A`` / ``voltage_V`` / ``temperature_C`` / ``soc_true``
    (see :data:`LG_COLUMN_MAP`).  If SoC is absent it is coulomb-counted from the
    current and ``capacity_Ah``; if temperature is absent, 25 °C is assumed.
    """
    import pandas as pd

    from .chemistry import get_chemistry_props

    df = pd.read_csv(path)
    cols = {"time_s": "time_s", "current_A": "current_A", "voltage_V": "voltage_V",
            "temperature_C": "temperature_C", "soc_true": "soc_true"}
    if columns:
        cols.update(columns)

    time_s = df[cols["time_s"]].to_numpy(float)
    current = df[cols["current_A"]].to_numpy(float)
    voltage = df[cols["voltage_V"]].to_numpy(float)
    temp = (df[cols["temperature_C"]].to_numpy(float)
            if cols["temperature_C"] in df.columns else np.full(len(df), 25.0))
    cap = float(capacity_Ah if capacity_Ah is not None
                else get_chemistry_props(chemistry)["default_capacity_Ah"])

    if cols["soc_true"] in df.columns:
        soc = df[cols["soc_true"]].to_numpy(float)
    else:
        dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 1.0
        soc = np.clip(soc0 - np.cumsum(current) * dt / (cap * 3600.0), 0.0, 1.0)

    return DriveCycleData(time_s=time_s, current_A=current, voltage_V=voltage,
                          temperature_C=temp, soc_true=soc, capacity_Ah=cap,
                          chemistry=chemistry, name=name or str(path), source="real")


# ======================================================================
def estimator_leaderboard(data: DriveCycleData, estimators: list[str] | None = None,
                          temperature_aware: bool = True,
                          hysteresis_aware: bool = True):
    """Run each estimator on *data* and rank them by SoC RMSE.

    Returns a DataFrame indexed by estimator with ``rmse``, ``mae``, ``max_err``,
    and ``runtime_s`` — the head-to-head accuracy table on a real (or synthetic)
    trace.  Defaults to the recursive estimators (the LSTM needs separate training).

    ``temperature_aware`` (default) feeds ``data.temperature_C`` to every
    estimator whose ``run`` accepts a ``temperatures`` argument, so on a cold or
    hot trace the filter uses the correct Arrhenius-shifted ECM and OCV.  Set it
    ``False`` to score temperature-*naive* filters (assuming 25 °C) against the
    same trace — the two runs quantify the value of a temperature sensor.

    ``hysteresis_aware`` (default) gives the estimator's OCV the chemistry's
    characteristic hysteresis; ``False`` forces a hysteresis-free OCV (matters
    on flat-OCV chemistries like LFP, whose hysteresis dominates the sparse OCV
    slope) — the two runs quantify the value of modelling hysteresis.
    """
    import inspect

    import pandas as pd

    from .chemistry import get_chemistry_props
    from .ecm import ECMParameters
    from .estimation import make_soc_estimator
    from .ocv_soc import OCVSOC

    names = estimators or ["coulomb", "ekf", "ukf", "joint_ekf"]
    d = get_chemistry_props(data.chemistry)["default_ecm"]
    params = ECMParameters(R0=d["R0"], R1=d["R1"], C1=d["C1"], R2=d["R2"], C2=d["C2"],
                           Q_nom_Ah=data.capacity_Ah, chemistry=data.chemistry)
    ocv = OCVSOC.from_chemistry(data.chemistry,
                                hysteresis_v=None if hysteresis_aware else 0.0)

    rows = []
    for name in names:
        est = make_soc_estimator(name, params=params, ocv_curve=ocv,
                                 capacity_Ah=data.capacity_Ah)
        est.reset(float(data.soc_true[0]))
        # Pass the trace temperature only to estimators that accept it.
        kw = {}
        if temperature_aware and "temperatures" in inspect.signature(est.run).parameters:
            kw["temperatures"] = data.temperature_C
        t0 = time.perf_counter()
        soc_hat = np.asarray(est.run(data.current_A, data.voltage_V, data.dt, **kw), float)
        wall = time.perf_counter() - t0
        err = soc_hat - data.soc_true
        rows.append({
            "estimator": name,
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "max_err": float(np.max(np.abs(err))),
            "runtime_s": round(float(wall), 4),
        })
    return pd.DataFrame(rows).set_index("estimator").sort_values("rmse")


# ======================================================================
# SoH / capacity-fade loading
# ======================================================================
def load_capacity_fade_csv(path, capacity_col: str = "capacity_Ah",
                           cycle_col: str = "cycle") -> tuple[np.ndarray, np.ndarray]:
    """Load a per-cycle capacity series from CSV → (cycle_numbers, capacity_Ah)."""
    import pandas as pd
    df = pd.read_csv(path)
    cap = df[capacity_col].to_numpy(float)
    cyc = (df[cycle_col].to_numpy(float) if cycle_col in df.columns
           else np.arange(1, len(cap) + 1, dtype=float))
    return cyc, cap


def nasa_mat_to_capacity(mat_path) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-cycle discharge capacity from a NASA PCoE ``.mat`` (e.g. B0005).

    Best-effort parser for the standard PCoE nested struct; requires SciPy.
    Returns (cycle_numbers, capacity_Ah).  Raises a clear error if no discharge
    capacity is found (structure mismatch).
    """
    from scipy.io import loadmat

    mat = loadmat(mat_path, simplify_cells=True)
    key = next(k for k in mat if not k.startswith("__"))
    cycles = mat[key]["cycle"]
    caps, nums, n = [], [], 0
    for c in (cycles if isinstance(cycles, (list, tuple)) else [cycles]):
        if str(c.get("type", "")).lower() == "discharge":
            cap = c.get("data", {}).get("Capacity")
            if cap is not None:
                n += 1
                caps.append(float(np.ravel(cap)[0]))
                nums.append(n)
    if not caps:
        raise ValueError("no discharge-capacity cycles found — check the .mat structure")
    return np.array(nums, float), np.array(caps, float)


def soh_curve(capacity_Ah: np.ndarray, nominal_Ah: float | None = None,
              eol: float = 0.8) -> tuple[np.ndarray, float]:
    """Return (soh_series, rul_cycles) from a capacity-fade series.

    ``soh = capacity / nominal``; RUL is a linear extrapolation of the recent
    fade trend to the ``eol`` threshold (``inf`` if not yet fading).
    """
    cap = np.asarray(capacity_Ah, float)
    nominal = float(nominal_Ah if nominal_Ah is not None else cap[0])
    soh = cap / max(nominal, 1e-9)
    if len(soh) < 2:
        return soh, float("inf")
    tail = slice(max(0, len(soh) - 20), len(soh))
    x = np.arange(len(soh))[tail]
    slope, intercept = np.polyfit(x, soh[tail], 1)
    if slope >= -1e-9:
        return soh, float("inf")
    x_eol = (eol - intercept) / slope
    return soh, float(max(0.0, x_eol - (len(soh) - 1)))
