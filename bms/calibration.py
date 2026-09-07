"""
Calibration: fit ECM parameters to real data, learn per-cell parameter
distributions, and report validation metrics bucketed by operating condition.

This is the bridge from a research *simulator* to a data-*calibrated* digital
twin: instead of fixed manufacturing scatter, parameters (and their spread) are
identified from HPPC / pulse or drive-cycle measurements, and accuracy is
reported per chemistry / temperature / C-rate — and separately for synthetic vs
real data (:attr:`bms.DriveCycleData.source`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PulseFitResult:
    """Result of an ECM parameter identification from one cell's trace."""

    params: object          # bms.ecm.ECMParameters
    rmse_v: float
    soc0: float
    success: bool


def fit_from_pulse(current, voltage, dt: float, capacity_Ah: float = 2.3,
                   chemistry: str = "nmc", ocv_curve=None) -> PulseFitResult:
    """Identify ECM parameters (R0, R1, C1, R2, C2) from an HPPC/pulse or drive trace.

    Wraps :func:`bms.fit_ecm_parameters` with an OCV curve for the chemistry and
    returns the fit and its voltage RMSE.
    """
    from .ecm import fit_ecm_parameters
    from .ocv_soc import OCVSOC

    ocv = ocv_curve or OCVSOC.from_chemistry(chemistry)
    fitted, info = fit_ecm_parameters(
        np.asarray(current, float), np.asarray(voltage, float), dt,
        Q_nom_Ah=capacity_Ah, ocv_curve=ocv)
    return PulseFitResult(params=fitted, rmse_v=float(info["rmse_v"]),
                          soc0=float(info["soc0"]), success=bool(info["success"]))


def fit_cell_distribution(cells, dt: float = 1.0, capacity_Ah: float = 2.3,
                          chemistry: str = "nmc") -> dict:
    """Learn per-parameter distributions across cells from their measured traces.

    Parameters
    ----------
    cells : list of (current, voltage)
        One I/V trace per physical cell.

    Returns
    -------
    dict with ``n_cells``; ``distribution`` (``{param: {"mean", "std"}}`` for
    R0/R1/C1/R2/C2); and ``pack_scatter`` (data-learned relative sigmas suitable
    for :class:`bms.PackConfig`, replacing fixed manufacturing scatter).
    """
    fits = [fit_from_pulse(i, v, dt, capacity_Ah, chemistry) for (i, v) in cells]
    names = ("R0", "R1", "C1", "R2", "C2")
    vals = {n: np.array([getattr(f.params, n) for f in fits], float) for n in names}
    distribution = {n: {"mean": float(np.mean(vals[n])), "std": float(np.std(vals[n]))}
                    for n in names}

    def rel_sigma(name):
        m = float(np.mean(vals[name]))
        return float(np.std(vals[name]) / m) if m > 1e-12 and len(fits) > 1 else 0.0

    return {
        "n_cells": len(fits),
        "distribution": distribution,
        "pack_scatter": {"r0_sigma": rel_sigma("R0")},
        "mean_rmse_v": float(np.mean([f.rmse_v for f in fits])),
    }


def validation_report(data, estimator: str = "ekf", buckets_by: str = "c_rate",
                      n_buckets: int = 3):
    """Validate an estimator on a drive cycle, reporting SoC error **per bucket**.

    ``buckets_by`` is ``"c_rate"`` or ``"temperature"``; the trace is split into
    ``n_buckets`` quantile bins of that variable and SoC RMSE/MAE/max-error is
    reported for each, tagged with the data's ``source`` (synthetic vs real) and
    chemistry — so real-data accuracy is never conflated with synthetic.
    """
    import pandas as pd

    from .chemistry import get_chemistry_props
    from .ecm import ECMParameters
    from .estimation import make_soc_estimator
    from .ocv_soc import OCVSOC

    d = get_chemistry_props(data.chemistry)["default_ecm"]
    params = ECMParameters(R0=d["R0"], R1=d["R1"], C1=d["C1"], R2=d["R2"], C2=d["C2"],
                           Q_nom_Ah=data.capacity_Ah, chemistry=data.chemistry)
    est = make_soc_estimator(estimator, params=params,
                             ocv_curve=OCVSOC.from_chemistry(data.chemistry),
                             capacity_Ah=data.capacity_Ah)
    est.reset(float(data.soc_true[0]))
    soc_hat = np.asarray(est.run(data.current_A, data.voltage_V, data.dt), float)

    if buckets_by == "temperature":
        bvar, label = np.asarray(data.temperature_C, float), "temperature_C"
    else:
        bvar = np.abs(data.current_A) / max(data.capacity_Ah, 1e-9)
        label = "c_rate"

    edges = np.unique(np.quantile(bvar, np.linspace(0.0, 1.0, n_buckets + 1)))
    edges[-1] += 1e-9
    source = getattr(data, "source", "synthetic")
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (bvar >= lo) & (bvar < hi)
        if not mask.any():
            continue
        e = soc_hat[mask] - data.soc_true[mask]
        rows.append({
            f"{label}_lo": round(float(lo), 3), f"{label}_hi": round(float(hi), 3),
            "n": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))),
            "max_err": float(np.max(np.abs(e))),
            "estimator": estimator, "chemistry": data.chemistry, "source": source,
        })
    return pd.DataFrame(rows)
