"""Portable BMS **control core** — the algorithms you'd ship to real firmware.

The rest of the package is a research/simulation twin (NumPy, SciPy, pandas).
Firmware can't run that.  This module is the opposite: the deterministic core a
real BMS actually executes each control cycle — a 1-state SoC EKF, State-of-Power
limits, and the safety threshold checks — written in **plain scalar arithmetic**
(only :mod:`math`, fixed-size state, no dynamic allocation, no library calls in
the hot path) so it transliterates almost line-for-line to embedded C.

The twin is its **software-in-the-loop (SIL) test oracle**: :func:`run_sil` drives
this core with the twin's plant and checks the core tracks the true SoC and that
its safety flags fire — the exact bench you'd use before porting.

Fault-flag bitmask (``flags`` in the output):
``1`` over-voltage · ``2`` under-voltage · ``4`` over-temperature · ``8`` over-current.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Fixed OCV lookup table (SoC, OCV[V]) — NMC, sampled from the twin's curve.
# A literal table + linear interpolation is C-portable (no PCHIP/SciPy).
_OCV_SOC = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_OCV_V = (3.00, 3.52, 3.61, 3.66, 3.71, 3.76, 3.82, 3.89, 3.96, 4.06, 4.20)

FLAG_OVERVOLTAGE = 1
FLAG_UNDERVOLTAGE = 2
FLAG_OVERTEMP = 4
FLAG_OVERCURRENT = 8


@dataclass
class CoreParams:
    """Fixed calibration a firmware build would carry in flash."""

    r0: float = 0.025
    r1: float = 0.015
    tau1: float = 30.0
    r2: float = 0.030
    tau2: float = 240.0
    q_nom_as: float = 2.3 * 3600.0       # nominal charge [A·s]
    v_max: float = 4.2
    v_min: float = 3.0
    t_max_C: float = 55.0
    i_max: float = 20.0
    ekf_q_soc: float = 1e-7              # SoC process variance
    ekf_r: float = 1e-4                 # voltage measurement variance


@dataclass
class CoreState:
    """The mutable state carried between control cycles (a C struct)."""

    soc: float = 1.0
    vrc1: float = 0.0
    vrc2: float = 0.0
    p_soc: float = 1e-2                  # SoC error covariance


def core_reset(soc0: float = 1.0) -> CoreState:
    return CoreState(soc=_clip01(soc0), vrc1=0.0, vrc2=0.0, p_soc=1e-2)


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def ocv_lut(soc: float) -> float:
    """Piecewise-linear OCV from the fixed table (C-portable)."""
    s = _clip01(soc)
    for i in range(len(_OCV_SOC) - 1):
        if s <= _OCV_SOC[i + 1]:
            frac = (s - _OCV_SOC[i]) / (_OCV_SOC[i + 1] - _OCV_SOC[i])
            return _OCV_V[i] + frac * (_OCV_V[i + 1] - _OCV_V[i])
    return _OCV_V[-1]


def _docv(soc: float) -> float:
    """Local OCV slope dV/dSoC from the table."""
    s = _clip01(soc)
    for i in range(len(_OCV_SOC) - 1):
        if s <= _OCV_SOC[i + 1]:
            return (_OCV_V[i + 1] - _OCV_V[i]) / (_OCV_SOC[i + 1] - _OCV_SOC[i])
    return (_OCV_V[-1] - _OCV_V[-2]) / (_OCV_SOC[-1] - _OCV_SOC[-2])


def core_step(state: CoreState, params: CoreParams, voltage: float,
              current: float, temperature_C: float, dt: float) -> dict:
    """One control cycle (all scalar).  ``current > 0`` = discharge.

    Mutates ``state`` (SoC EKF + RC feed-forward) and returns the outputs a BMS
    publishes: ``soc``, State-of-Power limits, and the safety ``flags`` bitmask
    with a ``contactor_open`` command.
    """
    p = params
    # --- RC over-potentials (deterministic feed-forward) ---------------
    a1 = math.exp(-dt / p.tau1)
    a2 = math.exp(-dt / p.tau2)
    state.vrc1 = a1 * state.vrc1 + (1.0 - a1) * p.r1 * current
    state.vrc2 = a2 * state.vrc2 + (1.0 - a2) * p.r2 * current

    # --- 1-state SoC EKF ----------------------------------------------
    state.soc = state.soc - current * dt / p.q_nom_as        # predict
    state.p_soc = state.p_soc + p.ekf_q_soc
    v_pred = ocv_lut(state.soc) - state.vrc1 - state.vrc2 - p.r0 * current
    h = _docv(state.soc)
    s_cov = h * state.p_soc * h + p.ekf_r
    k = state.p_soc * h / s_cov
    state.soc = _clip01(state.soc + k * (voltage - v_pred))   # update
    state.p_soc = (1.0 - k * h) * state.p_soc

    # --- State of Power (ohmic, voltage-window bounded) ---------------
    ocv = ocv_lut(state.soc)
    i_dis = (ocv - p.v_min) / p.r0
    i_chg = (p.v_max - ocv) / p.r0
    i_dis = i_dis if i_dis < p.i_max else p.i_max
    i_chg = i_chg if i_chg < p.i_max else p.i_max
    p_dis = p.v_min * (i_dis if i_dis > 0.0 else 0.0)
    p_chg = p.v_max * (i_chg if i_chg > 0.0 else 0.0)

    # --- Safety threshold checks (bitmask) ----------------------------
    flags = 0
    if voltage > p.v_max:
        flags |= FLAG_OVERVOLTAGE
    if voltage < p.v_min:
        flags |= FLAG_UNDERVOLTAGE
    if temperature_C > p.t_max_C:
        flags |= FLAG_OVERTEMP
    if current > p.i_max or current < -p.i_max:
        flags |= FLAG_OVERCURRENT

    return {
        "soc": state.soc,
        "soc_sigma": math.sqrt(state.p_soc) if state.p_soc > 0 else 0.0,
        "power_discharge_W": p_dis,
        "power_charge_W": p_chg,
        "flags": flags,
        "contactor_open": flags != 0,
    }


def run_sil(params: CoreParams, currents, voltages, temperatures, dt: float,
            soc_truth=None) -> dict:
    """Software-in-the-loop: run the core over a trace and summarise.

    Feed measured (current, voltage, temperature) — e.g. from the twin plant.
    Returns the SoC estimate, the raised-flag history, and (if ``soc_truth`` is
    given) the SoC tracking error — the check you'd run before porting to C.
    """
    st = core_reset(float(soc_truth[0]) if soc_truth is not None else 1.0)
    n = len(currents)
    soc = [0.0] * n
    any_flags = 0
    for k in range(n):
        out = core_step(st, params, float(voltages[k]), float(currents[k]),
                        float(temperatures[k]) if temperatures is not None else 25.0, dt)
        soc[k] = out["soc"]
        any_flags |= out["flags"]
    result = {"soc": soc, "flags_seen": any_flags, "final_soc": soc[-1]}
    if soc_truth is not None:
        err = [soc[k] - float(soc_truth[k]) for k in range(n)]
        result["soc_rmse"] = math.sqrt(sum(e * e for e in err) / n)
        result["soc_max_err"] = max(abs(e) for e in err)
    return result
