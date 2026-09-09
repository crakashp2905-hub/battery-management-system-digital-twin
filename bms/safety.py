"""State-of-Safety (SoS) — one composite 0–1 safety score.

Individual thresholds (over-temperature, over-voltage, pressure, gas, low SoH,
imbalance) each trip in isolation.  The **State of Safety** fuses them into a
single number in ``[0, 1]`` (1 = perfectly safe, 0 = critical) that degrades
*before* any one threshold trips, so the supervisor, the dashboard, and the
:class:`~bms.agent.ActionGate` can reason about one continuous margin instead of
a bag of booleans.

Each signal maps its current value to a **penalty** in ``[0, 1]`` via a linear
ramp between a *warn* level (penalty 0) and a *critical* level (penalty 1).  The
overall SoS is ``1 − max(penalty)`` — the worst signal governs, which is the
conservative choice for safety — and the per-signal breakdown says why.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyConfig:
    """Warn/critical levels for each State-of-Safety signal."""

    # Temperature [°C]
    t_warn_C: float = 45.0
    t_crit_C: float = 70.0
    # Cell temperature spread [°C]
    dt_warn_C: float = 8.0
    dt_crit_C: float = 20.0
    # Cell-voltage excursion beyond the [v_min, v_max] window [V]
    v_dev_warn_V: float = 0.02
    v_dev_crit_V: float = 0.15
    # Internal pressure [kPa]
    pressure_warn_kPa: float = 220.0
    pressure_crit_kPa: float = 350.0
    # SoH (capacity retention) — lower is worse
    soh_warn: float = 0.80
    soh_crit: float = 0.60
    # SoC imbalance across cells
    imbalance_warn: float = 0.05
    imbalance_crit: float = 0.20


def _ramp(value: float, warn: float, crit: float) -> float:
    """Linear penalty in [0, 1]: 0 at/below ``warn``, 1 at/above ``crit``."""
    if crit == warn:
        return 1.0 if value >= crit else 0.0
    return float(np.clip((value - warn) / (crit - warn), 0.0, 1.0))


def state_of_safety(result: dict, *, mechanical_state=None, soh: float | None = None,
                    v_min: float | None = None, v_max: float | None = None,
                    chemistry: str | None = None,
                    config: SafetyConfig | None = None) -> dict:
    """Compute the State of Safety from a supervisor step result.

    Parameters
    ----------
    result : dict
        A :meth:`bms.BMSSupervisor.step` result (uses ``T_cells``, ``v_cells``,
        ``imbalance``).
    mechanical_state : CellMechanicalState, optional
        If given, its ``pressure_kPa`` / ``vent_event`` contribute a gas penalty.
    soh : float, optional
        Capacity-retention SoH ∈ [0, 1].
    v_min, v_max : float, optional
        Cell voltage window; taken from ``chemistry`` props when omitted.
    chemistry : str, optional
        Used to look up the voltage window if ``v_min``/``v_max`` are not given.
    config : SafetyConfig, optional

    Returns
    -------
    dict
        ``sos`` (0–1), ``worst_signal``, ``severity`` (ok/info/warning/critical),
        and ``breakdown`` (per-signal penalty in [0, 1]).
    """
    cfg = config or SafetyConfig()
    if (v_min is None or v_max is None) and chemistry is not None:
        from .chemistry import get_chemistry_props
        props = get_chemistry_props(chemistry)
        v_min = props["v_min"] if v_min is None else v_min
        v_max = props["v_max"] if v_max is None else v_max

    penalties: dict[str, float] = {}

    if "T_cells" in result:
        T = np.asarray(result["T_cells"], float)
        penalties["temperature"] = _ramp(float(T.max()), cfg.t_warn_C, cfg.t_crit_C)
        penalties["temp_spread"] = _ramp(float(T.max() - T.min()),
                                         cfg.dt_warn_C, cfg.dt_crit_C)

    if "v_cells" in result and v_min is not None and v_max is not None:
        v = np.asarray(result["v_cells"], float)
        dev = max(0.0, float(v_min) - float(v.min()),
                  float(v.max()) - float(v_max))
        penalties["voltage"] = _ramp(dev, cfg.v_dev_warn_V, cfg.v_dev_crit_V)

    if mechanical_state is not None:
        p = float(getattr(mechanical_state, "pressure_kPa", 0.0))
        pen = _ramp(p, cfg.pressure_warn_kPa, cfg.pressure_crit_kPa)
        if getattr(mechanical_state, "vent_event", False):
            pen = 1.0
        penalties["gas_pressure"] = pen

    if soh is not None:
        # SoH is inverted (lower = worse), so ramp on the shortfall.
        penalties["soh"] = _ramp(-float(soh), -cfg.soh_warn, -cfg.soh_crit)

    if "imbalance" in result:
        penalties["imbalance"] = _ramp(float(result["imbalance"]),
                                       cfg.imbalance_warn, cfg.imbalance_crit)

    if not penalties:
        return {"sos": 1.0, "worst_signal": "none", "severity": "ok", "breakdown": {}}

    worst = max(penalties, key=penalties.get)
    worst_pen = penalties[worst]
    sos = float(1.0 - worst_pen)
    severity = ("critical" if worst_pen >= 1.0 else
                "warning" if worst_pen >= 0.5 else
                "info" if worst_pen > 0.0 else "ok")
    return {"sos": sos, "worst_signal": worst if worst_pen > 0.0 else "none",
            "severity": severity, "breakdown": penalties}
