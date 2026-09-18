"""The Counterfactual Battery Twin — *what would have happened if…?*

A monitoring twin answers "what is the battery doing?".  A counterfactual twin
answers the question an operator actually cares about: *"what if I had charged at
1C instead of 3C?"*, *"what if this pack had lived 10 °C cooler?"*.  It forks the
current state, runs an alternate operating policy forward through the same
physics, and reports how the trajectory — and, crucially, the **degradation** —
would have diverged.

Two entry points:

* :func:`counterfactual_charge` — the headline comparison.  Runs a CC-CV charge
  at several C-rates on the shared plant (:mod:`bms.charge_control`) and prices
  each into charge time, peak temperature, plating margin and, via
  :class:`bms.aging.AgingModel`, the capacity fade and resistance growth that
  policy would cost.  The fast policy wins on time and loses on health — the
  trade the twin makes explicit.
* :func:`alternate_history` — replays a mission (a real current trace) under a
  counterfactual modifier (scaled current, shifted temperature) and compares the
  degradation of the two histories.

Because it reuses the existing plant, charger and aging models, a counterfactual
is a genuine forward simulation of the same twin, not a separate heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aging import AgingModel
from .charge_control import ChargeLimits, MPCCharger, cccv_charge
from .ecm import ECMParameters, SecondOrderECM
from .ocv_soc import OCVSOC


@dataclass(frozen=True)
class PolicyOutcome:
    """What one operating policy would cost, in performance *and* health."""

    label: str
    charge_time_s: float
    reached_target: bool
    peak_temperature_C: float
    min_plating_margin_A: float
    capacity_fade_pct: float          # capacity lost by this one session
    resistance_growth_pct: float
    throughput_efc: float

    def to_dict(self) -> dict:
        return {
            "label": self.label, "charge_time_s": self.charge_time_s,
            "reached_target": self.reached_target,
            "peak_temperature_C": self.peak_temperature_C,
            "min_plating_margin_A": self.min_plating_margin_A,
            "capacity_fade_pct": self.capacity_fade_pct,
            "resistance_growth_pct": self.resistance_growth_pct,
            "throughput_efc": self.throughput_efc,
        }


def _charge_degradation(traj: dict, aging: AgingModel, q_nom_Ah: float
                        ) -> tuple[float, float, float, float]:
    """Price a charge trajectory into (cap_fade, res_growth, efc, plating_risk)."""
    cur = np.abs(np.asarray(traj["current_A"], float))
    soc = np.asarray(traj["soc"], float)
    dt = float(traj["t_s"][1] - traj["t_s"][0]) if len(traj["t_s"]) > 1 else 1.0
    ah = float(np.sum(cur) * dt / 3600.0)
    efc = ah / max(q_nom_Ah, 1e-9)
    c_rate = float(np.mean(cur)) / max(q_nom_Ah, 1e-9) if len(cur) else 0.0
    dod = float(soc.max() - soc.min()) if len(soc) else 0.0
    soc_avg = float(np.mean(soc)) if len(soc) else 0.5
    # Plating risk: how far the charge dipped below the plating current cap.
    plating_risk = max(0.0, -float(traj["min_plating_margin_A"])) / max(q_nom_Ah, 1e-9)
    peak_T = float(traj["peak_temperature_C"])
    cap, res = aging.charge_fade(c_rate, peak_T, dod, soc_avg, efc, plating_risk)
    return cap, res, efc, plating_risk


def counterfactual_charge(charger: MPCCharger, soc0: float,
                          c_rates: tuple[float, ...] = (1.0, 2.0, 3.0),
                          aging: AgingModel | None = None, dt: float = 1.0
                          ) -> dict:
    """Compare CC-CV charge policies at several C-rates: speed vs health.

    Returns ``outcomes`` (one :class:`PolicyOutcome` per C-rate) plus the
    ``fastest`` and ``gentlest`` (least-degrading) labels.
    """
    aging = aging or AgingModel()
    q_nom = charger.params.Q_nom_Ah
    outcomes: list[PolicyOutcome] = []
    for c in c_rates:
        traj = cccv_charge(charger, soc0, c_rate=c, dt=dt)
        cap, res, efc, _ = _charge_degradation(traj, aging, q_nom)
        outcomes.append(PolicyOutcome(
            label=f"{c:g}C",
            charge_time_s=traj["time_to_target_s"],
            reached_target=traj["reached_target"],
            peak_temperature_C=traj["peak_temperature_C"],
            min_plating_margin_A=traj["min_plating_margin_A"],
            capacity_fade_pct=100.0 * cap,
            resistance_growth_pct=100.0 * res,
            throughput_efc=efc,
        ))
    reached = [o for o in outcomes if o.reached_target] or outcomes
    fastest = min(reached, key=lambda o: o.charge_time_s)
    gentlest = min(outcomes, key=lambda o: o.capacity_fade_pct)
    return {
        "outcomes": outcomes,
        "fastest": fastest.label,
        "gentlest": gentlest.label,
        "table": [o.to_dict() for o in outcomes],
    }


@dataclass(frozen=True)
class HistoryComparison:
    """Baseline vs counterfactual degradation over a replayed mission."""

    baseline_capacity_fade_pct: float
    counterfactual_capacity_fade_pct: float
    baseline_resistance_growth_pct: float
    counterfactual_resistance_growth_pct: float
    label: str
    worse: str            # "counterfactual" or "baseline" — which aged more

    def to_dict(self) -> dict:
        return {
            "baseline_capacity_fade_pct": self.baseline_capacity_fade_pct,
            "counterfactual_capacity_fade_pct": self.counterfactual_capacity_fade_pct,
            "baseline_resistance_growth_pct": self.baseline_resistance_growth_pct,
            "counterfactual_resistance_growth_pct": self.counterfactual_resistance_growth_pct,
            "label": self.label, "worse": self.worse,
        }


def _mission_degradation(currents: np.ndarray, dt: float, params: ECMParameters,
                         ocv_curve: OCVSOC, soc0: float, temperatures: np.ndarray,
                         aging: AgingModel) -> tuple[float, float]:
    ecm = SecondOrderECM(params=params, ocv_curve=ocv_curve)
    sim = ecm.simulate(currents, dt, soc0=soc0, temperatures=temperatures)
    soc = sim["soc"]
    cur = np.abs(currents)
    ah = float(np.sum(cur) * dt / 3600.0)
    efc = ah / max(params.Q_nom_Ah, 1e-9)
    c_rate = float(np.mean(cur)) / max(params.Q_nom_Ah, 1e-9) if len(cur) else 0.0
    dod = float(soc.max() - soc.min())
    soc_avg = float(np.mean(soc))
    temp_avg = float(np.mean(temperatures))
    cap, res = aging.charge_fade(c_rate, temp_avg, dod, soc_avg, efc)
    return cap, res


def alternate_history(currents: np.ndarray, dt: float, *, params: ECMParameters,
                      soc0: float, ocv_curve: OCVSOC | None = None,
                      temperatures: np.ndarray | None = None,
                      aging: AgingModel | None = None,
                      current_scale: float = 1.0,
                      temperature_delta_C: float = 0.0,
                      label: str = "counterfactual") -> HistoryComparison:
    """Replay a mission, then a modified version, and compare their degradation.

    ``current_scale`` multiplies the current (e.g. 2.0 = "what if I had driven
    twice as hard?"); ``temperature_delta_C`` shifts the temperature (e.g. -10 =
    "what if it had run 10 °C cooler?").
    """
    ocv_curve = ocv_curve or OCVSOC()
    aging = aging or AgingModel()
    currents = np.asarray(currents, float)
    n = len(currents)
    T = np.full(n, 25.0) if temperatures is None else np.asarray(temperatures, float)

    base_cap, base_res = _mission_degradation(currents, dt, params, ocv_curve,
                                              soc0, T, aging)
    cf_cap, cf_res = _mission_degradation(currents * current_scale, dt, params,
                                          ocv_curve, soc0, T + temperature_delta_C, aging)
    worse = "counterfactual" if cf_cap > base_cap else "baseline"
    return HistoryComparison(
        baseline_capacity_fade_pct=100.0 * base_cap,
        counterfactual_capacity_fade_pct=100.0 * cf_cap,
        baseline_resistance_growth_pct=100.0 * base_res,
        counterfactual_resistance_growth_pct=100.0 * cf_res,
        label=label, worse=worse,
    )


@dataclass
class CounterfactualTwin:
    """Convenience wrapper binding a cell model, charger limits and aging model."""

    params: ECMParameters
    ocv_curve: OCVSOC = None  # type: ignore[assignment]
    limits: ChargeLimits = field(default_factory=ChargeLimits)
    aging: AgingModel = field(default_factory=AgingModel)
    ambient_C: float = 25.0

    def __post_init__(self) -> None:
        self.ocv_curve = self.ocv_curve or OCVSOC()

    def _charger(self) -> MPCCharger:
        return MPCCharger(params=self.params, ocv_curve=self.ocv_curve,
                          limits=self.limits, ambient_C=self.ambient_C)

    def compare_charge(self, soc0: float,
                       c_rates: tuple[float, ...] = (1.0, 2.0, 3.0)) -> dict:
        return counterfactual_charge(self._charger(), soc0, c_rates=c_rates,
                                     aging=self.aging)

    def what_if(self, currents: np.ndarray, dt: float, soc0: float, **kwargs
                ) -> HistoryComparison:
        return alternate_history(currents, dt, params=self.params, soc0=soc0,
                                 ocv_curve=self.ocv_curve, aging=self.aging, **kwargs)
