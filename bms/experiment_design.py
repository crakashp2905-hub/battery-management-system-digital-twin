"""The Autonomous Experiment Designer — *what should I do next to learn fastest?*

The observability engine (:mod:`bms.observability`) tells the twin **what it
cannot currently identify**.  This module closes the other half of the loop: it
picks the excitation that would make those parameters observable, as cheaply and
as *safely* as possible.

It scores a library of candidate experiments (rest, constant-current pulses at
several C-rates and both signs, an HPPC sequence, a rich multi-pulse train) with
the classical **optimal-experiment-design** criteria computed from the Fisher
information of each candidate:

* **D-optimal** — maximise ``log det FIM`` (most total information); or
* **targeted** — minimise the Cramér–Rao bound of one parameter the twin most
  wants to learn (e.g. capacity after a long rest has left ``Q`` uncertain).

Every candidate is first passed through a **safety filter**: it is simulated
through the cell model and rejected if it would push the terminal voltage outside
the chemistry's window or exceed a current limit.  The designer therefore only
ever proposes an excitation that is safe *from the twin's current state* — at a
low SoC the aggressive discharges are filtered out and a charge pulse wins, and
vice-versa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chemistry import get_chemistry_props
from .ecm import ECMParameters, SecondOrderECM
from .observability import DEFAULT_PARAMS, analyze_observability
from .ocv_soc import OCVSOC


@dataclass(frozen=True)
class Experiment:
    """A candidate excitation — a named current profile (``>0`` = discharge)."""

    name: str
    currents: np.ndarray
    dt: float
    description: str


@dataclass(frozen=True)
class ExperimentDesign:
    """The designer's choice plus the full ranking and what it rejected."""

    best: Experiment | None
    criterion: str                       # "d_opt" or "target:<param>"
    score: float
    target: str | None
    ranking: list[tuple[str, float]]     # (name, score), best first
    rejected_unsafe: list[str]

    def to_dict(self) -> dict:
        return {
            "best": self.best.name if self.best else None,
            "criterion": self.criterion, "score": self.score,
            "target": self.target, "ranking": self.ranking,
            "rejected_unsafe": self.rejected_unsafe,
        }


def experiment_library(capacity_Ah: float, dt: float = 1.0) -> list[Experiment]:
    """A default palette of excitations, current-scaled to the cell's capacity."""
    def const(c_rate: float, seconds: int) -> np.ndarray:
        return np.full(int(seconds / dt), c_rate * capacity_Ah, float)

    def hppc() -> np.ndarray:
        seg = []
        for _ in range(3):
            seg += [np.zeros(int(30 / dt)),
                    const(1.0, 30), np.zeros(int(40 / dt)),
                    const(-1.0, 30), np.zeros(int(40 / dt))]
        return np.concatenate(seg)

    def multipulse() -> np.ndarray:
        rng = np.random.default_rng(0)
        levels = rng.choice([-2.0, -1.0, 0.0, 1.0, 2.0], size=20)
        return np.concatenate([const(float(c), 30) for c in levels])

    return [
        Experiment("rest", np.zeros(int(300 / dt)), dt, "300 s relaxation (no current)"),
        Experiment("cc_discharge_0p5C", const(0.5, 600), dt, "0.5C discharge, 10 min"),
        Experiment("cc_discharge_1C", const(1.0, 600), dt, "1C discharge, 10 min"),
        Experiment("cc_discharge_2C", const(2.0, 300), dt, "2C discharge, 5 min"),
        Experiment("cc_charge_1C", const(-1.0, 400), dt, "1C charge, ~7 min"),
        Experiment("hppc", hppc(), dt, "HPPC: ±1C pulses with rests"),
        Experiment("multipulse", multipulse(), dt, "randomised ±2C pulse train"),
    ]


def is_safe(exp: Experiment, params: ECMParameters, soc0: float, *,
            ocv_curve: OCVSOC, chemistry: str = "nmc",
            temperature_C: float = 25.0, i_max_A: float | None = None) -> bool:
    """True if the excitation keeps voltage in-window and current within limit."""
    if i_max_A is not None and float(np.max(np.abs(exp.currents))) > i_max_A:
        return False
    props = get_chemistry_props(chemistry)
    v_min, v_max = float(props["v_min"]), float(props["v_max"])
    ecm = SecondOrderECM(params=params, ocv_curve=ocv_curve)
    temps = np.full(len(exp.currents), temperature_C)
    v = ecm.simulate(exp.currents, exp.dt, soc0=soc0, temperatures=temps)["v_terminal"]
    return bool(v.min() >= v_min - 1e-9 and v.max() <= v_max + 1e-9)


def expected_information_gain(exp: Experiment, params: ECMParameters, soc0: float, *,
                              ocv_curve: OCVSOC | None = None,
                              target: str | None = None, sigma_v: float = 0.005,
                              temperature_C: float = 25.0,
                              param_names: tuple[str, ...] = DEFAULT_PARAMS) -> float:
    """Score an experiment: ``d_opt`` (total info) or ``-CRLB[target]`` (targeted).

    Higher is always better.  A targeted score of ``-inf`` means the experiment
    would not identify the target parameter at all.
    """
    ocv_curve = ocv_curve or OCVSOC()
    temps = np.full(len(exp.currents), temperature_C)
    rep = analyze_observability(params, exp.currents, exp.dt, soc0=soc0,
                                ocv_curve=ocv_curve, temperatures=temps,
                                sigma_v=sigma_v, param_names=param_names)
    if target is not None:
        crlb = rep.crlb[target]
        return -crlb if np.isfinite(crlb) else float("-inf")
    return rep.d_opt


def design_next_experiment(params: ECMParameters, soc0: float, *,
                           capacity_Ah: float | None = None,
                           ocv_curve: OCVSOC | None = None, chemistry: str = "nmc",
                           target: str | None = None, sigma_v: float = 0.005,
                           candidates: list[Experiment] | None = None,
                           i_max_A: float | None = None,
                           temperature_C: float = 25.0) -> ExperimentDesign:
    """Choose the safest most-informative next excitation from the current state.

    Parameters
    ----------
    target : str, optional
        A parameter to prioritise (e.g. ``"Q_Ah"``); when ``None`` the designer
        maximises total information (D-optimality).
    """
    ocv_curve = ocv_curve or OCVSOC()
    capacity_Ah = float(capacity_Ah if capacity_Ah is not None else params.Q_nom_Ah)
    candidates = candidates or experiment_library(capacity_Ah, dt=1.0)
    criterion = f"target:{target}" if target else "d_opt"

    scored: list[tuple[str, float]] = []
    rejected: list[str] = []
    by_name: dict[str, Experiment] = {}
    for exp in candidates:
        by_name[exp.name] = exp
        if not is_safe(exp, params, soc0, ocv_curve=ocv_curve, chemistry=chemistry,
                       temperature_C=temperature_C, i_max_A=i_max_A):
            rejected.append(exp.name)
            continue
        score = expected_information_gain(exp, params, soc0, ocv_curve=ocv_curve,
                                          target=target, sigma_v=sigma_v,
                                          temperature_C=temperature_C)
        scored.append((exp.name, float(score)))

    scored.sort(key=lambda t: t[1], reverse=True)
    best = by_name[scored[0][0]] if scored else None
    best_score = scored[0][1] if scored else float("-inf")
    return ExperimentDesign(best=best, criterion=criterion, score=best_score,
                            target=target, ranking=scored, rejected_unsafe=rejected)


@dataclass
class ExperimentDesigner:
    """Convenience wrapper binding a cell model, chemistry and safety limits."""

    params: ECMParameters
    ocv_curve: OCVSOC = None  # type: ignore[assignment]
    chemistry: str = "nmc"
    sigma_v: float = 0.005
    i_max_A: float | None = None

    def __post_init__(self) -> None:
        self.ocv_curve = self.ocv_curve or OCVSOC()

    def design(self, soc0: float, *, target: str | None = None,
               temperature_C: float = 25.0,
               candidates: list[Experiment] | None = None) -> ExperimentDesign:
        return design_next_experiment(
            self.params, soc0, ocv_curve=self.ocv_curve, chemistry=self.chemistry,
            target=target, sigma_v=self.sigma_v, candidates=candidates,
            i_max_A=self.i_max_A, temperature_C=temperature_C)
