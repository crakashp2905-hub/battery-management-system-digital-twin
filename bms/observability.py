"""The Battery Observability Engine — *what can this data actually tell me?*

A twin can only estimate a state or parameter if the measurements carry
information about it.  Capacity ``Q`` is invisible while the cell rests (SoC
never moves, so voltage never reveals ``Q``); ``R0`` is invisible at zero
current (there is no ohmic drop to measure).  A twin that keeps reporting a crisp
SoH through a long rest is fooling itself.

This module quantifies that identifiability rigorously.  For a parameter set
``θ = [soc0, R0, R1, R2, Q]`` it computes the **sensitivity** of the terminal
voltage to each parameter along a given current/temperature trajectory, forms
the **Fisher Information Matrix**

    FIM = Sᵀ S / σ²        (S = normalised voltage sensitivities, σ = sensor noise)

and reports, per parameter, the **Cramér–Rao lower bound** — the best 1-σ
uncertainty *any* unbiased estimator could achieve from this data.  A parameter
whose CRLB is large (or infinite, when its sensitivity is zero) simply cannot be
identified from the trajectory, no matter how clever the filter.

The scalar summaries (``d_opt = log det FIM``, ``a_opt = tr FIM⁻¹``,
``e_opt = λ_min``) are the classical optimal-experiment-design criteria: they let
the :mod:`bms.experiment_design` designer *choose* the excitation that will make
the currently-unobservable parameters observable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .ecm import ECMParameters, SecondOrderECM
from .ocv_soc import OCVSOC

# The canonical identifiable set: initial SoC, the three resistances, capacity.
DEFAULT_PARAMS: tuple[str, ...] = ("soc0", "R0", "R1", "R2", "Q_Ah")


@dataclass(frozen=True)
class ObservabilityReport:
    """Identifiability of each parameter from one trajectory (voltage sensor).

    ``crlb`` values are **normalised** 1-σ lower bounds: a fraction of the
    parameter's own magnitude (for ``soc0``, an absolute SoC fraction).  So
    ``crlb['Q_Ah'] = 0.03`` means capacity cannot be pinned tighter than ±3 %
    from this data; ``inf`` means it is not identifiable at all.
    """

    params: tuple[str, ...]
    fisher: np.ndarray
    crlb: dict[str, float]
    identifiable: dict[str, bool]
    condition_number: float
    d_opt: float          # log det FIM — total information (bigger is better)
    a_opt: float          # trace FIM⁻¹ — average variance (smaller is better)
    e_opt: float          # smallest eigenvalue — worst direction (bigger is better)

    def most_uncertain(self) -> str:
        """The parameter this trajectory identifies *worst* (largest CRLB)."""
        return max(self.crlb, key=lambda k: self.crlb[k])

    def to_dict(self) -> dict:
        return {
            "params": list(self.params),
            "crlb": self.crlb,
            "identifiable": self.identifiable,
            "condition_number": self.condition_number,
            "d_opt": self.d_opt, "a_opt": self.a_opt, "e_opt": self.e_opt,
            "most_uncertain": self.most_uncertain(),
        }


def _simulate_voltage(params: ECMParameters, currents: np.ndarray, dt: float,
                      soc0: float, ocv_curve: OCVSOC,
                      temperatures: np.ndarray | None) -> np.ndarray:
    ecm = SecondOrderECM(params=params, ocv_curve=ocv_curve)
    return ecm.simulate(currents, dt, soc0=soc0, temperatures=temperatures)["v_terminal"]


def _perturb(params: ECMParameters, soc0: float, name: str, delta: float
             ) -> tuple[ECMParameters, float]:
    """Return (params, soc0) with parameter ``name`` scaled by its reference step.

    Each parameter is perturbed by ``delta * ref`` where ``ref`` is its own
    magnitude (``1.0`` for ``soc0``), so the resulting sensitivities are
    normalised and comparable across parameters of very different scale.
    """
    if name == "soc0":
        return params, float(np.clip(soc0 + delta, 0.0, 1.0))     # ref = 1.0
    if name == "Q_Ah":
        return replace(params, Q_nom_Ah=params.Q_nom_Ah * (1.0 + delta)), soc0
    if name in ("R0", "R1", "R2"):
        return replace(params, **{name: getattr(params, name) * (1.0 + delta)}), soc0
    raise ValueError(f"unknown parameter {name!r}")


def voltage_sensitivities(params: ECMParameters, currents: np.ndarray, dt: float,
                          *, soc0: float, ocv_curve: OCVSOC | None = None,
                          temperatures: np.ndarray | None = None,
                          param_names: tuple[str, ...] = DEFAULT_PARAMS,
                          rel_step: float = 1e-3) -> tuple[np.ndarray, tuple[str, ...]]:
    """Normalised voltage sensitivities ``S[:, j] = ref_j · ∂V/∂θ_j``.

    Central finite differences about the operating point; returns an
    ``(n_time, n_param)`` array and the parameter names.
    """
    ocv_curve = ocv_curve or OCVSOC()
    currents = np.asarray(currents, float)
    cols = []
    for name in param_names:
        p_up, s_up = _perturb(params, soc0, name, +rel_step)
        p_dn, s_dn = _perturb(params, soc0, name, -rel_step)
        v_up = _simulate_voltage(p_up, currents, dt, s_up, ocv_curve, temperatures)
        v_dn = _simulate_voltage(p_dn, currents, dt, s_dn, ocv_curve, temperatures)
        cols.append((v_up - v_dn) / (2.0 * rel_step))
    return np.column_stack(cols), tuple(param_names)


def fisher_information(params: ECMParameters, currents: np.ndarray, dt: float,
                       *, soc0: float, ocv_curve: OCVSOC | None = None,
                       temperatures: np.ndarray | None = None,
                       sigma_v: float = 0.005,
                       param_names: tuple[str, ...] = DEFAULT_PARAMS,
                       rel_step: float = 1e-3) -> np.ndarray:
    """Fisher Information Matrix ``FIM = Sᵀ S / σ²`` for the parameters."""
    S, _ = voltage_sensitivities(params, currents, dt, soc0=soc0,
                                 ocv_curve=ocv_curve, temperatures=temperatures,
                                 param_names=param_names, rel_step=rel_step)
    return (S.T @ S) / float(sigma_v) ** 2


def analyze_observability(params: ECMParameters, currents: np.ndarray, dt: float,
                          *, soc0: float, ocv_curve: OCVSOC | None = None,
                          temperatures: np.ndarray | None = None,
                          sigma_v: float = 0.005,
                          crlb_threshold: float = 0.1,
                          param_names: tuple[str, ...] = DEFAULT_PARAMS,
                          rel_step: float = 1e-3) -> ObservabilityReport:
    """Full identifiability analysis of a trajectory.

    A parameter is ``identifiable`` when it has non-negligible sensitivity and
    its Cramér–Rao lower bound is below ``crlb_threshold`` (a fractional 1-σ,
    default 10 %).
    """
    fim = fisher_information(params, currents, dt, soc0=soc0, ocv_curve=ocv_curve,
                             temperatures=temperatures, sigma_v=sigma_v,
                             param_names=param_names, rel_step=rel_step)
    diag = np.diag(fim)
    ref = max(float(np.max(diag)), 1e-300)
    # A parameter with essentially no voltage sensitivity is unobservable; the
    # rest form the observable block whose marginal CRLBs we can compute.
    observable = diag > 1e-9 * ref

    crlb: dict[str, float] = {name: float("inf") for name in param_names}
    if observable.any():
        idx = np.where(observable)[0]
        sub = fim[np.ix_(idx, idx)]
        # Tiny ridge keeps a collinear (rank-deficient) block invertible; a truly
        # unobservable direction then surfaces as a very large CRLB, not a crash.
        ridge = 1e-12 * float(np.trace(sub)) / max(len(idx), 1)
        inv = np.linalg.inv(sub + ridge * np.eye(len(idx)))
        for k, j in enumerate(idx):
            crlb[param_names[j]] = float(np.sqrt(max(inv[k, k], 0.0)))

    identifiable = {name: bool(observable[j] and crlb[name] < crlb_threshold)
                    for j, name in enumerate(param_names)}

    eig = np.linalg.eigvalsh((fim + fim.T) / 2.0)
    eig_min = float(np.min(eig))
    eig_max = float(np.max(eig))
    cond = float(eig_max / eig_min) if eig_min > 0 else float("inf")

    # D-optimality is log det over the *observable* block (the full FIM may be
    # singular when a parameter has no sensitivity at all).
    if observable.any():
        idx = np.where(observable)[0]
        sign, logabs = np.linalg.slogdet(fim[np.ix_(idx, idx)])
        d_opt = float(logabs) if sign > 0 else float("-inf")
    else:
        d_opt = float("-inf")
    # A-optimality: total achievable variance across the identifiable parameters.
    finite = [crlb[name] ** 2 for name in param_names if np.isfinite(crlb[name])]
    a_opt = float(sum(finite)) if finite else float("inf")

    return ObservabilityReport(
        params=tuple(param_names), fisher=fim, crlb=crlb,
        identifiable=identifiable, condition_number=cond,
        d_opt=d_opt, a_opt=a_opt, e_opt=eig_min,
    )


@dataclass
class ObservabilityEngine:
    """Convenience wrapper binding a cell model, sensor noise and threshold."""

    params: ECMParameters
    ocv_curve: OCVSOC = None  # type: ignore[assignment]
    sigma_v: float = 0.005
    crlb_threshold: float = 0.1
    param_names: tuple[str, ...] = DEFAULT_PARAMS

    def __post_init__(self) -> None:
        self.ocv_curve = self.ocv_curve or OCVSOC()

    def analyze(self, currents: np.ndarray, dt: float, soc0: float,
                temperatures: np.ndarray | None = None) -> ObservabilityReport:
        return analyze_observability(
            self.params, currents, dt, soc0=soc0, ocv_curve=self.ocv_curve,
            temperatures=temperatures, sigma_v=self.sigma_v,
            crlb_threshold=self.crlb_threshold, param_names=self.param_names)

    def information(self, currents: np.ndarray, dt: float, soc0: float,
                    temperatures: np.ndarray | None = None) -> float:
        """Scalar D-optimality (``log det FIM``) — total information in a trajectory."""
        return self.analyze(currents, dt, soc0, temperatures).d_opt
