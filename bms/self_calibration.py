"""The Self-Calibrating Twin — detect mismatch, find the wrong parameters, refit.

A twin drifts when the physical cell changes in a way the model does not capture
(resistance growth, capacity fade, an OCV shift).  :class:`TwinSync` *detects*
that drift; this module *acts* on it.

The novelty is the **observability gate**.  Naively re-fitting every parameter to
a short, low-excitation trace over-fits: it will happily move ``Q`` to absorb
noise even when the current never moved SoC enough to reveal capacity.  So the
calibrator first asks the observability engine *which parameters this trace can
actually identify*, and recalibrates **only those** — leaving the unobservable
ones at their prior.  You cannot fix what you cannot see, and pretending
otherwise is how a twin quietly poisons itself.

The recalibration is a bounded least-squares refit of the identifiable subset,
accepted only if it genuinely lowers the voltage residual.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares

from .ecm import ECMParameters, SecondOrderECM
from .observability import DEFAULT_PARAMS, analyze_observability
from .ocv_soc import OCVSOC


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of one self-calibration."""

    params: ECMParameters
    soc0: float
    calibrated: list[str]
    skipped_unidentifiable: list[str]
    rmse_before_V: float
    rmse_after_V: float
    improved: bool
    deltas: dict[str, float]        # fractional change of each calibrated parameter

    def to_dict(self) -> dict:
        return {
            "calibrated": self.calibrated,
            "skipped_unidentifiable": self.skipped_unidentifiable,
            "rmse_before_V": self.rmse_before_V, "rmse_after_V": self.rmse_after_V,
            "improved": self.improved, "deltas": self.deltas,
        }


def _bounds_for(name: str, params: ECMParameters) -> tuple[float, float]:
    if name == "soc0":
        return (0.0, 1.0)
    if name == "Q_Ah":
        return (0.3 * params.Q_nom_Ah, 1.5 * params.Q_nom_Ah)
    return (1e-4, 0.3)              # R0 / R1 / R2 [Ω]


def _value_of(name: str, params: ECMParameters, soc0: float) -> float:
    if name == "soc0":
        return float(soc0)
    if name == "Q_Ah":
        return float(params.Q_nom_Ah)
    return float(getattr(params, name))


def _apply(names: list[str], x: np.ndarray, params: ECMParameters, soc0: float
           ) -> tuple[ECMParameters, float]:
    p, s = params, soc0
    for name, val in zip(names, x):
        if name == "soc0":
            s = float(val)
        elif name == "Q_Ah":
            p = replace(p, Q_nom_Ah=float(val))
        else:
            p = replace(p, **{name: float(val)})
    return p, s


@dataclass
class SelfCalibratingTwin:
    """Autonomously recalibrate a cell model to a measured trace, gated by
    observability.

    Parameters
    ----------
    params, ocv_curve : the current twin cell model.
    sigma_v : sensor-noise std used by the observability analysis.
    rmse_tol_V : residual RMS above which the twin considers itself mismatched.
    crlb_threshold : identifiability threshold (fractional CRLB) below which a
        parameter is deemed observable enough to recalibrate.
    """

    params: ECMParameters
    ocv_curve: OCVSOC = None  # type: ignore[assignment]
    sigma_v: float = 0.005
    rmse_tol_V: float = 0.01
    crlb_threshold: float = 0.1
    param_names: tuple[str, ...] = DEFAULT_PARAMS

    def __post_init__(self) -> None:
        self.ocv_curve = self.ocv_curve or OCVSOC()

    # ------------------------------------------------------------------
    def _rmse(self, params: ECMParameters, currents, voltages, dt, soc0,
              temperatures) -> float:
        ecm = SecondOrderECM(params=params, ocv_curve=self.ocv_curve)
        v = ecm.simulate(np.asarray(currents, float), dt, soc0=soc0,
                         temperatures=temperatures)["v_terminal"]
        return float(np.sqrt(np.mean((v - np.asarray(voltages, float)) ** 2)))

    def residual_rmse(self, currents, voltages, dt, soc0,
                      temperatures=None) -> float:
        """Voltage RMSE of the current model against a measured trace."""
        return self._rmse(self.params, currents, voltages, dt, soc0, temperatures)

    def needs_calibration(self, currents, voltages, dt, soc0,
                          temperatures=None) -> bool:
        return self.residual_rmse(currents, voltages, dt, soc0, temperatures) > self.rmse_tol_V

    # ------------------------------------------------------------------
    def calibrate(self, currents, voltages, dt, soc0,
                  temperatures=None, *, commit: bool = True) -> CalibrationResult:
        """Refit the identifiable parameters to the measured trace.

        Only parameters the observability engine deems identifiable from this
        trace are refit; the rest are reported in ``skipped_unidentifiable`` and
        left at their prior.  When ``commit`` is True and the fit improves the
        residual, ``self.params`` is updated in place.
        """
        currents = np.asarray(currents, float)
        voltages = np.asarray(voltages, float)
        rmse_before = self._rmse(self.params, currents, voltages, dt, soc0, temperatures)

        rep = analyze_observability(self.params, currents, dt, soc0=soc0,
                                    ocv_curve=self.ocv_curve, temperatures=temperatures,
                                    sigma_v=self.sigma_v, crlb_threshold=self.crlb_threshold,
                                    param_names=self.param_names)
        names = [n for n in self.param_names if rep.identifiable[n]]
        skipped = [n for n in self.param_names if not rep.identifiable[n]]

        if not names:                          # nothing identifiable — do not touch
            return CalibrationResult(self.params, float(soc0), [], skipped,
                                     rmse_before, rmse_before, False, {})

        x0 = np.array([_value_of(n, self.params, soc0) for n in names], float)
        lo = np.array([_bounds_for(n, self.params)[0] for n in names], float)
        hi = np.array([_bounds_for(n, self.params)[1] for n in names], float)

        def resid(x: np.ndarray) -> np.ndarray:
            p, s = _apply(names, x, self.params, soc0)
            ecm = SecondOrderECM(params=p, ocv_curve=self.ocv_curve)
            v = ecm.simulate(currents, dt, soc0=s, temperatures=temperatures)["v_terminal"]
            return v - voltages

        res = least_squares(resid, x0, bounds=(lo, hi), method="trf",
                            x_scale=np.maximum(np.abs(x0), 1e-6), max_nfev=200)
        new_params, new_soc0 = _apply(names, res.x, self.params, soc0)
        rmse_after = self._rmse(new_params, currents, voltages, dt, new_soc0, temperatures)

        improved = rmse_after < rmse_before - 1e-9
        deltas = {n: float((res.x[i] - x0[i]) / (abs(x0[i]) + 1e-12))
                  for i, n in enumerate(names)}

        if improved and commit:
            self.params = new_params

        out_params = new_params if improved else self.params
        out_soc0 = new_soc0 if improved else float(soc0)
        return CalibrationResult(out_params, out_soc0, names, skipped,
                                 rmse_before, rmse_after, improved, deltas)
