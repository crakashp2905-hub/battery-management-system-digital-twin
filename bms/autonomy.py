"""Autonomous Battery Intelligence — the closed self-learning loop.

This is the capstone that ties the pieces together into the system the strategy
review pointed at: *a twin that knows what it doesn't know and autonomously
decides how to learn.*  Each round it

1. asks the **observability engine** which parameter it is currently least sure of;
2. asks the **experiment designer** for the safest excitation that best reduces
   that uncertainty (:mod:`bms.experiment_design`);
3. **performs** the experiment against the real cell (here a hidden ground-truth
   plant — a battery, a higher-fidelity model, or the SIL oracle);
4. **self-calibrates** its model to the returned data, gated by observability so
   it only moves parameters the experiment could actually identify
   (:mod:`bms.self_calibration`);
5. measures whether its **predictive fidelity improved**, and repeats.

The research question it operationalises:

    *Can a battery digital twin autonomously identify its own epistemic
    uncertainty, choose the experiments that reduce it most efficiently, update
    from them, and improve its predictive fidelity over time?*

In the reference bench a twin that starts with fresh-cell parameters, fed a
hidden aged cell, drives its parameter error down and its held-out voltage RMSE
from tens of millivolts to the sensor-noise floor within a few rounds — without
being told what was wrong.

.. note::
   "Autonomous" here means the *loop* is closed in software; every excitation is
   still passed through the safety filter, and (as the review stressed) any
   novelty claim should be checked against the prior-art / patent / literature
   record before it is made.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .ecm import ECMParameters, SecondOrderECM
from .experiment_design import design_next_experiment
from .observability import analyze_observability
from .ocv_soc import OCVSOC
from .self_calibration import SelfCalibratingTwin

_FIT_PARAMS = ("R0", "R1", "R2", "Q_Ah")


@dataclass(frozen=True)
class LearningRound:
    """What one iteration of the autonomous loop did and achieved."""

    round: int
    target: str | None            # the parameter it went after this round
    experiment: str               # the excitation it chose
    calibrated: list[str]
    param_error_before: float     # RMS relative error vs the true cell
    param_error_after: float
    prediction_rmse_V: float      # held-out voltage RMSE after this round

    def to_dict(self) -> dict:
        return {
            "round": self.round, "target": self.target,
            "experiment": self.experiment, "calibrated": self.calibrated,
            "param_error_before": self.param_error_before,
            "param_error_after": self.param_error_after,
            "prediction_rmse_V": self.prediction_rmse_V,
        }


def _param_error(p: ECMParameters, truth: ECMParameters) -> float:
    a = np.array([p.R0, p.R1, p.R2, p.Q_nom_Ah], float)
    b = np.array([truth.R0, truth.R1, truth.R2, truth.Q_nom_Ah], float)
    return float(np.sqrt(np.mean(((a - b) / b) ** 2)))


@dataclass
class AutonomousBatteryTwin:
    """A twin that closes the observe → design → excite → calibrate loop.

    Parameters
    ----------
    params : ECMParameters
        The twin's *current* (initially wrong) belief about the cell.
    chemistry : str
        Used by the experiment designer's safety filter.
    sigma_v : float
        Sensor-noise std for the measured voltage.
    i_max_A : float, optional
        Charger/hardware current limit for the excitation.
    seed : int
        Seeds the per-round starting SoC and measurement noise.
    """

    params: ECMParameters
    ocv_curve: OCVSOC = None  # type: ignore[assignment]
    chemistry: str = "nmc"
    sigma_v: float = 0.003
    i_max_A: float | None = None
    seed: int = 0

    history: list[LearningRound] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.ocv_curve = self.ocv_curve or OCVSOC()
        self._rng = np.random.default_rng(self.seed)
        # Best (smallest) CRLB any experiment has yet achieved for each parameter.
        # The one that is still worst-pinned is what the twin most needs to learn.
        self._best_crlb: dict[str, float] = {p: float("inf") for p in _FIT_PARAMS}
        self._saturated: set[str] = set()     # params targeting can no longer improve
        self._best_val: float = float("inf")  # best held-out RMSE accepted so far
        self._bol = replace(self.params)      # beginning-of-life reference snapshot

    # ------------------------------------------------------------------
    def _run_on_plant(self, plant: ECMParameters, currents: np.ndarray, dt: float,
                      soc0: float) -> np.ndarray:
        """The real cell's response to an excitation (with sensor noise)."""
        ecm = SecondOrderECM(params=plant, ocv_curve=self.ocv_curve)
        v = ecm.simulate(currents, dt, soc0=soc0)["v_terminal"]
        return v + self._rng.normal(0.0, self.sigma_v, len(v))

    def _target(self) -> str | None:
        """The worst-pinned parameter the twin has not yet given up on.

        Returns ``None`` on the first round (nothing learned → maximise total
        information) and once every remaining parameter is pinned below threshold.
        A parameter is dropped from consideration (``_saturated``) once targeting
        it stops improving its CRLB — so the loop does not chase an intrinsically
        hard-to-identify parameter (e.g. the slow ``R2`` branch) forever.
        """
        avail = {p: c for p, c in self._best_crlb.items()
                 if p not in self._saturated and np.isfinite(c)}
        if not avail:
            return None
        worst = max(avail, key=lambda k: avail[k])
        return worst if avail[worst] > 0.05 else None

    def step_once(self, true_plant: ECMParameters, *, soc0: float | None = None,
                  dt: float = 1.0,
                  validation: tuple[np.ndarray, float] | None = None) -> LearningRound:
        """One loop iteration against a hidden ground-truth ``true_plant``.

        A calibration is committed only if it does not worsen the held-out
        validation error (cross-validation gating), so the loop's fidelity is
        monotone non-increasing and cannot diverge on an unlucky short probe.
        """
        soc0 = float(self._rng.uniform(0.4, 0.8)) if soc0 is None else soc0
        err_before = _param_error(self.params, true_plant)
        # Baseline the acceptance gate on the cold model, so the first round can
        # only commit a calibration that already beats the starting fit.
        if validation is not None and not np.isfinite(self._best_val):
            self._best_val = self._validation_rmse(true_plant, validation, dt)

        # 1-2. observe (implicitly) → design the most informative safe excitation.
        target = self._target()
        design = design_next_experiment(
            self.params, soc0, ocv_curve=self.ocv_curve, chemistry=self.chemistry,
            target=target, sigma_v=self.sigma_v, i_max_A=self.i_max_A)
        exp = design.best
        if exp is None:                       # nothing safe from here — skip
            rmse = self._validation_rmse(true_plant, validation, dt)
            rnd = LearningRound(len(self.history), target, "none", [],
                                err_before, err_before, rmse)
            self.history.append(rnd)
            return rnd

        # 3. perform the experiment on the real cell.
        v_meas = self._run_on_plant(true_plant, exp.currents, exp.dt, soc0)

        # 4. self-calibrate to the returned data (observability-gated), on a copy.
        prev_params = replace(self.params)
        cal = SelfCalibratingTwin(params=replace(self.params), ocv_curve=self.ocv_curve,
                                  sigma_v=self.sigma_v)
        res = cal.calibrate(exp.currents, v_meas, exp.dt, soc0)
        candidate = cal.params

        # 5. accept only if it improves held-out fidelity (else keep prior).
        val_candidate = self._validation_rmse(true_plant, validation, dt,
                                              twin_params=candidate)
        calibrated = res.calibrated
        if validation is not None and not (val_candidate <= self._best_val + 1e-9):
            self.params = prev_params                     # reject — would regress
            calibrated = []
        else:
            self.params = candidate
            if validation is not None:
                self._best_val = val_candidate

        # update the observability belief and saturation bookkeeping.
        before_crlb = dict(self._best_crlb)
        rep = analyze_observability(self.params, exp.currents, exp.dt, soc0=soc0,
                                    ocv_curve=self.ocv_curve, sigma_v=self.sigma_v)
        for p in _FIT_PARAMS:
            self._best_crlb[p] = min(self._best_crlb[p], rep.crlb[p])
        if target is not None and self._best_crlb[target] >= before_crlb[target] - 1e-4:
            self._saturated.add(target)                   # targeting it stopped helping

        err_after = _param_error(self.params, true_plant)
        rmse = self._validation_rmse(true_plant, validation, dt)
        rnd = LearningRound(len(self.history), target, exp.name, calibrated,
                            err_before, err_after, rmse)
        self.history.append(rnd)
        return rnd

    def _validation_rmse(self, true_plant: ECMParameters,
                         validation: tuple[np.ndarray, float] | None,
                         dt: float, twin_params: ECMParameters | None = None) -> float:
        if validation is None:
            return float("nan")
        currents, vsoc0 = validation
        currents = np.asarray(currents, float)
        twin_params = twin_params or self.params
        v_true = SecondOrderECM(params=true_plant, ocv_curve=self.ocv_curve
                                ).simulate(currents, dt, soc0=vsoc0)["v_terminal"]
        v_twin = SecondOrderECM(params=twin_params, ocv_curve=self.ocv_curve
                                ).simulate(currents, dt, soc0=vsoc0)["v_terminal"]
        return float(np.sqrt(np.mean((v_true - v_twin) ** 2)))

    def learn(self, true_plant: ECMParameters, *, n_rounds: int = 5, dt: float = 1.0,
              validation: tuple[np.ndarray, float] | None = None
              ) -> list[LearningRound]:
        """Run the closed loop for ``n_rounds`` and return the per-round record."""
        for _ in range(n_rounds):
            self.step_once(true_plant, dt=dt, validation=validation)
        return self.history

    # ------------------------------------------------------------------
    def degradation_report(self, bol: ECMParameters | None = None) -> dict:
        """Attribute what the twin *learned* to degradation modes.

        Compares the current (learned) parameters against a beginning-of-life
        reference and splits the change into resistance growth vs capacity loss —
        the two headline SoH modes.  (A finer LLI/LAM split is available from
        :func:`bms.diagnose_degradation_modes` given incremental-capacity data.)
        """
        bol = bol or self._bol
        res_growth = self.params.R0 / max(bol.R0, 1e-12) - 1.0
        cap_loss = 1.0 - self.params.Q_nom_Ah / max(bol.Q_nom_Ah, 1e-12)
        dominant = "resistance_growth" if res_growth > cap_loss else "capacity_loss"
        return {
            "resistance_growth_pct": 100.0 * res_growth,
            "capacity_loss_pct": 100.0 * cap_loss,
            "dominant_mode": dominant,
            "soh_capacity": self.params.Q_nom_Ah / max(bol.Q_nom_Ah, 1e-12),
        }
