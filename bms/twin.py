"""The central digital twin — one authoritative, uncertainty-aware state.

The repository grew a strong collection of estimators (SoC EKF/UKF/PF, the joint
SoC+capacity EKF for SoH), a drift/assimilation monitor (:class:`TwinSync`), and
an online resistance tracker (:class:`RLSIdentifier`).  Each is excellent in
isolation, but a *digital twin* is supposed to be **one continuously calibrated
belief** about the cell, not a bag of independent algorithms.

:class:`BatteryDigitalTwin` is that unifying object.  It ingests live
``(voltage, current, temperature)`` and drives the existing engines in lock-step:

* the **joint EKF** is the probabilistic core — SoC, capacity and SoH each with a
  1-σ uncertainty;
* **TwinSync** runs the ECM plant alongside it and watches the model-vs-measurement
  residual, so the twin knows when it has *drifted* out of sync with reality;
* **RLS** tracks the ohmic resistance ``R0`` online.

On top of the estimates it derives two things a raw estimator cannot give:

* an **observability / excitation** measure — capacity (hence SoH) is only
  identifiable when SoC actually moves, so the twin reports whether it currently
  has enough excitation to trust its SoH; and
* a scalar **confidence** in ``[0, 1]`` fusing data freshness, estimator
  uncertainty, residual stability and excitation — the difference between
  "SoH 82 %, confidence 0.94" and "SoH 82 %, confidence 0.41 — insufficient
  excitation for a reliable capacity estimate".

Every ``update`` returns an immutable :class:`TwinState` snapshot (with 95 %
credible intervals and a JSON-ready ``to_dict``); the twin also keeps the latest
as :attr:`BatteryDigitalTwin.state`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from .ecm import ECMParameters
from .ocv_soc import OCVSOC
from .online_id import RLSIdentifier
from .soh_estimator import JointEKFSoH
from .twin_sync import TwinSync

_Z95 = 1.959963984540054   # 97.5th percentile of the standard normal


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


@dataclass(frozen=True)
class TwinState:
    """An immutable snapshot of the twin's belief at one instant.

    All uncertainties are 1-σ; the ``*_95_ci`` helpers widen them to a 95 %
    credible interval (Gaussian) and clip to physical ranges.
    """

    t_s: float
    soc: float
    soc_sigma: float
    soh: float
    soh_sigma: float
    capacity_Ah: float
    r0_ohm: float
    temperature_C: float
    voltage_V: float
    current_A: float
    # health / assimilation
    residual_V: float
    residual_rms_V: float
    drift: bool
    # observability
    excitation: float
    capacity_observable: bool
    # trust
    confidence: float
    soc_confidence: float
    soh_confidence: float
    confidence_breakdown: dict
    n_updates: int

    # ---- credible intervals -------------------------------------------
    @property
    def soc_95_ci(self) -> tuple[float, float]:
        lo, hi = self.soc - _Z95 * self.soc_sigma, self.soc + _Z95 * self.soc_sigma
        return (_clip01(lo), _clip01(hi))

    @property
    def soh_95_ci(self) -> tuple[float, float]:
        lo = max(0.0, self.soh - _Z95 * self.soh_sigma)
        hi = min(1.5, self.soh + _Z95 * self.soh_sigma)
        return (float(lo), float(hi))

    def to_dict(self) -> dict:
        """A JSON-ready view — estimates alongside their 95 % intervals."""
        d = asdict(self)
        d["soc_95_ci"] = list(self.soc_95_ci)
        d["soh_95_ci"] = list(self.soh_95_ci)
        return d


@dataclass
class BatteryDigitalTwin:
    """A single, continuously calibrated, uncertainty-aware cell twin.

    Parameters
    ----------
    params, ocv_curve
        The twin's ECM cell model.
    q_nominal_Ah
        Beginning-of-life capacity for SoH normalisation (defaults to
        ``params.Q_nom_Ah``).
    excitation_window
        Number of recent samples over which SoC movement is measured to judge
        capacity observability.
    excitation_ref
        SoC swing (fraction) considered "enough" excitation to identify capacity;
        ``excitation / excitation_ref`` (clipped to 1) is the excitation score.
    warmup_steps
        Updates over which the twin ramps from cold-start to full confidence.
    soc_sigma_ref, soh_sigma_ref
        1-σ uncertainties at which the corresponding confidence term reaches 0.
    """

    params: ECMParameters
    ocv_curve: OCVSOC = field(default_factory=OCVSOC)
    q_nominal_Ah: float | None = None
    excitation_window: int = 300
    excitation_ref: float = 0.05
    warmup_steps: int = 20
    soc_sigma_ref: float = 0.05
    soh_sigma_ref: float = 0.10

    def __post_init__(self) -> None:
        self.q_nominal_Ah = float(self.q_nominal_Ah
                                  if self.q_nominal_Ah is not None
                                  else self.params.Q_nom_Ah)
        # The probabilistic core (SoC + capacity + their σ).
        self.estimator = JointEKFSoH(params=self.params, ocv_curve=self.ocv_curve,
                                     q_nominal_Ah=self.q_nominal_Ah)
        # The drift / assimilation monitor (residual vs the real measurement).
        self.sync = TwinSync(params=self.params, ocv_curve=self.ocv_curve)
        # Online ohmic-resistance tracker.
        self.rls = RLSIdentifier(r0_init=self.params.R0)
        self.reset(1.0)

    # ------------------------------------------------------------------
    def reset(self, soc0: float = 1.0) -> None:
        self.estimator.reset(soc0)
        self.sync.reset(soc0)
        self.rls.reset(self.params.R0)
        self._t = 0.0
        self._n = 0
        self._soc_hist: deque[float] = deque(maxlen=self.excitation_window)
        self.state: TwinState | None = None

    # ------------------------------------------------------------------
    def _confidence(self, soc_sigma: float, soh_sigma: float,
                    residual_rms: float, excitation: float) -> dict:
        """Fuse the trust signals into per-state and overall confidences."""
        c_fresh = _clip01(self._n / max(self.warmup_steps, 1))
        # Residual stability: full trust below the drift threshold, 0 at twice it.
        c_resid = _clip01(1.0 - residual_rms / (2.0 * self.sync.drift_threshold_V))
        c_soc = _clip01(1.0 - soc_sigma / self.soc_sigma_ref)
        c_soh = _clip01(1.0 - soh_sigma / self.soh_sigma_ref)
        c_excite = _clip01(excitation / self.excitation_ref)
        # SoC needs a converged filter and a matching model; SoH additionally
        # needs excitation to make capacity observable.
        soc_conf = c_fresh * min(c_soc, c_resid)
        soh_conf = c_fresh * min(c_soh, c_excite, c_resid)
        overall = c_fresh * c_resid * (0.6 * c_soc + 0.4 * min(c_soh, c_excite))
        return {
            "confidence": _clip01(overall),
            "soc_confidence": _clip01(soc_conf),
            "soh_confidence": _clip01(soh_conf),
            "breakdown": {
                "freshness": c_fresh, "residual": c_resid,
                "soc_uncertainty": c_soc, "soh_uncertainty": c_soh,
                "excitation": c_excite,
            },
        }

    def update(self, voltage: float, current: float, dt: float = 1.0,
               temperature_C: float = 25.0) -> TwinState:
        """Assimilate one live sample (``current > 0`` = discharge)."""
        self.estimator.update(current, voltage, dt, temperature_C=temperature_C)
        sync_out = self.sync.assimilate(current, voltage, dt, temperature_C=temperature_C)
        r0 = self.rls.update(current, voltage)

        self._t += dt
        self._n += 1
        soc = self.estimator.soc
        self._soc_hist.append(soc)
        excitation = float(max(self._soc_hist) - min(self._soc_hist))

        soc_sigma = self.estimator.soc_uncertainty_1sigma
        soh_sigma = self.estimator.soh_uncertainty_1sigma
        conf = self._confidence(soc_sigma, soh_sigma,
                                sync_out["residual_rms_V"], excitation)

        self.state = TwinState(
            t_s=self._t,
            soc=soc, soc_sigma=soc_sigma,
            soh=self.estimator.soh, soh_sigma=soh_sigma,
            capacity_Ah=self.estimator.capacity_Ah,
            r0_ohm=float(r0),
            temperature_C=float(temperature_C),
            voltage_V=float(voltage), current_A=float(current),
            residual_V=sync_out["residual_V"],
            residual_rms_V=sync_out["residual_rms_V"],
            drift=bool(sync_out["drift"]),
            excitation=excitation,
            capacity_observable=bool(excitation >= self.excitation_ref),
            confidence=conf["confidence"],
            soc_confidence=conf["soc_confidence"],
            soh_confidence=conf["soh_confidence"],
            confidence_breakdown=conf["breakdown"],
            n_updates=self._n,
        )
        return self.state

    def run(self, currents: np.ndarray, voltages: np.ndarray, dt: float,
            temperatures: np.ndarray | None = None) -> list[TwinState]:
        """Assimilate a full trace; returns the per-sample state snapshots."""
        currents = np.asarray(currents, float)
        voltages = np.asarray(voltages, float)
        n = len(currents)
        T = np.full(n, 25.0) if temperatures is None else np.asarray(temperatures, float)
        return [self.update(float(voltages[k]), float(currents[k]), dt,
                            temperature_C=float(T[k])) for k in range(n)]
