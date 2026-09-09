"""Online ECM parameter identification via recursive least squares (RLS).

The estimators assume a fixed ohmic resistance ``R0``, but ``R0`` drifts with
temperature and age.  :class:`RLSIdentifier` tracks it **online** from the
terminal voltage's response to current steps, so the twin can keep its ECM
current without an offline re-fit — lighter than a dual-EKF.

Method
------
At a current *step* the RC-branch voltages and the OCV are continuous (they do
not jump), so the instantaneous change in terminal voltage is dominated by the
ohmic drop::

    ΔV ≈ −R0 · ΔI

RLS with a forgetting factor fits ``R0`` to that relation.  Steps carry the
information (large ``|ΔI|``); between steps ``ΔI ≈ 0`` and the estimate simply
coasts, which is exactly the desired behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RLSIdentifier:
    """Recursive-least-squares tracker for the ohmic resistance ``R0`` [Ω].

    Parameters
    ----------
    r0_init : float
        Initial ``R0`` estimate [Ω].
    forgetting : float
        RLS forgetting factor λ ∈ (0, 1]; smaller tracks faster but noisier.
        0.995 is a good default for slow thermal/aging drift.
    p0 : float
        Initial covariance — larger lets the estimate move off ``r0_init`` fast.
    min_abs_di_A : float
        Ignore steps smaller than this |ΔI| [A]; below it ΔV is mostly
        OCV/RC drift and noise, which would bias the fit.
    """

    r0_init: float = 0.025
    forgetting: float = 0.995
    p0: float = 1.0
    min_abs_di_A: float = 0.05
    r0: float = field(init=False)
    _P: float = field(init=False)
    _prev: tuple[float, float] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self, r0: float | None = None) -> None:
        self.r0 = float(self.r0_init if r0 is None else r0)
        self._P = float(self.p0)
        self._prev = None

    def update_step(self, dv: float, di: float) -> float:
        """One RLS update from a voltage/current *difference* (ΔV, ΔI)."""
        if abs(di) < self.min_abs_di_A:
            return self.r0                       # uninformative — coast
        phi = -float(di)                          # regressor: ΔV = φ·R0
        lam = self.forgetting
        denom = lam + phi * self._P * phi
        K = self._P * phi / denom
        err = float(dv) - phi * self.r0
        self.r0 = float(max(1e-6, self.r0 + K * err))   # resistance stays positive
        self._P = float((self._P - K * phi * self._P) / lam)
        return self.r0

    def update(self, current: float, voltage: float) -> float:
        """Feed raw (current, voltage) samples; differences are formed internally."""
        if self._prev is not None:
            self.update_step(voltage - self._prev[1], current - self._prev[0])
        self._prev = (float(current), float(voltage))
        return self.r0

    def run(self, currents: np.ndarray, voltages: np.ndarray) -> np.ndarray:
        """Track ``R0`` over a full trace; returns the per-sample estimate."""
        currents = np.asarray(currents, float)
        voltages = np.asarray(voltages, float)
        out = np.empty(len(currents))
        for k in range(len(currents)):
            out[k] = self.update(float(currents[k]), float(voltages[k]))
        return out
