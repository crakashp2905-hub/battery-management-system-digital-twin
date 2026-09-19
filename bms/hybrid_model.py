"""Physics + ML hybrid — learn only what the physics misses.

Replacing an ECM with a black-box neural net throws away everything the physics
already gets right and needs mountains of data to relearn it.  The defensible
hybrid keeps the physics and learns only its **residual**:

    V_terminal = V_ECM(SoC, I, T)  +  f_θ(I, SoC, |I|, T)

``f_θ`` is a small regressor trained on ``V_measured − V_ECM``, so it only has to
capture the higher-order effects the 2-RC model cannot — nonlinear polarisation,
diffusion tails, mild hysteresis — while the ECM carries the bulk of the signal.
This is the same principle behind physics-informed networks and neural-ODE
residuals; here it is implemented dependency-light on top of scikit-learn (a core
dependency) rather than a deep-learning stack, and it is honest about being a
residual learner, not a from-scratch electrochemical net.

:meth:`HybridResidualModel.score` reports the pure-physics vs hybrid RMSE so the
value added by the learned residual is always measurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ecm import ECMParameters, SecondOrderECM
from .ocv_soc import OCVSOC


def _default_regressor():
    """A small, scaled MLP — a smooth residual learner with no tuning needed."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        # alpha (L2) regularises the residual net so it generalises to unseen
        # traces in the same regime instead of memorising the training trajectory.
        MLPRegressor(hidden_layer_sizes=(32, 32), alpha=1e-2, max_iter=4000,
                     random_state=0, tol=1e-7),
    )


@dataclass
class HybridResidualModel:
    """ECM physics plus a learned voltage residual.

    Parameters
    ----------
    params, ocv_curve : the physics model (its own parameters are *not* changed).
    regressor : any scikit-learn-style regressor (``fit``/``predict``); defaults
        to a small scaled MLP.
    """

    params: ECMParameters
    ocv_curve: OCVSOC = field(default_factory=OCVSOC)
    regressor: object = None

    def __post_init__(self) -> None:
        if self.regressor is None:
            self.regressor = _default_regressor()
        self._fitted = False

    # ------------------------------------------------------------------
    def _physics(self, currents: np.ndarray, dt: float, soc0: float,
                 temperatures: np.ndarray | None):
        sim = SecondOrderECM(params=self.params, ocv_curve=self.ocv_curve
                             ).simulate(np.asarray(currents, float), dt, soc0=soc0,
                                        temperatures=temperatures)
        return sim["v_terminal"], sim["soc"]

    @staticmethod
    def _features(currents: np.ndarray, soc: np.ndarray,
                  temperatures: np.ndarray) -> np.ndarray:
        i = np.asarray(currents, float)
        return np.column_stack([i, soc, np.abs(i), i ** 2, temperatures])

    def _temps(self, n: int, temperatures) -> np.ndarray:
        return np.full(n, 25.0) if temperatures is None else np.asarray(temperatures, float)

    # ------------------------------------------------------------------
    def fit(self, currents: np.ndarray, voltages: np.ndarray, dt: float,
            soc0: float, temperatures: np.ndarray | None = None
            ) -> "HybridResidualModel":
        """Fit ``f_θ`` to the ECM residual on a measured trace."""
        currents = np.asarray(currents, float)
        voltages = np.asarray(voltages, float)
        T = self._temps(len(currents), temperatures)
        v_phys, soc = self._physics(currents, dt, soc0, T)
        residual = voltages - v_phys
        self.regressor.fit(self._features(currents, soc, T), residual)
        self._fitted = True
        return self

    def predict(self, currents: np.ndarray, dt: float, soc0: float,
                temperatures: np.ndarray | None = None) -> np.ndarray:
        """Hybrid terminal voltage = physics + learned residual."""
        currents = np.asarray(currents, float)
        T = self._temps(len(currents), temperatures)
        v_phys, soc = self._physics(currents, dt, soc0, T)
        if not self._fitted:
            return v_phys
        return v_phys + self.regressor.predict(self._features(currents, soc, T))

    def score(self, currents: np.ndarray, voltages: np.ndarray, dt: float,
              soc0: float, temperatures: np.ndarray | None = None) -> dict:
        """Compare pure-physics vs hybrid voltage RMSE on a trace."""
        currents = np.asarray(currents, float)
        voltages = np.asarray(voltages, float)
        T = self._temps(len(currents), temperatures)
        v_phys, _ = self._physics(currents, dt, soc0, T)
        v_hyb = self.predict(currents, dt, soc0, T)
        physics_rmse = float(np.sqrt(np.mean((voltages - v_phys) ** 2)))
        hybrid_rmse = float(np.sqrt(np.mean((voltages - v_hyb) ** 2)))
        improvement = (100.0 * (physics_rmse - hybrid_rmse) / physics_rmse
                       if physics_rmse > 0 else 0.0)
        return {"physics_rmse_V": physics_rmse, "hybrid_rmse_V": hybrid_rmse,
                "improvement_pct": float(improvement)}
