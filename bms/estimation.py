"""
Model-agnostic estimator framework: protocols, a registry, and a BYO-model hook.

The twin ships four SoC estimators (Coulomb counter, EKF, UKF, NumPy LSTM) that
already share almost the same surface.  This module formalises that surface so
**any** algorithm is interchangeable:

* :class:`SocEstimator` — a ``runtime_checkable`` structural protocol.  Anything
  with ``reset`` / ``update`` / ``soc`` / ``run`` satisfies it; the built-in
  estimators do so without modification.
* A **registry** (:func:`make_soc_estimator`, :func:`available_soc_estimators`)
  so estimators are selected by name/config instead of hard-wired classes —
  the supervisor, dashboard, and benchmarks take an interface, not a class.
* :class:`FunctionSocEstimator` — wrap any trained model (scikit-learn, PyTorch,
  an ONNX session, a lookup table) behind the standard interface with a single
  callable.  This is what makes the framework genuinely model-agnostic.
* :func:`soc_estimate` — read a value **with uncertainty** (:class:`Estimate`)
  from any estimator, using its 1-σ covariance when it exposes one.

Every registered estimator is exercised by a shared contract test
(``tests/test_bms.py::TestEstimatorRegistry``), so a new model "just works"
once it satisfies the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Estimate:
    """A scalar estimate with optional 1-σ uncertainty.

    ``sigma`` is ``nan`` when the underlying model does not quantify uncertainty
    (e.g. an open-loop Coulomb counter), which callers can render as "unknown".
    """

    value: float
    sigma: float = float("nan")

    @property
    def has_uncertainty(self) -> bool:
        return not np.isnan(self.sigma)


@runtime_checkable
class SocEstimator(Protocol):
    """Universal interface every SoC estimator provides: name, reset, run.

    This is the batch contract (run a whole current/voltage trace).  Both the
    recursive filters and the sequence LSTM satisfy it.
    """

    name: str

    def reset(self, soc0: float = 1.0) -> None: ...
    def run(self, currents: np.ndarray, voltages: np.ndarray,
            dt: float) -> np.ndarray: ...


@runtime_checkable
class RecursiveSocEstimator(SocEstimator, Protocol):
    """Online estimators that additionally support single-step ``update`` and a
    live ``soc`` read-out (Coulomb counter, EKF, UKF, BYO adapters).  The
    sequence LSTM is a :class:`SocEstimator` but not a recursive one.
    """

    def update(self, current: float, voltage: float, dt: float) -> float: ...
    @property
    def soc(self) -> float: ...


def soc_estimate(estimator: RecursiveSocEstimator) -> Estimate:
    """Read an estimator's current SoC with uncertainty when available.

    Uses ``soc_uncertainty_1sigma`` (EKF/UKF expose it from their covariance);
    falls back to ``nan`` for estimators that do not quantify uncertainty.
    """
    sigma = getattr(estimator, "soc_uncertainty_1sigma", float("nan"))
    sigma = float(sigma() if callable(sigma) else sigma)
    return Estimate(value=float(estimator.soc), sigma=sigma)


# ======================================================================
# Bring-your-own-model adapter
# ======================================================================
class FunctionSocEstimator:
    """Adapt any callable into a :class:`SocEstimator`.

    ``fn(voltage, current, dt, temperature_C, prev_soc) -> soc`` — wrap a
    scikit-learn ``.predict``, a Torch forward pass, an ONNX Runtime session,
    or a hand-written rule.  The result is clipped to ``[0, 1]`` and exposed
    through the standard recursive interface.

    Examples
    --------
    ::

        import onnxruntime as ort
        sess = ort.InferenceSession("soc.onnx")
        f = lambda v, i, dt, T, s: float(
            sess.run(None, {"x": [[v, i, T, s]]})[0][0][0])
        est = FunctionSocEstimator(f, name="onnx")
    """

    def __init__(self, fn: Callable[[float, float, float, float, float], float],
                 name: str = "custom", soc0: float = 1.0):
        self.fn = fn
        self.name = name
        self._soc = float(np.clip(soc0, 0.0, 1.0))

    def reset(self, soc0: float = 1.0) -> None:
        self._soc = float(np.clip(soc0, 0.0, 1.0))

    def update(self, current: float, voltage: float, dt: float,
               temperature_C: float = 25.0) -> float:
        raw = self.fn(voltage, current, dt, temperature_C, self._soc)
        self._soc = float(np.clip(raw, 0.0, 1.0))
        return self._soc

    @property
    def soc(self) -> float:
        return self._soc

    def run(self, currents: np.ndarray, voltages: np.ndarray, dt: float,
            soc0: float = 1.0, temperatures: np.ndarray | None = None) -> np.ndarray:
        self.reset(soc0)
        n = len(currents)
        T = np.full(n, 25.0) if temperatures is None else np.asarray(temperatures, float)
        out = np.empty(n)
        for k in range(n):
            out[k] = self.update(float(currents[k]), float(voltages[k]), dt,
                                 temperature_C=float(T[k]))
        return out


# ======================================================================
# Registry
# ======================================================================
_SOC_REGISTRY: dict[str, Callable[..., SocEstimator]] = {}


def register_soc_estimator(name: str) -> Callable:
    """Decorator registering a factory ``(**ctx) -> SocEstimator`` under *name*."""
    def deco(factory: Callable[..., SocEstimator]) -> Callable[..., SocEstimator]:
        _SOC_REGISTRY[name] = factory
        return factory
    return deco


def available_soc_estimators() -> list[str]:
    """Names of all registered SoC estimators."""
    return sorted(_SOC_REGISTRY)


def make_soc_estimator(name: str, *, params=None, ocv_curve=None,
                       capacity_Ah: float = 2.3, **kwargs) -> SocEstimator:
    """Construct a registered SoC estimator by name.

    A common construction context (``params``, ``ocv_curve``, ``capacity_Ah``)
    is passed to every factory; each uses what it needs and ignores the rest,
    so callers can swap algorithms without knowing their constructors.
    """
    if name not in _SOC_REGISTRY:
        raise KeyError(f"unknown SoC estimator {name!r}; "
                       f"available: {available_soc_estimators()}")
    return _SOC_REGISTRY[name](params=params, ocv_curve=ocv_curve,
                               capacity_Ah=capacity_Ah, **kwargs)


# ---- built-in registrations (lazy imports avoid import cycles) --------
@register_soc_estimator("coulomb")
def _make_coulomb(*, capacity_Ah: float = 2.3, soc0: float = 1.0, **_):
    from .soc_estimators import CoulombCounter
    return CoulombCounter(capacity_Ah=capacity_Ah, soc0=soc0)


@register_soc_estimator("ekf")
def _make_ekf(*, params=None, ocv_curve=None, **_):
    from .ecm import ECMParameters
    from .ocv_soc import OCVSOC
    from .soc_estimators import EKFEstimator
    return EKFEstimator(params=params or ECMParameters(),
                        ocv_curve=ocv_curve or OCVSOC())


@register_soc_estimator("ukf")
def _make_ukf(*, params=None, ocv_curve=None, **_):
    from .ecm import ECMParameters
    from .ocv_soc import OCVSOC
    from .soc_estimators import UKFEstimator
    return UKFEstimator(params=params or ECMParameters(),
                        ocv_curve=ocv_curve or OCVSOC())


@register_soc_estimator("lstm")
def _make_lstm(*, seed: int = 0, hidden_size: int = 16, **_):
    from .soc_estimators import LSTMEstimator
    return LSTMEstimator(hidden_size=hidden_size, seed=seed)


@register_soc_estimator("joint_ekf")
def _make_joint_ekf(*, params=None, ocv_curve=None, capacity_Ah: float = 2.3, **_):
    from .ecm import ECMParameters
    from .ocv_soc import OCVSOC
    from .soh_estimator import JointEKFSoH
    return JointEKFSoH(params=params or ECMParameters(Q_nom_Ah=capacity_Ah),
                       ocv_curve=ocv_curve or OCVSOC(), q_nominal_Ah=capacity_Ah)


@register_soc_estimator("bias_ekf")
def _make_bias_ekf(*, params=None, ocv_curve=None, capacity_Ah: float = 2.3, **_):
    from .ecm import ECMParameters
    from .ocv_soc import OCVSOC
    from .soc_estimators import BiasEKFEstimator
    return BiasEKFEstimator(params=params or ECMParameters(Q_nom_Ah=capacity_Ah),
                            ocv_curve=ocv_curve or OCVSOC())


# ======================================================================
# Framework adapters — plug a *trained* model in as a SoC estimator
# ======================================================================
def _default_features(voltage: float, current: float, dt: float,
                      temperature_C: float, prev_soc: float) -> list[float]:
    """Default per-step feature vector: [voltage, current, temperature, prev_soc]."""
    return [voltage, current, temperature_C, prev_soc]


@dataclass
class ModelCard:
    """Provenance + validity metadata for a plugged-in ML SoC model.

    Records the feature contract and the regime the model was trained on, so a
    deployed model can be checked against how it was built.  ``feature_names``
    also documents the **feature order** the model expects (the classic silent
    bug is feeding ``[I, V, T]`` to a model trained on ``[V, I, T]``).
    """

    feature_names: list[str]
    chemistry: str | None = None
    model_version: str = "1.0"
    training_temp_range_C: tuple[float, float] | None = None
    training_soc_range: tuple[float, float] | None = None
    train_rmse: float | None = None
    val_rmse: float | None = None
    created: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if self.created is None:
            import datetime as _dt
            self.created = _dt.date.today().isoformat()


@dataclass
class OODDetector:
    """Out-of-distribution flag via Mahalanobis distance to the training set.

    A model plugged in as an estimator silently extrapolates on inputs it never
    saw.  This flags them: ``d² = (x−μ)ᵀ Σ⁻¹ (x−μ)`` exceeding a threshold learnt
    from the training features (an empirical quantile — no Gaussian assumption).
    """

    mean: np.ndarray
    inv_cov: np.ndarray
    threshold: float

    @classmethod
    def fit(cls, X: np.ndarray, quantile: float = 0.99,
            margin: float = 1.5, ridge: float = 1e-9) -> "OODDetector":
        X = np.asarray(X, float)
        mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov = np.atleast_2d(cov) + ridge * np.eye(X.shape[1])
        inv = np.linalg.pinv(cov)
        d2 = np.einsum("ij,jk,ik->i", X - mean, inv, X - mean)
        thr = float(np.quantile(d2, quantile) * margin)
        return cls(mean=mean, inv_cov=inv, threshold=thr)

    def mahalanobis_sq(self, x: np.ndarray) -> float:
        d = np.asarray(x, float).ravel() - self.mean
        return float(d @ self.inv_cov @ d)

    def is_ood(self, x: np.ndarray) -> bool:
        return self.mahalanobis_sq(x) > self.threshold


class SklearnSocEstimator(FunctionSocEstimator):
    """Wrap a fitted scikit-learn regressor as a recursive SoC estimator.

    The model maps a per-step feature vector → SoC.  Default features are
    ``[voltage, current, temperature, prev_soc]``; pass ``feature_fn`` to change.

    Optionally attach a :class:`ModelCard` (provenance/feature contract) and an
    :class:`OODDetector`; when the detector flags an input the estimate is still
    returned but ``last_ood`` is set and ``n_ood`` counts it — feed that flag to
    the :class:`~bms.agent.ActionGate` so an out-of-distribution reading cannot
    drive an automated action.
    """

    def __init__(self, model, name: str = "sklearn", feature_fn=None,
                 card: "ModelCard | None" = None, ood_detector: "OODDetector | None" = None):
        ff = feature_fn or _default_features
        self.card = card
        self.ood_detector = ood_detector
        self.last_ood = False
        self.n_ood = 0

        def _fn(v, i, dt, T, s):
            x = np.asarray(ff(v, i, dt, T, s), float).reshape(1, -1)
            if self.ood_detector is not None:
                self.last_ood = self.ood_detector.is_ood(x)
                self.n_ood += int(self.last_ood)
            return float(np.ravel(model.predict(x))[0])

        super().__init__(_fn, name=name)
        self.model = model


class OnnxSocEstimator(FunctionSocEstimator):
    """Wrap an ONNX model (path or ``onnxruntime`` session) as a SoC estimator.

    Truly framework-neutral: train in *any* framework, export to ``.onnx``, run
    it here.  Requires ``onnxruntime`` (``pip install '.[onnx]'``).  Accepts the
    same optional ``card`` / ``ood_detector`` as :class:`SklearnSocEstimator`.
    """

    def __init__(self, model, name: str = "onnx", input_name=None, feature_fn=None,
                 card: "ModelCard | None" = None, ood_detector: "OODDetector | None" = None):
        if hasattr(model, "run"):
            session = model
        else:
            import onnxruntime as ort
            session = ort.InferenceSession(str(model))
        inp = input_name or session.get_inputs()[0].name
        ff = feature_fn or _default_features
        self.card = card
        self.ood_detector = ood_detector
        self.last_ood = False
        self.n_ood = 0

        def _fn(v, i, dt, T, s):
            x = np.asarray(ff(v, i, dt, T, s), np.float32).reshape(1, -1)
            if self.ood_detector is not None:
                self.last_ood = self.ood_detector.is_ood(x)
                self.n_ood += int(self.last_ood)
            return float(np.ravel(session.run(None, {inp: x})[0])[0])

        super().__init__(_fn, name=name)
        self.session = session


def _verify_sha256(path, expected: str) -> None:
    import hashlib
    with open(path, "rb") as fh:
        actual = hashlib.sha256(fh.read()).hexdigest()
    if actual.lower() != str(expected).lower():
        raise ValueError(f"checksum mismatch for {path!r}: expected {expected}, got {actual}")


def model_from_file(path, name=None, feature_fn=None, *, trust_pickle: bool = False,
                    sha256: str | None = None) -> SocEstimator:
    """Load a trained SoC model by file extension into a :class:`SocEstimator`.

    ``.onnx`` → :class:`OnnxSocEstimator` (data-only, **safe**);
    ``.joblib`` / ``.pkl`` → :class:`SklearnSocEstimator`.

    Security
    --------
    ``.joblib`` / ``.pkl`` are Python **pickles and execute arbitrary code when
    loaded** — only ever load artifacts you produced or fully trust.  This
    function refuses to unpickle unless you pass ``trust_pickle=True`` explicitly,
    and ``.onnx`` (no code execution) is the recommended deployment format.  Pass
    ``sha256`` to verify the file's checksum against an allowlisted hash first.
    """
    p = str(path)
    if sha256 is not None:
        _verify_sha256(path, sha256)
    if p.endswith(".onnx"):
        return OnnxSocEstimator(p, name=name or "onnx", feature_fn=feature_fn)
    if p.endswith((".joblib", ".pkl")):
        if not trust_pickle:
            raise ValueError(
                f"{p!r} is a Python pickle and can execute arbitrary code on load. "
                "Pass trust_pickle=True only for artifacts you trust (ideally with a "
                "sha256= checksum), or export the model to ONNX (.onnx) instead.")
        import joblib
        return SklearnSocEstimator(joblib.load(p), name=name or "sklearn",
                                   feature_fn=feature_fn)
    raise ValueError(f"unsupported model file {p!r}; use .onnx, .joblib, or .pkl")
