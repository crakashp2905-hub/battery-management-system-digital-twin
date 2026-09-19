"""Digital-twin replay engine — an experimentation platform for twin versions.

Battery-twin work lives or dies on being able to answer *"is version B actually
better than version A on real data?"*.  This module stores a drive trace once and
**replays** it through any number of twin configurations (SoC estimators, ECM
variants, the hybrid model), reporting the same metrics for each so they compare
apples to apples: SoC RMSE, final-SoC error and runtime.

    engine = ReplayEngine(trace)
    board = engine.compare({"ekf": make_soc_estimator("ekf", ...),
                            "ukf": make_soc_estimator("ukf", ...),
                            "pf":  make_soc_estimator("pf",  ...)})

The result is a ranked leaderboard — the reproducible A/B bench a twin needs
before shipping a change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DriveTrace:
    """A stored drive cycle to replay: current, measured voltage, optional truth."""

    currents: np.ndarray
    voltages: np.ndarray
    dt: float = 1.0
    soc_true: np.ndarray | None = None
    temperatures: np.ndarray | None = None
    soc0: float = 0.9
    name: str = "trace"

    def __post_init__(self) -> None:
        self.currents = np.asarray(self.currents, float)
        self.voltages = np.asarray(self.voltages, float)
        if self.soc_true is not None:
            self.soc_true = np.asarray(self.soc_true, float)

    @property
    def n(self) -> int:
        return len(self.currents)


@dataclass(frozen=True)
class ReplayResult:
    """One twin configuration's performance on a replayed trace."""

    config: str
    soc_rmse: float
    final_soc_error: float
    runtime_s: float
    n_samples: int

    def to_dict(self) -> dict:
        return {"config": self.config, "soc_rmse": self.soc_rmse,
                "final_soc_error": self.final_soc_error,
                "runtime_s": self.runtime_s, "n_samples": self.n_samples}


@dataclass
class ReplayEngine:
    """Replay a stored :class:`DriveTrace` through twin configurations."""

    trace: DriveTrace
    results: list = field(default_factory=list)

    def replay(self, config: str, estimator) -> ReplayResult:
        """Run one SoC estimator over the trace and score it.

        ``estimator`` is any object with ``reset(soc0)`` and
        ``run(currents, voltages, dt) -> soc_series`` (the ``bms`` estimator
        interface).
        """
        tr = self.trace
        estimator.reset(tr.soc0)
        t0 = time.perf_counter()
        soc_est = np.asarray(estimator.run(tr.currents, tr.voltages, tr.dt), float)
        runtime = time.perf_counter() - t0

        if tr.soc_true is not None:
            err = soc_est - tr.soc_true
            soc_rmse = float(np.sqrt(np.mean(err ** 2)))
            final = float(abs(soc_est[-1] - tr.soc_true[-1]))
        else:
            soc_rmse = float("nan")
            final = float("nan")
        res = ReplayResult(config, soc_rmse, final, float(runtime), tr.n)
        self.results.append(res)
        return res

    def compare(self, configs: dict) -> list:
        """Replay several configurations and return them ranked by SoC RMSE."""
        out = [self.replay(name, est) for name, est in configs.items()]
        finite = [r for r in out if np.isfinite(r.soc_rmse)]
        rank = sorted(finite, key=lambda r: r.soc_rmse)
        rest = [r for r in out if not np.isfinite(r.soc_rmse)]
        return rank + rest

    def best(self, configs: dict) -> ReplayResult:
        return self.compare(configs)[0]
