"""Performance benchmark — throughput of the twin's hot paths.

Times the core operations and reports **steps per second**, so a change that
slows the plant or an estimator is visible.  Everything is deterministic and
seeded; run it before/after a change to catch regressions.

    python scripts/benchmark_perf.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import bms  # noqa: E402


def _time(fn, n_steps: int, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return n_steps / best                       # steps per second


def benchmark(n: int = 3600, n_sup: int = 600) -> dict:
    i = np.asarray(bms.generate_load_profile(n, dt=1.0, mode="drive", c_rate=1.0,
                                             capacity_Ah=2.3, seed=0), float)
    p = bms.ECMParameters(Q_nom_Ah=2.3)
    ocv = bms.OCVSOC()
    v = bms.SecondOrderECM(params=p, ocv_curve=ocv).simulate(i, 1.0, soc0=0.9)["v_terminal"]

    results = {}
    results["ecm_simulate"] = _time(
        lambda: bms.SecondOrderECM(params=p, ocv_curve=ocv).simulate(i, 1.0, soc0=0.9), n)

    for name in ("coulomb", "ekf", "ukf", "pf"):
        est = bms.make_soc_estimator(name, params=p, ocv_curve=ocv, capacity_Ah=2.3)
        results[f"estimator_{name}"] = _time(
            lambda est=est: (est.reset(0.9), est.run(i, v, 1.0)), n)

    results["spm_simulate"] = _time(
        lambda: bms.SingleParticleModel(bms.SPMParams(Q_nom_Ah=2.3)).simulate(i, 1.0, soc0=0.9), n)

    core = bms.CoreParams(q_nom_as=2.3 * 3600)

    def _core_run():
        st = bms.core_reset(0.9)
        for k in range(n):
            bms.core_step(st, core, float(v[k]), float(i[k]), 25.0, 1.0)
    results["control_core_step"] = _time(_core_run, n)

    def _sup_run():
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=0))
        sup = bms.BMSSupervisor(pack, bms.ThermalModel(n_cells=4), bms.HybridFaultDetector())
        for k in range(n_sup):
            sup.step(1.0, 1.0, k=k)
    results["supervisor_step"] = _time(_sup_run, n_sup)
    return results


if __name__ == "__main__":
    print(f"{'operation':22s} {'steps/sec':>12s}")
    for name, sps in benchmark().items():
        print(f"{name:22s} {sps:12,.0f}")
