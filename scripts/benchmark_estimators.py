"""Benchmark the SoC estimators head-to-head across chemistries and conditions.

Runs :func:`bms.estimator_leaderboard` on a physically-consistent synthetic
drive cycle (the plant ECM's SoC is the exact ground truth) for every chemistry
and three operating conditions:

* **nominal**       — clean current, light voltage noise.
* **current bias**  — a constant current-sensor offset (the classic case where a
  Coulomb counter drifts but a voltage-feedback filter rejects the bias).
* **voltage noise** — heavier terminal-voltage measurement noise.

Prints a ranked table per (chemistry, condition) and, with ``--markdown``, emits
a compact winner-matrix suitable for pasting into docs.

Usage::

    python scripts/benchmark_estimators.py            # human-readable tables
    python scripts/benchmark_estimators.py --markdown  # docs matrix
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import bms  # noqa: E402

CHEMISTRIES = ["nmc", "lfp", "lmfp", "nca", "lmo", "lto", "ssb"]
ESTIMATORS = ["coulomb", "ekf", "ukf", "joint_ekf"]
CONDITIONS = {
    "nominal":       dict(noise_v=0.005, current_bias_A=0.0),
    "current_bias":  dict(noise_v=0.005, current_bias_A=0.15),
    "voltage_noise": dict(noise_v=0.020, current_bias_A=0.0),
}


def _board(chemistry: str, **kw):
    data = bms.synthetic_drivecycle(chemistry, duration_s=1800, seed=1, **kw)
    return bms.estimator_leaderboard(data, ESTIMATORS)


def run(markdown: bool = False) -> None:
    winners: dict[str, dict[str, str]] = {c: {} for c in CHEMISTRIES}
    rmse_pct: dict[tuple[str, str], dict[str, float]] = {}

    for chem in CHEMISTRIES:
        for cond, kw in CONDITIONS.items():
            board = _board(chem, **kw)
            winners[chem][cond] = str(board.index[0])          # lowest RMSE
            rmse_pct[(chem, cond)] = {
                est: float(board.loc[est, "rmse"]) * 100.0 for est in board.index}
            if not markdown:
                print(f"\n=== {chem.upper()} · {cond} "
                      f"(bias={kw['current_bias_A']} A, noise={kw['noise_v'] * 1e3:.0f} mV) ===")
                view = board.copy()
                view["rmse_%"] = (view["rmse"] * 100).round(3)
                view["mae_%"] = (view["mae"] * 100).round(3)
                print(view[["rmse_%", "mae_%", "runtime_s"]].to_string())

    if markdown:
        print("\n### Winner matrix (lowest SoC RMSE)\n")
        header = "| chemistry | " + " | ".join(CONDITIONS) + " |"
        print(header)
        print("|" + "---|" * (len(CONDITIONS) + 1))
        for chem in CHEMISTRIES:
            row = " | ".join(winners[chem][c] for c in CONDITIONS)
            print(f"| {chem} | {row} |")

    # Aggregate: mean RMSE per estimator over all (chemistry, condition) cells.
    print("\n### Mean SoC RMSE across all chemistries/conditions [%]\n")
    agg = {est: np.mean([rmse_pct[k][est] for k in rmse_pct]) for est in ESTIMATORS}
    for est in sorted(agg, key=agg.get):
        print(f"  {est:10s} {agg[est]:.3f}")


if __name__ == "__main__":
    run(markdown="--markdown" in sys.argv)
