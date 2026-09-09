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


# Temperatures for the sweep — isothermal traces the plant genuinely runs at.
TEMPERATURES = [25.0, 15.0, 5.0, -5.0, -15.0]


def run_temperature(markdown: bool = False,
                    chemistries=("nmc", "lfp"), filt: str = "ekf") -> None:
    """Quantify the value of a temperature sensor, fairly.

    Cross-*temperature* RMSE is confounded (the true SoC trajectory and the
    physical operating point both shift with temperature), so we do not rank
    across temperatures.  Instead, at each temperature we score the SAME trace
    two ways — the filter **told** the temperature vs the filter **assuming
    25 °C** — an apples-to-apples measure of what a temperature sensor buys you.
    """
    if markdown:
        print("\n### Temperature-aware vs temperature-naive "
              f"(`{filt}`, SoC RMSE %, current bias 0.05 A)\n")
        cols = " | ".join(f"{int(t)}°C" for t in TEMPERATURES)
        print(f"| {filt} @ | {cols} |")
        print("|" + "---|" * (len(TEMPERATURES) + 1))

    for chem in chemistries:
        aware_row, naive_row = [], []
        for T in TEMPERATURES:
            data = bms.synthetic_drivecycle(chem, duration_s=1800, seed=1,
                                            current_bias_A=0.05, temperature_C=T)
            aware = bms.estimator_leaderboard(
                data, [filt], temperature_aware=True).loc[filt, "rmse"] * 100
            naive = bms.estimator_leaderboard(
                data, [filt], temperature_aware=False).loc[filt, "rmse"] * 100
            aware_row.append(aware)
            naive_row.append(naive)
        if markdown:
            print(f"| {chem} · aware | "
                  + " | ".join(f"{v:.2f}" for v in aware_row) + " |")
            print(f"| {chem} · naive | "
                  + " | ".join(f"{v:.2f}" for v in naive_row) + " |")
        else:
            print(f"\n=== {chem.upper()} · temperature sweep "
                  f"({filt}, bias 0.05 A) ===")
            print("  T[°C]   aware_RMSE%   naive_RMSE%   sensor_gain%")
            for T, a, n in zip(TEMPERATURES, aware_row, naive_row):
                print(f"  {T:>5.0f}   {a:10.3f}   {n:10.3f}   {n - a:+10.3f}")


def run_hysteresis(markdown: bool = False, filt: str = "ekf", n_seeds: int = 5) -> None:
    """Value of modelling OCV hysteresis, per chemistry (mean over seeds).

    The plant carries each chemistry's characteristic hysteresis; we score the
    same traces with the filter's OCV hysteresis-**aware** vs **naive** (forced
    to 0).  Flat-OCV chemistries (LFP/LMFP) gain the most, since there the
    hysteresis dominates the sparse OCV slope.
    """
    if markdown:
        print(f"\n### Hysteresis-aware vs naive (`{filt}`, SoC RMSE %, "
              f"mean of {n_seeds} seeds)\n")
        print("| chemistry | aware | naive | gain |")
        print("|---|---|---|---|")
    else:
        print(f"\n=== Hysteresis aware vs naive ({filt}, mean of {n_seeds} seeds) ===")
        print("  chem    aware%   naive%    gain%")
    for chem in CHEMISTRIES:
        a, n = [], []
        for seed in range(n_seeds):
            data = bms.synthetic_drivecycle(chem, duration_s=1800, seed=seed)
            a.append(bms.estimator_leaderboard(
                data, [filt], hysteresis_aware=True).loc[filt, "rmse"] * 100)
            n.append(bms.estimator_leaderboard(
                data, [filt], hysteresis_aware=False).loc[filt, "rmse"] * 100)
        am, nm = float(np.mean(a)), float(np.mean(n))
        if markdown:
            print(f"| {chem} | {am:.2f} | {nm:.2f} | **{nm - am:+.2f}** |")
        else:
            print(f"  {chem:5s}   {am:6.2f}   {nm:6.2f}   {nm - am:+6.2f}")


def run_bias(markdown: bool = False, chemistries=("nmc", "lfp"),
             biases=(0.15, -0.10)) -> None:
    """The `bias_ekf` recovers a current-sensor offset online (vs coulomb/ekf)."""
    from bms.ecm import ECMParameters
    from bms.ocv_soc import OCVSOC

    if markdown:
        print("\n### Current-sensor bias: SoC RMSE % and recovered bias\n")
        print("| chem · bias | coulomb | ekf | bias_ekf | recovered |")
        print("|---|---|---|---|---|")
    for chem in chemistries:
        d = bms.get_chemistry_props(chem)["default_ecm"]
        for bias in biases:
            data = bms.synthetic_drivecycle(chem, duration_s=1800, seed=1,
                                            current_bias_A=bias)
            board = bms.estimator_leaderboard(data, ["coulomb", "ekf", "bias_ekf"])
            est = bms.BiasEKFEstimator(
                params=ECMParameters(R0=d["R0"], R1=d["R1"], C1=d["C1"],
                                     R2=d["R2"], C2=d["C2"],
                                     Q_nom_Ah=data.capacity_Ah, chemistry=chem),
                ocv_curve=OCVSOC.from_chemistry(chem))
            est.reset(float(data.soc_true[0]))
            est.run(data.current_A, data.voltage_V, data.dt)
            cells = (board.loc["coulomb", "rmse"] * 100, board.loc["ekf", "rmse"] * 100,
                     board.loc["bias_ekf", "rmse"] * 100)
            if markdown:
                print(f"| {chem} · {bias:+.2f} A | {cells[0]:.2f} | {cells[1]:.2f} "
                      f"| **{cells[2]:.2f}** | {est.current_bias_A:+.3f} A |")
            else:
                print(f"{chem} bias {bias:+.2f}A -> coulomb {cells[0]:.2f}%  "
                      f"ekf {cells[1]:.2f}%  bias_ekf {cells[2]:.2f}%  "
                      f"(recovered {est.current_bias_A:+.3f} A)")


if __name__ == "__main__":
    md = "--markdown" in sys.argv
    if "--temperature" in sys.argv:
        run_temperature(markdown=md)
    elif "--hysteresis" in sys.argv:
        run_hysteresis(markdown=md)
    elif "--bias" in sys.argv:
        run_bias(markdown=md)
    else:
        run(markdown=md)
