"""Validate the estimators against a real (or synthetic) drive cycle.

Runs the full pipeline — optional ECM parameter-ID, the estimator leaderboard,
and a bucketed validation report — on a drive-cycle CSV, keeping **real** and
**synthetic** results clearly tagged.  The public datasets are large and
licence-bound, so none are committed; drop a file in and point this at it:

    python scripts/validate_real.py path/to/lg_18650_us06_0degC.csv --chem nmc
    python scripts/validate_real.py            # synthetic fixture (a demo run)

Recommended sources (see bms.DATASET_SOURCES for URLs):
* SoC  — LG 18650HG2 drive cycles at several temperatures (coulomb-counted SoC).
* SoH  — NASA PCoE 18650 cycled to failure (capacity + EIS).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bms  # noqa: E402


def _arg(flag: str, default: str | None = None) -> str | None:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main() -> int:
    chem = _arg("--chem", "nmc")
    paths = [a for a in sys.argv[1:] if not a.startswith("--")
             and a != (_arg("--chem") or "")]
    path = paths[0] if paths else None

    if path:
        data = bms.load_drivecycle_csv(path, chemistry=chem)
        print(f"Loaded REAL trace: {data.name}  ({data.n} samples, "
              f"source={data.source})")
    else:
        data = bms.synthetic_drivecycle(chem, duration_s=1800, seed=1)
        print(f"No file given — using the SYNTHETIC fixture ({data.name}). "
              "Drop a real CSV in for a real-data run.")

    print(f"\n== Estimator leaderboard (SoC RMSE, source={data.source}) ==")
    board = bms.estimator_leaderboard(data, temperature_aware=True)
    view = board.copy()
    view["rmse_%"] = (view["rmse"] * 100).round(3)
    view["mae_%"] = (view["mae"] * 100).round(3)
    print(view[["rmse_%", "mae_%", "runtime_s"]].to_string())

    print("\n== Validation report — SoC error bucketed by C-rate ==")
    rep = bms.validation_report(data, board.index[0], buckets_by="c_rate", n_buckets=4)
    print(rep.to_string(index=False))

    print(f"\nBest estimator: {board.index[0]} "
          f"({board.iloc[0]['rmse'] * 100:.2f}% SoC RMSE, source={data.source})")
    if data.source == "synthetic":
        print("NOTE: synthetic — for a credibility claim, run this on a real "
              "dataset (LG/NASA); results are tagged 'real' automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
