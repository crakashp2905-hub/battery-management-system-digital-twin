"""Validate the twin against the **real** NASA PCoE battery dataset.

The NASA Ames Prognostics Center of Excellence Li-ion aging dataset (18650 cells
cycled to failure) is US-Government **public domain**.  This script fetches the
``.mat`` files from a public mirror (or takes local paths), then runs two
genuine real-data validations, keeping results tagged ``source="real"``:

* **SoC estimators** — build a drive cycle from a real discharge (coulomb-counted
  SoC as ground truth) and run the estimator leaderboard; and
* **State of health** — extract the real capacity-fade curve per cell and report
  SoH and the cycle at which it crosses end-of-life.

    python scripts/validate_nasa.py                 # downloads B0005..B0018
    python scripts/validate_nasa.py path/to/B0005.mat

Data: NASA Ames PCoE Battery Data Set (public domain).  Nothing here is a fit to
this data — it is an out-of-the-box run of the shipped models on real cells.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import bms  # noqa: E402

_MIRROR = "https://raw.githubusercontent.com/natskiu/Nasa-Battery/main/Data/{}.mat"
_CACHE = Path(__file__).resolve().parents[1] / ".nasa_cache"


def fetch(cell: str) -> Path:
    """Download a NASA cell's .mat from the public mirror (cached)."""
    _CACHE.mkdir(exist_ok=True)
    dest = _CACHE / f"{cell}.mat"
    if not dest.exists():
        print(f"  downloading {cell} …", flush=True)
        urllib.request.urlretrieve(_MIRROR.format(cell), dest)
    return dest


def _discharges(mat_path: str):
    import scipy.io as sio
    m = sio.loadmat(str(mat_path), simplify_cells=True)
    key = next(k for k in m if k.startswith("B0"))
    return m[key]["cycle"], key


def soc_validation(mat_path: str, cell: str, cycle_index: int = 2) -> None:
    cyc, _ = _discharges(mat_path)
    disch = [c for c in cyc if c["type"] == "discharge"]
    d = disch[cycle_index]["data"]
    t = np.asarray(d["Time"], float)
    current = -np.asarray(d["Current_measured"], float)     # repo: discharge > 0
    voltage = np.asarray(d["Voltage_measured"], float)
    temp = np.asarray(d["Temperature_measured"], float)
    cap = float(d["Capacity"])
    dt = np.diff(t, prepend=t[0])
    soc_true = np.clip(1.0 - np.cumsum(current * dt) / 3600.0 / cap, 0.0, 1.0)

    data = bms.DriveCycleData(time_s=t, current_A=current, voltage_V=voltage,
                              temperature_C=temp, soc_true=soc_true,
                              capacity_Ah=cap, chemistry="nmc",
                              name=f"NASA_{cell}", source="real")
    print(f"\n== SoC estimators on REAL {cell} discharge "
          f"({len(t)} samples, {cap:.3f} Ah, ~{current.mean():.1f} A) ==")
    board = bms.estimator_leaderboard(data)
    for est, row in board.iterrows():
        print(f"  {est:11s} {row['rmse'] * 100:6.2f}% SoC RMSE")
    print("  note: coulomb-counting excels on real data; the voltage filters need "
          "a cell-specific OCV (the generic table is not this LCO cell) -- see bms.calibration.")


def soh_validation(mats: dict) -> None:
    print("\n== State of health on REAL NASA cells (capacity fade) ==")
    print(f"  {'cell':7s}{'cycles':>7s}{'SoH_0':>7s}{'SoH_N':>7s}{'EOL@80%rated':>14s}")
    for cell, path in mats.items():
        n, q = bms.nasa_mat_to_capacity(str(path))
        below = np.where(q < 1.6)[0]                        # 80 % of 2 Ah rated
        eol = int(n[below[0]]) if below.size else -1
        print(f"  {cell:7s}{len(q):7d}{q[0] / 2 * 100:6.1f}%{q[-1] / 2 * 100:6.1f}%"
              f"{eol:>14d}")


def main() -> int:
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if paths:
        mats = {Path(p).stem: p for p in paths}
    else:
        print("No paths given — fetching NASA cells from the public mirror:")
        mats = {c: fetch(c) for c in ("B0005", "B0006", "B0007", "B0018")}

    first = next(iter(mats))
    soc_validation(mats[first], first)
    soh_validation(mats)
    print("\nAll results are REAL (source='real'); the models were not fit to this data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
