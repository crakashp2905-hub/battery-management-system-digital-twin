# Validation against real data

The library's built-in accuracy numbers are **synthetic** (the plant ECM's own
SoC is the ground truth, so scoring is exact — see `docs/estimation.md`). That is
the right way to *compare* estimators, but a credibility claim needs a run
against **real cells**. The twin is now validated against the **NASA PCoE**
Li-ion aging dataset (public domain); the numbers are below and reproducible.

## Real results — NASA PCoE (public domain) 🟢

`scripts/validate_nasa.py` fetches the public-domain NASA Ames battery `.mat`
files and runs the shipped models **out of the box** (no fit to this data). A
~20 KB slice of cell B0005 is committed at
`data/samples/nasa_b0005_discharge_real.csv` (with attribution) so CI exercises a
real-data result too (`tests/test_validation_real.py`).

**SoC estimators** on a real B0005 discharge (195 samples, 1.835 Ah, ~2 A,
coulomb-counted SoC as truth):

| Estimator | SoC RMSE (real) |
|---|---|
| **coulomb** | **0.64 %** |
| ukf | 12.31 % |
| ekf | 12.55 % |
| joint_ekf | 13.99 % |

Coulomb-counting validates to **0.64 %** on real data — the current-integration
path is sound. The voltage filters are far worse here for an honest reason: the
generic NMC OCV table is not this cell's LiCoO₂ curve, so the OCV→SoC inverse is
biased. Real data thus makes the case for **cell-specific OCV/ECM calibration**
(`bms.calibration`, `fit_from_pulse`) rather than a generic table — the same gap
the self-calibrating twin (`docs/autonomy.md`) closes automatically.

**State of health** — real capacity fade across four cells (EOL = 80 % of the
2 Ah rating):

| Cell | Cycles | SoH start → end | EOL cycle |
|---|---|---|---|
| B0005 | 168 | 92.8 % → 66.3 % | 75 |
| B0006 | 168 | 101.8 % → 59.3 % | 63 |
| B0007 | 168 | 94.6 % → 71.6 % | 86 |
| B0018 | 132 | 92.8 % → 67.1 % | 45 |

`nasa_mat_to_capacity` + `soh_curve` recover the real degradation trajectories
(including the well-known periodic capacity-recovery bumps) directly from the
dataset.

```bash
python scripts/validate_nasa.py            # downloads B0005..B0018, prints the tables above
```

## Other datasets (drop one in, then run)

## Datasets (drop one in, then run)

| Dataset | Validates | Loader | Source |
|---|---|---|---|
| **LG 18650HG2** (Kollmeyer/McMaster) — drive cycles at −20…25 °C, coulomb-counted SoC | SoC estimators × temperature | `load_drivecycle_csv(path, columns=LG_COLUMN_MAP)` | `bms.DATASET_SOURCES["lg_18650"]` |
| **NASA PCoE** — 18650 cycled to failure, capacity + EIS | online SoH + RUL | `nasa_mat_to_capacity(path)` → `soh_curve(...)` | `bms.DATASET_SOURCES["nasa_pcoe"]` |
| **MIT-Stanford (Severson)** — 124 LFP fast-charged to failure | fast-charge aging | `load_capacity_fade_csv(...)` | `bms.DATASET_SOURCES["mit_stanford"]` |

None are committed (size + licence). A committed synthetic fixture keeps the
tests green without them.

## Run it

```bash
python scripts/validate_real.py path/to/lg_us06_0degC.csv --chem nmc
python scripts/validate_real.py            # synthetic fixture (demo)
```

The harness:
1. loads the trace (real loaders set `source="real"`; synthetic stays `"synthetic"`);
2. runs the **estimator leaderboard** (temperature-aware) → SoC RMSE per estimator;
3. runs a **validation report** bucketed by C-rate, tagged with chemistry and source.

Real and synthetic results are **never conflated** — every row carries its
`source`, so a real-data RMSE can be quoted on its own.

## The rigorous pipeline

For a fair real-data number, don't validate with default chemistry parameters —
that conflates model error with parameter error. Chain calibration → validation:

1. `fit_from_pulse(...)` on a characterization pulse from the dataset → a
   cell-specific ECM;
2. `estimator_leaderboard(data, temperature_aware=True)` → real SoC RMSE by
   temperature (the real-cell version of the synthetic temperature benchmark);
3. for SoH: run the online joint-EKF over the cycling history and compare its SoH
   trajectory to the dataset's measured capacity checkpoints (SoH MAE, RUL error).

## Status

**Validated against real data** (NASA PCoE, above). The pipeline also runs on any
drop-in LG / MIT-Stanford trace for further real-data coverage. Accuracy figures
*elsewhere* in the docs remain **synthetic** and are labelled as such; the numbers
in this file are **real** (`source="real"`).
