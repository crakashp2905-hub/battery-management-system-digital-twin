# Validation against real data

The library's built-in accuracy numbers are **synthetic** (the plant ECM's own
SoC is the ground truth, so scoring is exact — see `docs/estimation.md`). That is
the right way to *compare* estimators, but a credibility claim needs a run
against **real cells**. The loaders and harness for that already exist; only the
data is missing, because the public datasets are large and licence-bound.

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

The pipeline is complete and runs today on any drop-in trace; what remains is to
place a licensed dataset and record the numbers here. Until then, treat the
accuracy figures elsewhere in the docs as **synthetic** and labelled as such.
