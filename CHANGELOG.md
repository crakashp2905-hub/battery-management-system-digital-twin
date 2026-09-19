# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [0.54.0] - 2026-09-19

**Universal auto-calibration — accurate on any cell, any chemistry.** The NASA
validation showed a generic model is only so accurate on a specific cell; the fix
is to calibrate to the cell automatically. (Deliberately *not* 100 thin algorithm
wrappers — the genuinely useful subset, wired into one pipeline that works.)

### Added
- **`bms/autocal.py`** — `auto_calibrate(current, voltage, dt)` identifies a
  calibrated cell model from a raw trace of **unknown chemistry**: estimates
  capacity, then tries every shipped OCV template, fits the ECM
  `(R0, R1, C1, R2, C2)` by nonlinear least squares (Levenberg-Marquardt) for
  each, and picks the lowest-residual template. Returns a `CalibratedCell` that
  spins up any estimator ready-tuned (`cal.make_estimator("ekf")`). Matching to
  the closest well-behaved template + fitting side-steps the ill-posed direct
  OCV inversion (which fails by ~60 % SoC in testing). **Validated on real NASA
  data:** with no chemistry hint it auto-selects NCA/NMC and drops the EKF from
  the naive ~12.5 % to **3.7 %** SoC RMSE (≈1.5 % on held-out cycles).
- **`bms/signal.py`** — robust signal conditioning for raw BMS data: `hampel`
  (median/MAD despiking), `median_filter`, `savitzky_golay` (peak-preserving
  smoothing, safe before DVA/ICA), `ewma`, `moving_average`, and a `clean_signal`
  pipeline (Hampel despike → gentle SG) reporting how many spikes it removed.
- **`docs/calibration.md`** (in the mkdocs nav) documents both, with the real-data
  numbers and an honest note on remaining OCV-template error.
- 12 tests (now **507**): auto-cal identifies the right template, reconstructs
  voltage, tracks SoC, estimates capacity, and — on the committed **real** NASA
  fixture — beats the naive estimator (<5 %); signal cleaners despike, smooth
  without distorting shape, and preserve clean signals.

## [0.53.0] - 2026-09-19

**Validated against real data — NASA PCoE (public domain).** The twin's accuracy
figures were synthetic; it is now run against real cells from the NASA Ames
Li-ion battery aging dataset, out of the box (no fit to the data).

### Added
- **`scripts/validate_nasa.py`** — fetches the public-domain NASA `.mat` files
  and runs two real-data validations, tagged `source="real"`:
  - **SoC estimators** on a real B0005 discharge (coulomb-counted SoC as truth):
    **coulomb 0.64 %**, ukf 12.31 %, ekf 12.55 %, joint_ekf 13.99 % RMSE. Coulomb
    counting validates on real data; the voltage filters are worse for an honest
    reason — the generic NMC OCV is not this cell's LiCoO₂ curve, which is exactly
    what `bms.calibration` / the self-calibrating twin address.
  - **State of health** across four cells (B0005/6/7/18): real capacity fade from
    ~93–102 % to 59–72 %, crossing 80 %-of-rated EOL at cycles 45–86.
- **`data/samples/nasa_b0005_discharge_real.csv`** — a committed ~20 KB slice of
  real B0005 (with NASA attribution) so CI exercises a real-data result without
  the multi-MB `.mat` files.
- 4 real-data tests (`tests/test_validation_real.py`, now **495**): the fixture
  loads and is tagged real; coulomb-counting is accurate (<2 %) on real data; the
  voltage filters are worse (the calibration finding); the loader skips headers.

### Changed
- `load_drivecycle_csv` now skips `#` comment lines, so a dataset can carry an
  attribution/licence header (real public datasets usually do) without breaking.
- `docs/validation.md` records the real NASA numbers and marks the twin
  **validated against real data**.

## [0.52.0] - 2026-09-19

Moderate-novelty tier — **replay engine + knowledge graph** (completes the tier).

### Added
- **`bms/replay.py`** — `ReplayEngine` stores a `DriveTrace` once and **replays**
  it through any number of twin configurations (SoC estimators, etc.), scoring
  each with the same metrics (SoC RMSE, final-SoC error, runtime) so versions
  compare apples to apples. `compare({...})` returns a ranked leaderboard — the
  reproducible A/B bench a twin needs before shipping a change.
- **`bms/knowledge_graph.py`** — `BatteryKnowledgeGraph`, a typed directed graph
  of the system (`Battery → Module → Cell → Sensor`, plus `Fault`/`Degradation`
  events) with named relations, so a diagnosis can ask structured questions —
  *"which cells contributed to fault 42?"* (`contributors_to`), *"what module is
  cell 17 in?"* (`ancestors`), *"what sensors are on this cell?"* (`neighbors`).
  Dependency-light (adjacency list, no networkx); renders to Mermaid;
  `build_pack_graph` seeds the Battery→Module→Cell hierarchy from a `BatteryPack`.
- 9 tests (now **491**): replay scores a config and returns a reproducible ranked
  leaderboard, runs without ground truth; graph typed queries, contributor/
  ancestry traversal, bad-edge rejection, Mermaid/summary, pack-graph hierarchy.

**This completes the moderate-novelty (🟠) tier** (the already-present items —
virtual sensors, sensor FDI, second-life, living passport — plus these new ones:
cell-level pack twin, fleet twins, probabilistic prognostics, physics+ML hybrid,
replay engine, knowledge graph).

## [0.51.0] - 2026-09-19

Moderate-novelty tier — **3 of 3: physics + ML hybrid model** (learn only the
residual).

### Added
- **`bms/hybrid_model.py`** — `HybridResidualModel`: keeps the ECM and learns
  only its **residual**, `V = V_ECM(SoC, I, T) + f_θ(I, SoC, |I|, I², T)`. The
  regressor (a small L2-regularised scaled MLP by default; any scikit-learn
  regressor accepted) is trained on `V_measured − V_ECM`, so it only has to
  capture the higher-order effects the 2-RC model misses. On a cell carrying a
  smooth residual the ECM cannot fit, it cuts voltage RMSE **~8 mV → ~1 mV
  (86 %)** and — crucially — **generalises to an independent trace in the same
  regime**, not just the training data. This is the physics-informed /
  neural-ODE-residual principle implemented dependency-light on scikit-learn (a
  core dep), and honest about being a residual learner, not a from-scratch
  electrochemical net; `score()` always reports pure-physics vs hybrid RMSE.
- 4 tests (now **482**): unfitted hybrid is pure physics; beats physics in-sample
  (>40 %); generalises to a holdout trace (>30 %); accepts a custom regressor.

## [0.50.0] - 2026-09-19

Moderate-novelty tier — **2 of 3: probabilistic prognostics** (probabilities and
lead time, not thresholds).

### Added
- **`bms/prognostics.py`** — three probabilistic read-outs:
  - `thermal_runaway_probability` — `P(runaway < t)` over horizons (30 s / 5 min /
    30 min) from temperature and its trend, with an **uncertain onset** (Gaussian
    crossing model) that imbalance and over-pressure lower (a stressed cell lets
    go sooner). Replaces the deterministic threshold with a probability curve.
  - `rul_distribution` — remaining useful life as a **distribution** (mean, median,
    5–95 % interval) by propagating the uncertainty in the observed fade rate, so
    "600 cycles" becomes "≈495 cycles (322–765)".
  - `predict_failure` — precursor-fusion failure probability (logistic of weighted
    early-warning evidence: pressure rise, resistance jump, voltage divergence,
    temperature spread, gas) with an estimated **lead time** that shortens as the
    dominant precursor's severity rises.
- 9 tests (now **478**): runaway probability monotone in horizon, low when
  cooling, raised by imbalance/pressure; RUL interval brackets the point estimate
  and widens with fade uncertainty, zero at EOL; more/stronger precursors raise
  failure probability and shorten lead time.

## [0.49.0] - 2026-09-19

Moderate-novelty tier — **1 of 3: scaling the twin (cell-level pack + fleet).**

### Added
- **`bms/pack_twin.py`** — `PackTwin` maintains a state **per individual cell**
  across the series/parallel stack and surfaces the two cells that matter: the
  **limiting cell** (lowest SoC — bounds usable capacity now) and the **weakest
  cell** (lowest capacity — the most-aged link driving pack EOL), with per-cell
  SoH/R0/voltage and deviation-from-mean. So a diagnosis can say *"cell 17 is the
  limiting cell"* instead of *"pack SoH = 82 %"*; `pack_soh` is the weakest link,
  not the average.
- **`bms/fleet.py`** — `FleetTwin` aggregates per-vehicle `VehicleState` roll-ups
  into a `FleetInsight`: population SoH stats (mean/spread/**bottom-decile**), an
  **at-risk list** (low SoH outliers, below-floor, or top-decile per-cycle fade),
  the fastest degraders, and a fast/normal/slow degradation **clustering** — the
  fleet-analytics layer that pairs with `digital_thread` and `reliability`.
- 11 tests (now **469**): per-cell coverage; limiting = lowest SoC; weakest =
  lowest capacity; imbalance/deviation consistency; fleet flags a low outlier;
  fastest degrader ranked first; clusters label every vehicle; empty fleet safe.

## [0.48.0] - 2026-09-19

High-novelty follow-ups — **2 of 2: Manufacturing → Field Digital Thread.**
This completes the review's ⭐⭐⭐⭐⭐ list (the other seven shipped in 0.42–0.47).

### Added
- **`bms/digital_thread.py`** — threads a cell's factory **birth certificate**
  (`ManufacturingRecord`: formation coulombic efficiency, capacity grade,
  end-of-line `R0`, self-discharge grade) forward into a **predicted field
  degradation trajectory**, so early-failure cells can be flagged *before*
  deployment. `link_formation_to_aging` maps the metrics to physically-motivated
  aging-rate multipliers (low formation efficiency → faster LLI/capacity fade;
  high end-of-line `R0` → faster resistance growth); `project_field_trajectory`
  turns a record + `FieldUsage` into cycles/days-to-EOL. Reference: a well-formed
  cell (CE 0.93) projects **2058 cycles** to EOL vs a poorly-formed one
  (CE 0.85, high `R0`, low grade) at **1258** — the factory data forecasts a
  ~40 % shorter life. `DigitalThread` ranks a batch by field risk and reports its
  life distribution (**B10**, worst cell), linking the factory to
  `bms.reliability`; `formation_life_correlation` shows formation efficiency
  predicts life (r > 0.8).
- 6 tests (now **458**): poor formation raises the multipliers and shortens life;
  the batch rank surfaces the worst cell first; the life distribution reports B10;
  formation efficiency positively correlates with projected life; records/
  projections serialise.

## [0.47.0] - 2026-09-19

High-novelty follow-ups — **1 of 2: Degradation-Mode Inference** (why a cell
ages, not just how much).

### Added
- **`bms/degradation_inference.py`** — turns the LLI/LAM incremental-capacity
  split (`degradation_modes.py`) into a full diagnosis. `infer_degradation_modes`
  fuses **LLI / LAM** capacity fade with **resistance growth** (`R0` now vs BOL)
  into one normalised attribution — e.g. *"SoH 72 %: LLI 37 %, LAM 13 %,
  resistance 50 %"* — and `infer_mechanisms` reasons from an `OperatingHistory`
  (mean SoC, temperature, calendar time, cycles, fast/cold-charge events) to the
  **likely physical mechanism**: calendar/SEI growth at high SoC, high-temperature
  cycling, lithium plating from cold/fast charging, or high-rate stress — scored
  deterministically and boosted for consistency with the observed dominant mode.
  Rule-based and fully explainable (physics in code, not an LLM). `narrative`
  gives the one-line verdict; `DegradationInference` wraps the BOL reference.
  Falls back to a capacity-vs-resistance two-mode split when no IC curve is given.
- 7 tests (now **452**): LLI-dominant IC → LLI largest; resistance growth its own
  mode; two-mode fallback; calendar/high-SoC history → SEI; cold/fast charging →
  plating; high-rate history → rate stress; wrapper + serialisation.

## [0.46.0] - 2026-09-18

Autonomous self-learning twin — **wave 4 of 4: the closed loop (capstone).**

### Added
- **`bms/autonomy.py`** — `AutonomousBatteryTwin`, the closed learning loop that
  ties waves 1–3 together into the system the strategy review pointed at: *a twin
  that knows what it doesn't know and decides how to learn.* Each round it (1)
  targets the parameter still worst-pinned by its Cramér–Rao bounds — dropping
  one once targeting stops helping, so it never chases the intrinsically hard
  slow-`R2` branch forever; (2) asks the experiment designer for the safest probe
  that best reduces that uncertainty; (3) performs it on the real cell (a hidden
  ground-truth plant); (4) self-calibrates, gated by **observability** (which
  parameters the probe could identify) *and* a **held-out validation check** that
  commits an update only if it does not worsen fidelity — making the loop
  **monotone non-increasing in error; it cannot diverge**.
  - Reference: a twin starting from fresh-cell parameters, shown a hidden aged
    cell (`R0` +120 %, `Q` −12 %), drives held-out voltage RMSE from ~19 mV to
    the **~0.7 mV sensor floor** and parameter error from 0.39 to 0.13 within a
    few rounds *without being told what was wrong*, recovering `R0` onto truth.
  - `degradation_report()` attributes what it learned to modes and correctly
    names **resistance growth** as dominant (degradation-mode inference).
- **`docs/autonomy.md`** documents the loop, the research question, and — as the
  review stressed — the honest prior-art caveat on any novelty claim. Added to
  the mkdocs nav (`build --strict` clean).
- 7 tests (now **445**): loop improves fidelity over rounds; validation gating is
  monotone non-increasing; recovers the identifiable parameter; targets its own
  uncertainty; infers the dominant degradation mode; round record serialises;
  runs without validation.

**This completes the 4-wave autonomous-self-learning-twin program** (0.43 → 0.46):
observability → experiment design → self-calibration → counterfactual, closed
into one autonomous loop.

## [0.45.0] - 2026-09-18

Autonomous self-learning twin — **wave 3 of 4: the Counterfactual Simulator.**

### Added
- **`bms/counterfactual.py`** — the **Counterfactual Battery Twin**: forks the
  current state and runs an alternate operating policy forward through the same
  physics to answer *"what would have happened if…?"*.
  - `counterfactual_charge` compares CC-CV charge policies at several C-rates on
    the shared plant and prices each into charge time, peak temperature, plating
    margin and — via `AgingModel` — the capacity fade and resistance growth it
    would cost. Reference: charging a 2.3 Ah cell 0.2→0.9 SoC costs **1C: 2529 s,
    +0.4 A plating margin, 0.006 % fade** vs **3C: 1247 s, −2.8 A margin
    (plating), 0.018 % fade** — the speed-vs-health trade made explicit
    (`fastest="3C"`, `gentlest="1C"`).
  - `alternate_history` replays a real mission under a modifier (scaled current,
    shifted temperature) and compares the degradation of the two histories —
    "what if I had driven twice as hard?" ages the pack measurably more; "what if
    it had run 15 °C cooler?" ages it less.
  - `CounterfactualTwin` wraps a cell model, charger limits and aging model.
- 6 tests (now **438**): faster charge costs more health; fastest/gentlest are
  the expected extremes; harder driving ages more; cooler operation ages less.

## [0.44.0] - 2026-09-18

Autonomous self-learning twin — **wave 2 of 4: the learning half** (decide what
to do, then update from it).

### Added
- **`bms/experiment_design.py`** — the **Autonomous Experiment Designer**. Given
  the twin's state it scores a library of candidate excitations (rest, CC pulses
  at several C-rates and both signs, an HPPC sequence, a randomised ±2C
  multipulse) by the optimal-design criteria from each candidate's Fisher
  information — **D-optimal** (`log det FIM`, most total information) or
  **targeted** (minimise one parameter's CRLB) — and returns the best. Every
  candidate first passes a **safety filter** that simulates it and rejects any
  that would leave the chemistry's voltage window or exceed a current limit, so
  the proposal is always safe *from the current state*: at 3 % SoC every
  discharge is rejected and a charge pulse wins. In the reference bench the rich
  `multipulse` is correctly ranked most-informative and `rest` least.
- **`bms/self_calibration.py`** — the **observability-gated Self-Calibrating
  Twin**. When the model mismatches a measured trace it recalibrates — but only
  the parameters the observability engine says the trace can actually identify;
  the rest are reported in `skipped_unidentifiable` and left at their prior (you
  cannot fix what you cannot see). On a dynamic trace from an aged cell it
  recovers `R0` to 0.0551 Ω (truth 0.0550) and `Q` to 2.055 Ah (truth 2.050),
  dropping the residual from 55 mV to 2 mV; on a pure rest it refuses to touch
  `R0`/`R1`/`R2`/`Q` and adjusts only the identifiable `soc0`.
- 10 tests (now **432**): designer prefers informative excitation over rest,
  targets capacity, rejects unsafe discharges at low SoC and high-C under a
  current limit; calibrator needs none on a match, recovers aged parameters on a
  dynamic trace, and refuses unobservable parameters on a rest.

## [0.43.0] - 2026-09-18

Autonomous self-learning twin — **wave 1 of 4: the Observability Engine.**

The goal of this program (from the strategy review) is a twin that *knows what it
doesn't know and decides how to learn*: observability → experiment design →
self-calibration → counterfactual. This wave lays the rigorous foundation the
rest build on.

### Added
- **`bms/observability.py`** — the **Battery Observability Engine**. For the
  parameter set `θ = [soc0, R0, R1, R2, Q]` it computes the voltage
  **sensitivities** along a trajectory, forms the **Fisher Information Matrix**
  (`FIM = SᵀS / σ²`), and reports the **Cramér–Rao lower bound** per parameter —
  the tightest 1-σ *any* estimator could reach from that data. A resting cell
  correctly makes `R0`/`R1`/`R2`/`Q` unidentifiable (no current → no ohmic drop;
  no SoC movement → capacity invisible); a dynamic HPPC-like profile identifies
  all five. Scalar optimal-design criteria (`d_opt = log det FIM`,
  `a_opt = tr FIM⁻¹`, `e_opt = λ_min`) are exposed so the next wave's experiment
  designer can *choose* the excitation that makes the unobservable observable.
  `ObservabilityEngine` wraps a cell model; `analyze_observability`,
  `fisher_information`, `voltage_sensitivities` are the free functions.
- 6 tests (now **422**): dynamic trajectory identifies `R0`/`Q`; rest makes them
  unobservable (infinite CRLB); larger current carries more information; lower
  sensor noise tightens the CRLB; the report surfaces the worst-identified
  parameter; sensitivities are zero for `R0` at rest.

## [0.42.0] - 2026-09-18

The unifying layer — one authoritative, uncertainty-aware digital-twin state.

The repository had strong estimators (SoC EKF/UKF/PF, the joint SoC+capacity EKF
for SoH), a drift monitor (`TwinSync`), and an online resistance tracker
(`RLSIdentifier`), but they ran as independent algorithms. `BatteryDigitalTwin`
makes them **one continuously calibrated belief** about the cell.

### Added
- **`bms/twin.py`** — `BatteryDigitalTwin` + immutable `TwinState`. One
  `update(voltage, current, dt, temperature_C)` drives the joint EKF (the
  probabilistic SoC/SoH core, each with 1-σ uncertainty), `TwinSync` (the
  model-vs-measurement residual / drift flag), and RLS (online `R0`) in
  lock-step, and returns a snapshot with **95 % credible intervals** and a
  JSON-ready `to_dict()`.
- **Observability / excitation** — capacity (hence SoH) is only identifiable
  when SoC actually moves, so the twin reports `excitation` (recent SoC swing)
  and a `capacity_observable` flag instead of pretending SoH is always trustworthy.
- **Twin confidence** — a scalar in `[0, 1]` (plus per-state `soc_confidence` /
  `soh_confidence` and a breakdown) fusing data freshness, estimator
  uncertainty, residual stability and excitation. This is the difference between
  "SoH 82 %, confidence 0.94" and "SoH 82 %, confidence 0.41 — insufficient
  excitation for a reliable capacity estimate".
- **`docs/twin.md`** documents the central-state architecture and the confidence
  model.
- 7 tests (now **416**): matched twin is confident and observable; a mismatched
  (aged) cell trips drift and lowers mean confidence; cold-start and rested
  traces correctly report low confidence / unobservable capacity; credible
  intervals bracket and clip; online `R0` is tracked.

## [0.41.0] - 2026-09-17

Phase-2 wave F — prepare to ship: performance benchmark, docs site, packaging.

### Added
- **Performance benchmark** (`scripts/benchmark_perf.py`) — throughput
  (steps/sec) of the hot paths, deterministic and seeded, to catch regressions.
  Highlights: the dependency-light **control core runs ~63 k steps/s** (≈ 1300×
  the full supervisor), and Coulomb ≫ EKF ≈ PF > UKF, as expected.
- **Documentation site** — `mkdocs.yml` + `docs/index.md` (Material theme);
  `mkdocs build --strict` is clean. `docs` extra (`mkdocs`, `mkdocs-material`).
- **`RELEASE.md`** — the publish checklist. The package **builds a valid
  sdist + wheel** (`python -m build`) and passes `twine check`; the final PyPI
  upload is a step the maintainer runs with their own token.
- 5 tests (now **409**; the 3 pyproject-metadata tests use `tomllib`, so they
  skip on Python 3.10 where it is not stdlib).

## [0.40.0] - 2026-09-17

Phase-2 wave E — the firmware bridge: a portable control core + SIL.

### Added
- **`bms/control_core.py`** — the deterministic algorithms a real BMS runs each
  control cycle, written in **plain scalar arithmetic** (only `math`, fixed-size
  `CoreState`, a literal OCV lookup table, no dynamic allocation, no library
  calls in the hot path) so they **transliterate to embedded C**: a 1-state SoC
  EKF (RC as feed-forward), State-of-Power limits, and safety threshold checks as
  a fault bitmask with a contactor-open command. `core_step(...)` is the single
  control-cycle entry point.
- **Software-in-the-loop** — `run_sil(...)` drives the core with the twin's plant
  (its SIL oracle) and reports SoC tracking + raised flags. In the reference
  bench the C-portable core tracks the twin's SoC to **~0.003 RMSE** and its
  safety flags fire on injected over-temp / over-voltage / over-current.
- **`docs/firmware.md`** documents the twin→firmware path and the
  simulation-only vs firmware-portable boundary (foxBMS-style control core).
- 6 tests (now **404**).

## [0.39.0] - 2026-09-17

Phase-2 wave D2 — local fault explanations + second-life economics.

### Added
- **`explain_fault_prediction`** (`bms/interpret.py`) — a **per-prediction**
  (local) attribution for the fault detector, answering *why the model flagged
  this sample* (vs the global `feature_importances`): each feature's contribution
  is how much occluding it to a nominal baseline drops the predicted class
  probability. Dependency-free (no `shap`). Example: an over-charge call driven
  by `T_spread` / `v_spread` / `v_min`.
- **Second-life economics** (`bms/second_life.py`): `assess_second_life` makes the
  retire / **repurpose** / recycle call from SoH, with the **residual value** and
  projected **second-life years** (from the aging model at a gentle stationary
  duty); `levelized_cost_per_kWh` rolls a first+second-life cost of energy — a
  second life strictly lowers it (≈ 44 % in the reference case).
- 6 tests (now **398**).

## [0.38.0] - 2026-09-17

Phase-2 wave D1 — fleet reliability / warranty curves (uncertainty quantification).

### Added
- **`monte_carlo_life`** (`bms/reliability.py`) — Monte-Carlo SoH over a fleet
  with a scattered (lognormal) cycle-aging rate, run through `AgingModel`.
  Returns the **SoH band** (P5/P50/P95 per year), a **warranty curve**
  `P(SoH < EoL)` vs year, the **RUL distribution** (mean + P5–P95 confidence
  interval), and the **B10 life** (year 10 % of the fleet reaches EoL) — the
  numbers a warranty desk needs, from the twin's own aging model.
  Verified: hotter fleets reach B10 sooner (25 °C → 6.4 yr vs 42 °C → 3.8 yr) and
  more manufacturing scatter widens the SoH band.
- **`warranty_reserve`** — the fleet fraction expected to need replacement within
  a given warranty term (reads the warranty curve).
- 6 tests. Measured suite total is now **392 passing** (+1 skipped); the running
  per-wave counts in a few earlier entries drifted slightly above the measured
  figure — the badge is corrected to the true number here.

## [0.37.0] - 2026-09-17

Phase-2 wave C — electrochemical fidelity: a native Single Particle Model.

### Added
- **`SingleParticleModel`** (`bms/spm.py`) — a genuine (reduced) *electrochemical*
  cell model, the one fidelity level the ECM can't reach. Each electrode is a
  spherical particle in which lithium **diffuses** (Fick's law, conservative
  finite-volume solver in NumPy — no PyBaMM dependency); the terminal voltage
  comes from the **surface** stoichiometry through each electrode's OCP.
  - Captures what an ECM cannot: under load the particle **surface depletes
    faster than the bulk** (at 3 C the surface reaches SoC 0.31 while the bulk is
    0.52), voltage sags with rate, and **recovers on rest** as the gradient
    relaxes — real diffusion limitation / rate capability. Bulk SoC is conserved
    by coulomb counting.
- 5 tests (now **396**).

### Note
This is the SPM (the base of PyBaMM's model hierarchy), implemented natively so
it runs and is tested in CI without a heavyweight electrochemistry dependency; a
full DFN/PyBaMM coupling would be a separate optional extra.

## [0.36.0] - 2026-09-17

Phase-2 wave B2 — dynamic (Plett) OCV hysteresis.

### Added
- **`PlettHysteresis`** (`bms/hysteresis.py`) — a one-state dynamic hysteresis
  model, `h[k+1] = e^(−|γ·ΔAh|)·h[k] + (1−e^(−|γ·ΔAh|))·(−sign(I))`, with the
  hysteresis voltage `M·h`. Unlike the default *static* `M·sign(I)` term it moves
  between the charge and discharge OCV branches **gradually** (a brief current
  reversal does not fully flip it). **Opt-in** on `SecondOrderECM`
  (`hysteresis=PlettHysteresis(...)`) — enabling it replaces the static term so
  nothing is double-counted; default behaviour is unchanged.
- 6 tests (now **391**).

## [0.35.0] - 2026-09-17

Phase-2 wave B1 — deferred estimators: particle filter + EIS/DRT analysis.

### Added
- **Particle filter** (`ParticleFilterEstimator`, registered `"pf"`): a bootstrap
  SIR filter for SoC that makes **no Gaussian assumption**, so it copes with the
  flat, non-linear OCV of LFP-class cells where an EKF linearisation is weakest —
  **0.49 % vs the EKF's 1.79 % SoC RMSE on synthetic LFP** (3.7× better), and
  competitive on NMC.
- **EIS analysis** (`bms/eis_analysis.py`): `compute_drt` deconvolves an EIS
  spectrum into a **Distribution of Relaxation Times** (Tikhonov-smoothed,
  non-negative) — recovering the cell's characteristic time constants;
  `eis_resistances` extracts the ohmic `R0` and charge-transfer `R_ct`; and
  `eis_soh` turns their growth into an **impedance-based SoH** (a power-fade
  signal capacity alone misses).
- 6 tests (now **385**).

## [0.34.0] - 2026-09-16

Phase-2 wave A — dashboard resync (make the new modules visible).

### Changed
- **New "🧪 Advanced" dashboard tab** (`app/streamlit_app.py`,
  `render_advanced`) surfacing the seven capabilities added since the last
  dashboard update, which were previously invisible to a user: **MPC vs CC-CV
  charging**, **grid-storage** peak-shaving + optimal storage SoC, online
  **twin-sync drift** detection, **LLI/LAM degradation-mode** diagnosis, the
  **State-of-Safety** index, the **analog front-end** (estimator-ranking shift),
  and **thermal-runaway propagation** — each an interactive panel. Verified
  end-to-end in-browser (no server errors). README dashboard section updated to
  five modes. No library code changed.

## [0.33.0] - 2026-09-16

Delivery wave 6e — live API service + container (the final stretch item).

### Added
- **FastAPI service** (`bms/api.py`): `create_app()` exposes the twin as a live
  endpoint backed by a stateful `TwinSync` — `POST /step` assimilates a device's
  measured (voltage, current, temperature) and returns the SoC + model-drift
  flag; `POST /safety` returns the State-of-Safety; plus `/state`, `/reset`,
  `/health`, `/info`, and auto-generated OpenAPI docs at `/docs`. Kept out of
  `bms/__init__` so the core library never requires FastAPI; install with
  `pip install '.[api]'`.
- **`Dockerfile`** — containerises the service
  (`uvicorn --factory bms.api:create_app`), so `docker run -p 8000:8000 bms-twin`
  serves the twin.
- `api` extra (fastapi, uvicorn); `fastapi`/`httpx` added to `dev` so CI exercises
  the API tests.
- 6 tests (now **379**).

### Completed
This finishes the full "add everything" program (waves 1–6), including the
FastAPI/Docker deployment wrapper.

## [0.32.0] - 2026-09-14

Delivery wave 6d — CAN FD frames + UDS (ISO 14229) diagnostics.

### Added
- **CAN FD** (`bms/can.py`): `CANFDFrame` supports the full CAN FD DLC set
  (0–8, 12/16/20/24/32/48/64 bytes) with validation, plus `CANFDFrame.pad_to_dlc`
  to round a payload up to a legal length — the richer frames classic CAN's
  8 bytes cannot hold.
- **UDS diagnostic server** (`bms/uds.py`): a minimal in-memory `UDSServer`
  answering the services a battery ECU uses — `0x22` ReadDataByIdentifier (live
  pack V / SoC / SoH / temperature), `0x19`/`0x14` read & clear **DTCs** (the
  twin's fault labels map to 3-byte DTCs via `FAULT_TO_DTC`), `0x10` session
  control, `0x3E` tester-present — with proper positive (`SID+0x40`) and negative
  (`0x7F`) responses.
- 19 tests (now **373**).

## [0.31.0] - 2026-09-14

Delivery wave 6c — one-command HTML health & validation report.

### Added
- **`build_health_report` / `save_report`** (`bms/report.py`,
  `scripts/generate_report.py`) — run the twin's key analyses and render a
  **self-contained HTML** report (inline CSS, no external assets): estimator
  leaderboard, safety case (FMEA top risks + fault-tree top-event probability),
  MPC-vs-CC-CV charging comparison, and a degradation-mode example, with KPI
  tiles. `python scripts/generate_report.py <chem> <out.html>`.
- 3 tests (now **354**).

## [0.30.0] - 2026-09-14

Realism wave 6b — analog front-end (AFE) measurement model.

### Added
- **`AFE` / `AFEConfig`** (`bms/afe.py`) — the measurement chain a real BMS reads
  through: ADC **quantisation**, **gain/offset** error, **thermal noise**, and a
  bandwidth-limited (and offset) **current sensor**. `AFE.apply_to_drivecycle`
  passes a clean trace through the chain so estimators can be scored against
  realistic measurements.
- Result: through a realistic AFE the estimator **ranking flips** — with an
  ideal current the Coulomb counter is unbeatable (0.00 % on the synthetic
  fixture), but under a real current sensor (offset + bandwidth + quantisation)
  it degrades to ~0.6 % and a voltage-feedback filter (UKF/EKF) wins. A genuinely
  novel, honest result: what "best estimator" means depends on the front-end.
- 6 tests (now **351**).

## [0.29.0] - 2026-09-14

Domains wave 6a — stationary / grid energy storage (a new application domain
beside the EV range predictor).

### Added
- **Stationary storage dispatch** (`bms/grid_storage.py`): `StationaryStorage`
  (energy-level battery with power/energy limits and round-trip efficiency),
  duty-cycle generators `peak_shaving_dispatch` and `arbitrage_schedule`, and
  `simulate_dispatch` (SoC/power trajectories, energy served/unmet, throughput,
  EFC). With `degradation_aware=True` + an `AgingModel` it prices each discharge
  into fade and skips cycles not worth the wear.
- **`optimal_storage_soc(days, temperature_C)`** — the SoC that minimises
  **calendar** fade over a horizon (for a battery that mostly sits): sweeps the
  calendar-aging model and returns the best SoC and fade at each. Lower/mid SoC
  and cooler storage fade least, as expected.
- 7 tests (now **345**).

## [0.28.0] - 2026-09-14

New-capability wave 5c — online data assimilation (simulator → **digital twin**).

### Added
- **`TwinSync`** (`bms/twin_sync.py`) — continuously corrects the ECM plant
  against a live (current, voltage, temperature) stream: it runs the plant
  forward on the measured current, then nudges the plant's SoC toward the
  measured terminal voltage with a Luenberger correction, so the twin *tracks*
  the physical cell instead of drifting open-loop. It also watches the
  **model-vs-measurement residual under load** and raises a **`drift`** flag when
  the residual grows — the twin knowing its model no longer matches reality
  (resistance growth, OCV shift) and needs recalibration.
  - Verified: a matched twin tracks truth to <1 % SoC and never flags drift;
    it pulls a 15 %-wrong initial SoC back toward truth; and an aged cell (R0 >2×)
    fed to a fresh-model twin is correctly flagged as drifted. `correction_gain=0`
    reduces it to open-loop Coulomb counting.
- 5 tests (now **338**).

## [0.27.0] - 2026-09-14

New-capability wave 5b — degradation-mode diagnosis (LLI vs LAM).

### Added
- **Degradation-mode diagnosis** (`bms/degradation_modes.py`) — decompose
  capacity fade into **loss of lithium inventory (LLI)** vs **loss of active
  material (LAM)** from incremental-capacity (dQ/dV) curves (Dubarry-style).
  LAM lowers the IC **peak height**; LLI narrows the accessible window so the IC
  **area** shrinks at fixed height — an invertible two-mode split:
  `LAM = 1 − height_ratio`, `LLI = 1 − area_ratio/height_ratio`.
  `diagnose_degradation_modes(v, ic_fresh, ic_aged)` recovers both (and the
  `dominant_mode`); `synthetic_degraded_ic(lli, lam)` builds a known fresh/aged
  pair, and the diagnosis recovers the injected fractions to <2 %.
- 7 tests (now **333**).

## [0.26.0] - 2026-09-13

New-capability wave 5a — model-predictive fast charging.

### Added
- **`MPCCharger` — optimal / model-predictive charging** (`bms/charge_control.py`).
  Each step it picks the largest charge current that keeps the next state inside
  **all** limits at once — terminal voltage, cell temperature, and the
  lithium-**plating** cap `plating_c_limit(T, SoC)·Q` — driving SoC to a target
  as fast as the physics allows (closed-form per-constraint current limits over
  the twin's ECM + lumped thermal plant). `cccv_charge()` runs CC-CV on the same
  plant and `compare_charging()` reports both.
- Result: on a 2.3 Ah NMC cell, MPC reaches 80 % SoC **~43 % faster than CC-CV**
  (1236 s vs 2160 s) while riding the plating limit exactly (margin ≥ 0, never
  plating) and staying well under the temperature ceiling.
- 5 tests (now **326**).

## [0.25.0] - 2026-09-09

Dependability wave 4c — property-based invariants + real-data harness.

### Added
- **Property-based invariant tests** (`tests/test_properties.py`, `hypothesis`):
  charge conservation (Coulomb integrates exactly), OCV monotonicity in SoC,
  non-negative irreversible heat with the ohmic floor, State-of-Safety bounded to
  [0, 1], recursive-estimator SoC in [0, 1], and CAN checksum ∈ [0, 255] — laws
  that must hold for *any* input. `hypothesis` added to the `dev` extra.
- **Real-data validation harness** (`scripts/validate_real.py`) — runs the
  calibration → leaderboard → bucketed validation-report pipeline on any
  drive-cycle CSV, keeping real vs synthetic results tagged; falls back to the
  synthetic fixture as a demo. `docs/validation.md` documents the datasets
  (LG / NASA / MIT-Stanford) and the rigorous calibrate-then-validate pipeline.
- 6 tests (now **321**).

### Note
Public datasets are large and licence-bound, so none are committed; the pipeline
runs the moment a licensed trace is dropped in, and every result is tagged with
its `source` so a real-data number is never conflated with a synthetic one.

## [0.24.0] - 2026-09-09

Dependability wave 4b — the safety case: FMEA → test traceability + Fault Tree.

### Added
- **Fault Tree Analysis** (`bms/fta.py`) — a tiny FTA engine (`Event`, AND/OR
  gates, `probability`, `minimal_cut_sets`, `basic_events`, `to_mermaid`) and a
  reference `thermal_runaway_tree()` whose basic events each name the twin
  detector that watches them (`faults` / `mechanics` / `sensor_fdi` / `hv_safety`
  / `propagation`). Internal short is a single-event cut; overcharge and
  overtemperature are AND gates.
- **FMEA → test traceability** — `FMEA_TEST_LINKS` maps each failure mode to its
  covering tests; `fmea_traceability()` adds `linked_tests` / `covered` / `gap`
  columns, and `scripts/traceability.py` prints the matrix and **exits non-zero
  if any RPN ≥ 100 mode has no covering test** (a CI-able safety gate).
- **`docs/safety_case.md`** ties FMEA + FTA + traceability into one argument.
- 8 tests (now **315**).

## [0.23.0] - 2026-09-09

Dependability wave 4a — harden the model-agnostic adapters.

### Added
- **`OODDetector` — out-of-distribution detection** for plugged-in ML models.
  `OODDetector.fit(X)` learns the training feature distribution; at inference
  `d² = (x−μ)ᵀ Σ⁻¹ (x−μ)` above a learnt (empirical-quantile) threshold flags an
  input the model never saw. Wired into `SklearnSocEstimator` / `OnnxSocEstimator`
  via `ood_detector=`; the estimate is still returned but `last_ood`/`n_ood` are
  set — feed that flag to the `ActionGate` so an OOD reading cannot drive an
  automated action.
- **`ModelCard`** — provenance + feature contract for a plugged-in model
  (`feature_names`/order, chemistry, training temp/SoC ranges, train/val RMSE,
  version, created date, sha256). Attach via `card=`.
- 5 tests (now **307**).

## [0.22.0] - 2026-09-09

Safety-depth wave 3b — the flagship module-level safety capability.

### Added
- **Thermal-runaway propagation** (`bms/propagation.py`) — `RunawayPropagation`
  models a module as lumped cells coupled to their neighbours and ambient
  (`C·dT/dt = Q_exo + Σk·ΔT − h·ΔT`). A triggered cell releases its exotherm over
  a window and heats neighbours; each crossing the onset temperature ignites and
  cascades. With realistic coupling a single-cell runaway propagates down the
  module (~13 s per cell in the reference case); a **thermal barrier**
  (low coupling / more spacing) arrests it. `propagation_arrested_below()` sweeps
  coupling to find the largest value at which propagation stops — a design aid
  for sizing spacing/barriers.
- 5 tests (now **302**).

## [0.21.0] - 2026-09-09

Safety-depth wave 3 of the "add everything" program — three real BMS safety
functions that live outside the cell models.

### Added
- **State-of-Safety index** (`bms/safety.py`) — `state_of_safety(result, …)`
  fuses temperature, temperature-spread, voltage excursion, pressure/gas, SoH,
  and imbalance into one `sos ∈ [0, 1]` (1 = safe) that degrades *before* any
  single threshold trips, with a per-signal `breakdown` and the `worst_signal`.
  `SoS = 1 − max(penalty)` (the worst signal governs — conservative for safety).
- **Insulation monitor (IMD)** (`bms/hv_safety.py`) — `InsulationMonitor`
  estimates HV-bus-to-chassis isolation resistance (`R_iso = V_pack / I_leak`)
  and alarms on the regulatory **Ω/V** criterion (warn/fault).
- **Contactor weld detection** — `ContactorWeldDetector`: after an open command,
  a healthy DC-link bleeds down; a link that stays near pack voltage past the
  settle time is flagged **welded**.
- **Sensor FDI** (`bms/sensor_fdi.py`) — `SensorMonitor` / `SensorFDI` detect
  per-channel **dropout / out-of-range / rate-spike / stuck** faults on V/I/T,
  and `virtual_cell_voltage` gives a model-based replacement (OCV − IR − V_RC)
  so a single sensor failure degrades gracefully instead of blinding the filter.

- 14 tests (now **297**).

## [0.20.0] - 2026-09-09

Estimation-frontier wave 2 of the "add everything" program.

### Added
- **`RLSIdentifier` — online ECM parameter identification** (`bms/online_id.py`).
  Recursive least squares tracks the ohmic resistance `R0` live from the terminal
  voltage's response to current steps (`ΔV ≈ −R0·ΔI`), so the twin keeps its ECM
  current under temperature/age drift without an offline re-fit — lighter than a
  dual-EKF. Recovers a known R0 to a few mΩ, tracks a change in R0, and coasts
  (does not wander) when the current is flat.
- **Entropic (reversible) heat** in the thermal model. Each chemistry gains an
  `entropic_coeff_V_per_K` (dU/dT); `ThermalModel.heat_generation` adds the
  reversible term `Q_rev = −I·T·(dU/dT)`, which — unlike I²R — **flips sign**
  between charge and discharge. Opt-in via `SupervisorConfig.entropic_heat`
  (off by default → no change to existing thermal behaviour).

### Note
Dynamic (Plett) hysteresis, also slated for wave 2, is deferred to a later wave
in favour of these net-new capabilities; the static per-chemistry hysteresis from
0.19.0 already delivers the flat-OCV accuracy gain.

- 7 tests (now **283**).

## [0.19.0] - 2026-09-09

Estimation-fidelity wave 1 of the "add everything" program — both items fix a
weakness the project's own benchmarks exposed.

### Added
- **`bias_ekf` — online current-sensor-bias recovery** (`BiasEKFEstimator`). Augments
  the EKF state with the sensor offset `b` (`x = [SoC, V_RC1, V_RC2, b]`, `i_true =
  i_meas − b`) and estimates it as a random walk. It keeps SoC accurate under a
  biased sensor **and recovers the offset** (`current_bias_A`, with 1σ) for
  recalibration — the online answer to the sensor-bias failure the benchmark
  highlighted. NMC +0.15 A → recovers +0.154 A at 0.14 % SoC RMSE (Coulomb 1.88 %).
  Registered as `bias_ekf`; `benchmark_estimators.py --bias`.

### Changed
- **OCV hysteresis is now modelled per chemistry by default** (fixes the flat-OCV
  weakness). Each chemistry carries a characteristic `hysteresis_v`
  (largest for LFP/LMFP); `OCVSOC.from_chemistry` applies it unless overridden.
  Mean-of-5-seeds EKF SoC RMSE improves **LFP +4.51 %, LMFP +2.12 %, LMO +1.91 %**,
  NMC/NCA/LTO +0.6 %. `estimator_leaderboard(..., hysteresis_aware=…)` toggles it;
  `benchmark_estimators.py --hysteresis`.
- `docs/estimation.md` gains **Hysteresis** and **Current-sensor bias** sections.

### Note
The static `±hysteresis_v·sign(I)` model already existed in `OCVSOC`; this wave
gives it physical per-chemistry values and proves the benefit. A smoother
**dynamic (Plett) one-state** hysteresis model is scheduled for wave 2.

- 5 tests (now **276**).

## [0.18.0] - 2026-09-09

### Added
- **Temperature dimension for the estimator benchmark.**
  - `synthetic_drivecycle(..., temperature_C=25.0)` now generates an isothermal
    trace at any temperature: the plant ECM follows its Arrhenius resistance
    shift and the OCV its temperature coefficient, so the voltage is genuinely
    cold/hot. Written into `DriveCycleData.temperature_C`. Default 25 °C
    reproduces prior behaviour exactly.
  - `estimator_leaderboard(..., temperature_aware=True)` feeds the trace
    temperature to every estimator whose `run` accepts it, so a cold/hot trace
    is scored with the correct model. `temperature_aware=False` scores a
    25 °C-assuming filter against the same trace.
  - `scripts/benchmark_estimators.py --temperature` reports the fair,
    apples-to-apples comparison: a temperature-**aware** filter stays flat across
    temperature (~0.1–0.3 % on NMC) while a temperature-**naive** one degrades to
    ~22 % RMSE at −15 °C. (Cross-temperature ranking is deliberately avoided — it
    is confounded by the shifting true SoC trajectory and operating point.)
  - `docs/estimation.md` gains a **Temperature** section with the results and a
    plain-English reading ("the takeaway is *feed the filter a cell
    temperature*", not "which filter").
- 3 tests (now **271**).

## [0.17.0] - 2026-09-09

### Added
- **Online joint SoC + SoH estimation in the supervisor.** With
  `SupervisorConfig.estimate_online=True`, `BMSSupervisor.step()` runs one
  `JointEKFSoH` each step and returns `soc_estimated`, `soc_sigma`,
  `soh_estimated`, `soh_sigma`, and `capacity_est_Ah` — **SoC and SoH from a
  single online filter**. When `online_feeds_soh` (default) and `soh_aware` are
  set, that SoH also drives the control-layer derating (one source of truth).
  Off by default → existing behaviour unchanged (fields are `nan`).
- **Estimator benchmark** (`scripts/benchmark_estimators.py`) — head-to-head SoC
  RMSE across all 7 chemistries × 3 conditions (nominal / current-sensor bias /
  voltage noise) via the built-in `estimator_leaderboard`, with a winner matrix.
- **`docs/estimation.md`** — documents exactly which model produces each quantity
  (SoC / SoH / SoE / health / RUL / thermal-runaway) and the benchmark results
  with an honest interpretation (e.g. EKF beats Coulomb ~6× under a biased
  current sensor on sloped-OCV chemistries; flat-OCV chemistries blunt all
  voltage-feedback filters).

### Changed
- **Dashboard now uses the online joint-EKF as the default estimator.** The SoC
  overlay (Live Signals) and a new live **Online SoH** panel (SoH & Aging tab)
  both come from that one filter, each with its ±1σ band.

### Fixed
- Dashboard crashed on every simulation run under NumPy < 2.0 (`np.trapezoid`
  was added in 2.0). Now uses the same `np.trapz` fallback shim as `bms/pack.py`.

- 3 tests (now **268**).

## [0.16.0] - 2026-09-08

### Added
- **Hardware realism — CAN robustness, a shipped DBC, and the pre-charge plant**
  (`bms/can.py`, `bms/control.py`), closing the gap between a state-machine and a
  bus a real ECU would trust:
  - **`bms.dbc`** — an actual Vector DBC shipped alongside the encoder. It is
    generated by `BMSCanBus.to_dbc()` and cross-checked against real `cantools`
    (byte-identical decode); a test fails if the committed file drifts from the
    encoder.
  - **Bus-health frame** (`0x186`, `BMS_Health`) added to every `broadcast()`: a
    rolling **alive counter** (stalled-transmitter detection), an 8-bit additive
    **checksum** over the status frame (`can_checksum`), and a **tx counter**.
  - **`CanBusMonitor`** — receiver-side integrity: message **freshness/timeout**,
    alive-counter **continuity**, checksum validation, and **bus-off** via the
    ISO 11898 error counter (+8 on error, −1 on success, bus-off at TEC ≥ 256)
    with `recover()`.
  - **`PrechargeCircuit`** — the physical DC-link plant: a pre-charge resistor
    charging the inverter bus capacitor through the exact RC transient
    `V(t)=V_pack(1−e^(−t/RC))`, tracking **peak inrush** (`V_pack/R`) and the
    **energy burned in the resistor** (→ its pulse rating). Optionally wired into
    the supervisor (`SupervisorConfig.simulate_precharge`) so the sequencer sees a
    realistic rising bus voltage instead of an externally supplied number.
- `can` extra (`cantools`) for validating/consuming the shipped DBC.
- 18 tests (now **266**).

## [0.15.0] - 2026-09-07

### Added
- **Real-data calibration** (`bms/calibration.py`) — the bridge from a research
  simulator to a data-calibrated twin:
  - `fit_from_pulse()` — identify ECM parameters (R0/R1/C1/R2/C2) from an HPPC /
    pulse or drive trace (recovers R0 to ~0.2 mΩ on a noisy synthetic pulse).
  - `fit_cell_distribution()` — learn per-parameter mean/std across cells and a
    data-driven `PackConfig` scatter, replacing fixed manufacturing scatter.
  - `validation_report()` — SoC-error metrics bucketed by C-rate or temperature,
    tagged with chemistry and **source (synthetic vs real)**.
- `DriveCycleData.source` (`"synthetic"` | `"real"`; CSV loaders set `"real"`) so
  real-data accuracy is never conflated with synthetic.
- 4 tests (now **248**).

## [0.14.1] - 2026-09-07

### Added
- **`OllamaLLM`** — a local, on-device Ollama model as an `llm` callable (`str→str`),
  dependency-free (urllib). Fits the privacy/redaction posture — no telemetry leaves
  the machine — and works with `DiagnosticAgent(llm=…)`, `explain_state(…, llm=…)`,
  and the LangGraph agent (LangChain's `ChatOllama` works through the same hook).
  Because agent actions are deterministic, even a small local model is safe: it only
  phrases the report.
- 2 tests (now **244**).

## [0.14.0] - 2026-09-06

### Added
- **Safety-first diagnostic agent** (`bms/agent.py`) — the agent *decides*, never actuates:
  - **Structured, validated actions** — `DiagnosisReport.proposed_actions` are
    `ProposedAction`s derived **deterministically** from findings, so an LLM can
    phrase the report but can never introduce or alter a control action. A prompt
    injection in telemetry cannot produce a command.
  - **`ActionGate`** — a deny-by-default approval boundary: actuating actions
    (contactor / charger / cloud) and any `requires_approval` action are refused
    unless an explicit approver returns True.
  - **`redact_telemetry()`** — strips identifier fields (serial / VIN / customer /
    location / device id …) before telemetry reaches a hosted LLM or cloud; the
    LLM only ever sees the controlled summary (no raw identifiers).
  - **`evaluation_scenarios()`** — reference fixtures (nominal / over-temperature /
    gas-precursor / low-SoH / conflicting-signals) with expected severity + signals.
- 6 tests incl. prompt-injection immunity, no-identifier-leak, deny-by-default,
  and the eval scenarios (now **242**; coverage 90%).

## [0.13.1] - 2026-09-06

### Changed
- **CI quality gates are now mandatory.** Ruff is blocking (no longer advisory),
  and a coverage job enforces **≥ 85%** line coverage (currently 90%) via
  `pytest-cov`. Added `pytest-cov` to the `[dev]` extra and `[tool.coverage]`
  config to `pyproject.toml`.

## [0.13.0] - 2026-09-06

### Added
- **Framework-agnostic estimator adapters** (`bms/estimation.py`):
  `SklearnSocEstimator`, `OnnxSocEstimator`, and `model_from_file()` — plug a
  trained scikit-learn / PyTorch / ONNX (or any) model in as a first-class
  `SocEstimator` (`.onnx` via the `[onnx]` extra; PyTorch works through
  `FunctionSocEstimator`).
- **LLM-agnostic narration** — `explain_state(..., llm=…)` and
  `explain_charge(..., llm=…)` accept any `str -> str` callable (a Claude/OpenAI
  SDK call, a LangChain model's `.invoke`, or a local model). The default stays
  the deterministic, zero-cost, no-hallucination template.
- **Optional LLM diagnostic-agent layer** (`bms/agent.py`, `[agent]` extra):
  - `DiagnosticAgent.diagnose()` — deterministic reasoning over faults, thermal,
    gas/pressure, and SoH → a severity + recommended action (or an LLM-written one).
  - `twin_tools()` exposes the twin's read-only functions as agent tools
    (`Tool` / `Finding` / `DiagnosisReport`).
  - Lazy integration points — `to_langchain_tools()` (LangChain),
    `build_langgraph_agent()` (a LangGraph diagnostic graph), and `traced()`
    (Langfuse observability) — imported only when used, so the core keeps no hard
    dependency on them. Default LLM provider is Claude; fully swappable.
- 14 tests incl. pickle-load safety (opt-in `trust_pickle` + `sha256`); now **236**
  (the ONNX round-trip skips when onnxruntime is absent).

## [0.12.0] - 2026-09-06

### Added
- **Real-dataset validation scaffold** (`bms/datasets.py`):
  - `DriveCycleData` + `synthetic_drivecycle()` (a physically-consistent fixture,
    with an optional current-sensor bias); `load_drivecycle_csv()` /
    `save_drivecycle_csv()` — real LG-18650 CSVs or the fixture, with column
    remapping and coulomb-counted SoC when ground truth is absent.
  - `estimator_leaderboard()` — runs every registered estimator on a trace and
    ranks them by SoC RMSE / MAE / max-error / runtime. Under a current bias the
    EKF/UKF beat the Coulomb counter, as expected.
  - SoH path: `load_capacity_fade_csv()`, `nasa_mat_to_capacity()` (parses the
    NASA PCoE `.mat`), and `soh_curve()` (SoH series + RUL extrapolation).
  - `DATASET_SOURCES` (LG-18650 / NASA PCoE / MIT-Stanford URLs + usage) and a
    committed synthetic sample under `data/samples/`.
- 7 tests (now **222** total). Raw datasets are not committed — drop them in and
  the loaders + leaderboard run unchanged.

## [0.11.0] - 2026-09-06

### Added
- **Mechanical / gas failure modes** (`bms/mechanics.py`) — the failure class that
  voltage/temperature-only BMS logic misses:
  - `PressureModel` couples cell temperature/SoC to internal **pressure, gas
    generation (Arrhenius, accelerated above an onset), swelling, H₂
    concentration, and safety-venting** (releasing gas + an exotherm fed back to
    the thermal model). Because gas and pressure rise before temperature, the
    pressure rule trips **~17 s before** the temperature-runaway rule on the
    reference heat ramp — quantified early warning.
  - `MechanicalFaultDetector` — rule detector for `GAS_VENTING`, `INTERNAL_SHORT`
    (microfracture, via coulombic efficiency < 1), and `SWELLING`.
  - `coulombic_efficiency()` helper (from passport Ah totals).
- Four new `FaultMode`s — `gas_venting`, `internal_short`, `swelling`,
  `electrolyte_leak` — with matching CAN telemetry codes.
- 7 tests including a **pressure-leads-temperature** lead-time test (now **215** total).

## [0.10.0] - 2026-09-06

### Added
- **SoH-aware control** — the supervisor uses state-of-health to protect an
  aging pack, closing the loop from the charging/plating physics and the
  aging/SoH estimate to control action:
  - `BMSSupervisor.set_soh(soh_capacity, soh_resistance)` feeds live SoH (from
    `JointEKFSoH` or `AgingModel`) into control; `step()` reports `soh_capacity`.
  - With `SupervisorConfig.soh_aware` on, current is derated piecewise-linearly
    as capacity SoH falls (full above 0.90, down to a floor fraction at/below
    0.70), and charge current is capped below the lithium-plating C-rate limit
    (evaluated at the coldest cell / highest SoC). Off by default (SoH = 1 →
    identical behaviour).
- `plating_c_limit()` promoted to a reusable module function shared by the
  charging model and the supervisor.
- **Dashboard:** new 🔋 *Charging & Aging* tab — an AC/DC & fast/slow comparison
  table, charge-time / efficiency / peak-temperature bars, a projected-SoH-over-
  cycles chart, and a plain-language `explain_charge` summary per method.
- 5 tests (now **208** total).

## [0.9.0] - 2026-09-06

### Added
- **Online state-of-health** (`bms/soh_estimator.py`): `JointEKFSoH`, a joint
  Extended Kalman Filter estimating SoC **and** usable capacity together
  (state = [SoC, V_RC1, V_RC2, Q]). It tracks capacity as the cell ages,
  exposing a live `soh = Q/Q_nominal` with 1-σ uncertainty, and converges from a
  beginning-of-life guess toward the true aged capacity within a cycle.
  Registered as `"joint_ekf"` in the estimator registry; pairs with the offline
  `AgingModel` (which *predicts* fade — this *estimates* it from data).
- **Interpretability layer** (`bms/interpret.py`):
  - `explain_state()` / `explain_charge()` — plain-language summaries of a
    supervisor step or a charge session.
  - `feature_importances()` — the fault detector's RandomForest importances
    mapped onto named features, so an ML alarm is explainable.
  - `estimator_agreement()` — spread, a disagreement flag, and an
    inverse-variance **fused** estimate across models.
  - `soc_report()` — SoC rendered with its ±kσ uncertainty band.
- 11 tests (now **203** total).

## [0.8.0] - 2026-09-06

### Added
- **Model-agnostic estimator framework** (`bms/estimation.py`):
  - `SocEstimator` (universal `name`/`reset`/`run`) and `RecursiveSocEstimator`
    (adds `update`/`soc`) structural protocols — the four built-in estimators
    satisfy them unchanged.
  - A registry — `make_soc_estimator(name, ...)`, `available_soc_estimators()`,
    `register_soc_estimator(name)` — so SoC algorithms are chosen by name/config
    instead of hard-wired classes.
  - `FunctionSocEstimator` — wrap **any** trained model (scikit-learn, PyTorch,
    an ONNX Runtime session, a lookup table) behind the standard interface.
  - `Estimate` + `soc_estimate()` — read SoC **with 1-σ uncertainty** (from the
    EKF/UKF covariance) where the model provides it.
- 8 contract tests every registered estimator must pass (now **195** total).

## [0.7.0] - 2026-09-06

### Added
- **Charging-method physics** (`bms/charging.py`): `ChargingModel`,
  `ChargeProtocol`, `ChargeMethod` (AC L1/L2, DC fast/ultra), `ChargeResult`,
  and `compare_methods()`. Quantifies effective C-rate, wall-to-battery
  efficiency, cell heating, charge time, and lithium-plating risk for any
  method. Headline (60 kWh pack, 20→90%): AC and 50 kW DC age the pack
  negligibly (~3300 charges to 80% SoH), while 250 kW ultra-rapid (~4C)
  reaches 80% in ~190 charges — roughly **17× faster wear**, driven by lithium
  plating and heat; cold ultra-charging is worse still.
- **Dynamic state-of-health** (`bms/aging.py`): `AgingModel`, `AgingState`,
  `AgingParams`. Capacity fade + resistance growth from C-rate, temperature,
  depth-of-discharge, plating, and a √-time calendar term; `apply_to_pack()`
  writes SoH back onto the cells (idempotent) so the twin finally **ages**, and
  `rul_cycles()` estimates remaining useful life. Closes the "no dynamic aging"
  gap.
- 8 regression tests (now **187** total).

## [0.6.1] - 2026-09-06

### Fixed
- **Thermal-runaway injection** — `FaultInjector.apply_to_temperatures` now
  models a runaway as a measured temperature that starts just above the onset
  threshold and climbs with dwell time, so injecting `THERMAL_RUNAWAY` from a
  normal temperature actually trips the detector (previously inert unless the
  cell was already above the threshold).
- **SOP default voltages** — `StateOfPower.calculate` evaluates terminal
  voltages at the supplied `pack_current_A` (loaded) instead of no-load when
  `cell_voltages_V` is omitted. Unchanged at `pack_current_A == 0`.
- Doc drift: documented the `UNDERVOLTAGE` mode, the `PRECHARGE` state, and the
  `contactor_state` / `precharge_elapsed_s` step keys; removed dead
  `typing.Callable` / `dataclasses.field` imports.

### Added
- `BMSSupervisor.state_of_power()` — SOP from the supervisor's own state.

## [0.6.0] - 2026-09-05

### Fixed
- **OCV hysteresis direction** — positive current is discharge throughout the
  model, so a non-zero hysteresis now lowers the discharge branch and raises
  the charge branch as physical cell measurements require.

### Added
- **State of Power** — `StateOfPower` provides conservative 2 s, 10 s, and
  30 s traction and regenerative-braking limits from terminal voltage,
  present current, temperature-adjusted 2-RC impedance, current caps, and a
  thermal derating envelope.
- **Classic CAN telemetry** — `BMSCanBus` broadcasts fixed 8-byte, 11-bit CAN
  2.0B status, cell-extrema, thermal, and multi-horizon SOP frames, with an
  engineering-unit parser compatible with Intel/little-endian DBC signals.
- **Pre-charge sequencing** — `BMSSupervisor` can now model an HV contactor
  open → pre-charge → closed sequence, gate current until the measured DC link
  reaches 95% of pack voltage, and fail safely on timeout.
- Seven focused regression tests for hysteresis, SOP, CAN telemetry, and
  pre-charge sequencing (**176** total).

## [0.5.0] - 2026-09-05

### Fixed
- **LSTM training reproducibility** — `LSTMEstimator.fit` shuffles with the
  instance's seeded RNG instead of the global NumPy state, so a fixed `seed`
  reproduces training exactly.
- **Heat generation** — corrected to `Q = |I·(OCV − V_t)|`, valid for both
  charge and discharge. The previous `I²R0 + I·max(OCV − V_t, 0)` double-counted
  the ohmic term on discharge and dropped all polarisation heat on charge.
- **Short-circuit detection** — added a current-based rule with trip authority.
  Previously short circuit had no rule and relied on the advisory ML layer,
  which cannot trip the contactor.
- **Supervisor fault evaluation** — uses the real per-cell currents (was
  hard-coded to zeros), and an optional `FaultInjector` is applied to the
  measured signals so injected faults actually reach the detector.
- **Peak power** — `peak_power_W` reports the deliverable power bounded by the
  min-voltage cutoff (consistent with `diagnostics.compute_crate_map`) instead
  of the unreachable matched-load `V_oc²/(4·R0)`.
- **Kalman filters** — SOC is no longer hard-clipped inside the EKF/UKF
  recursion (which biased the covariance); it is clipped only when reported.
- **Thermal solver** — the explicit-Euler step sub-steps internally to remain
  stable for any `dt` (CFL guard); `dt = 1 s` behaviour is unchanged.
- **NumPy compatibility** — `state_of_energy_Wh` falls back to `np.trapz` when
  `np.trapezoid` (NumPy ≥ 2.0) is unavailable.
- **OCV / filter consistency** — the EKF/UKF measurement models pass current
  into the OCV lookup so they match the plant when hysteresis is enabled.
- Documented the `soe_Wh` field of `BMSSupervisor.step`; removed dead
  hidden-state assignments in `LSTMEstimator.reset`.

### Added
- **Terminal SHUTDOWN + fault recovery** — thermal runaway latches to the
  terminal `SHUTDOWN` state (previously defined but unreachable), and
  `BMSSupervisor.clear_fault()` provides operator reset from the recoverable
  `FAULT` state.
- **Parallel-group circulating currents** — parallel cells share a common
  terminal node solved from KCL, so cells at different SOC exchange circulating
  currents instead of having their voltages averaged.
- **Battery Passport series capacity** — `nominal_capacity_Ah` uses the
  per-group cell capacity (`np.mean(capacities_Ah)`) instead of `capacities_Ah.sum()`,
  so equivalent full cycles (EFC) is physically accurate for series packs.
- **Switched-capacitor balancer directionality** — efficiency attenuation is
  consistently applied to the receiving cell regardless of transfer direction ($v_i > v_{i+1}$
  or $v_i < v_{i+1}$), eliminating unphysical charge creation.
- **Instant short-circuit trip** — short-circuit current spike trips contactor
  immediately to `FAULT` on step 0 without a 3-step debounce delay.
- **Cell under-voltage protection** — added `UNDERVOLTAGE` fault mode and rule
  guarding against over-discharge ($v_{cells} < v_{min}$ and $> v_{dropout}$).
- **Streamlit double-stepping resolved** — removed redundant manual `sup.pack.step`
  call and wired `FaultInjector` directly into `BMSSupervisor`.
- Packaging (`pyproject.toml`, pip-installable), GitHub Actions CI (test matrix
  on Python 3.10–3.13 + advisory ruff), `CONTRIBUTING.md`, and 14 regression
  tests (**169** total).

### Changed
- `heat_generation`, parallel-group physics, and `peak_power_W` change numeric
  outputs for multi-parallel packs and power reporting. Packs with one cell per
  series group at `dt = 1 s` are unaffected.

## [0.4.0]
- Seven chemistries, battery passport, DVA/ICA + EIS diagnostics, and the EV
  range predictor. (Baseline prior to this changelog.)
