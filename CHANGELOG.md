# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

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
