# BMS Digital Twin

A research-grade, **AI-augmented Battery Management System (BMS) digital twin** for multi-cell
Li-ion / solid-state packs. It unifies electrochemical modelling, thermal simulation, cell
balancing, **model-agnostic** state estimation (SoC **and** online SoH with uncertainty), hybrid
fault detection, charging-method physics, dynamic aging, SoH-aware control, electrochemical
diagnostics, an EV range predictor, and a plain-language interpretability layer — into one
reproducible, fully-tested framework where every module is independently usable.

<p>
  <img alt="version" src="https://img.shields.io/badge/version-0.23.0-blue">
  <img alt="CI" src="https://github.com/crakashp2905-hub/battery-management-system-digital-twin/actions/workflows/ci.yml/badge.svg">
  <img alt="coverage" src="https://img.shields.io/badge/coverage-90%25-brightgreen">
  <img alt="tests" src="https://img.shields.io/badge/tests-307%20passing-brightgreen">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%E2%80%933.13-blue">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="dashboard" src="https://img.shields.io/badge/dashboard-Streamlit-ff4b4b">
</p>

> **Status.** ✅ 307/307 unit tests pass • 32 library modules • 7 chemistries • ruff-clean •
> CI on Python 3.10–3.13 • Streamlit dashboard + EV range predictor + executed demo notebook.

---

## Table of contents

- [Highlights](#highlights)
- [Supported chemistries](#supported-chemistries)
- [Project structure](#project-structure)
- [Module-to-capability map](#module-to-capability-map)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Design highlights](#design-highlights)
- [Interactive dashboard](#interactive-dashboard)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [License](#license)

---

## Highlights

- **Seven cell chemistries** — NMC, LFP, LMFP, LTO, NCA, LMO, and a solid-state (SSB) model,
  each with its own OCV–SOC table, Arrhenius resistance scaling, voltage window, runaway
  temperature, and default ECM. Switch chemistry with a single argument.
- **Configurable pack topology** — arbitrary series × parallel (`SxP`) packs with per-cell
  manufacturing scatter and a shared-node parallel model (circulating currents between mismatched
  cells, not just averaged voltages).
- **Model-agnostic estimator framework** — `SocEstimator` / `RecursiveSocEstimator` protocols, a
  registry (`make_soc_estimator("ukf")`), and adapters to plug in any trained model
  (`SklearnSocEstimator`, `OnnxSocEstimator`, `FunctionSocEstimator`; `model_from_file` with opt-in
  pickle + checksum, ONNX as the safe default). `Estimate` values carry **1-σ uncertainty**.
- **Five SoC estimators + online SoH** — Coulomb counter, EKF, UKF, NumPy LSTM, and a **joint EKF
  that estimates SoC *and* capacity together** (`soh = Q/Q₀` with uncertainty).
- **Charging-method physics + dynamic aging** — quantify AC vs DC and fast vs slow charging into
  C-rate, efficiency, cell heating, lithium-plating risk, and capacity/resistance fade; the pack
  **actually ages** and feeds SoH/RUL.
- **SoH-aware control** — the supervisor derates current as the pack ages and caps charge current
  below the lithium-plating limit — closing the loop from charging physics + SoH estimate to
  control action.
- **Hybrid fault detection** — deterministic rule layer (sole trip authority; overcharge,
  undervoltage, short-circuit, thermal-runaway) OR-fused with an advisory Random-Forest layer.
- **Mechanical / gas fault detection** — internal pressure, swelling, gas venting, and
  microfracture internal shorts; pressure/gas cross their thresholds **~17 s before** the
  temperature-runaway rule in the reference ramp (early warning that temperature alone misses).
- **Intelligent supervisor** — IDLE → PRECHARGE → OPERATING → BALANCING → FAULT → SHUTDOWN state
  machine with HV pre-charge contactor sequencing, predictive cooling, and current- or power-mode
  loads.
- **State of Power + CAN telemetry** — multi-horizon (2 s / 10 s / 30 s) traction/regen limits and
  DBC-compatible classic-CAN broadcast frames.
- **Interpretability layer** — plain-language state/charge summaries (optionally LLM-narrated,
  provider-agnostic), named RandomForest feature importances, multi-estimator agreement +
  inverse-variance fusion, and SoC ±kσ bands.
- **Optional LLM diagnostic agent** — a *deterministic* diagnostic engine (fault / thermal / gas /
  SoH → severity + recommended action) with optional LLM phrasing; **read-only tools** and lazy
  LangChain / LangGraph / Langfuse integration behind the `[agent]` extra. Provider-agnostic
  (`llm=` any `str→str`), Claude by default — or a local **Ollama** model (`OllamaLLM`) for
  on-device / offline / no-telemetry-egress. **Safety-first:** structured actions are
  derived deterministically (prompt-injection-immune), gated **deny-by-default** (`ActionGate`),
  and telemetry is **redacted** before any hosted LLM sees it.
- **Diagnostics + EV range** — DVA/ICA fingerprints, simulated EIS (Nyquist), C-rate map, and a
  first-principles range predictor with weather/traffic/road coupling (India presets included).
- **Data-calibrated** — fit ECM parameters from HPPC/pulse or drive data (`fit_from_pulse`),
  learn per-cell parameter distributions instead of fixed scatter (`fit_cell_distribution`), and
  report accuracy bucketed by C-rate/temperature — kept separate for **synthetic vs real** data.
- **Reproducible & tested** — every randomness source is seeded; 307 unit tests; pip-installable
  with GitHub Actions CI (ruff-blocking + 90% coverage gate).

---

## Supported chemistries

| Code | Cell type | Nominal V | Window (V) | Runaway onset | Notable trait |
|------|-----------|-----------|------------|---------------|---------------|
| **NMC**  | LiNiMnCoO₂        | 3.7  | 3.0–4.2  | 70 °C  | High energy density, automotive/consumer |
| **LFP**  | LiFePO₄           | 3.2  | 2.5–3.65 | 90 °C  | Long cycle life, flat plateau, thermally stable |
| **LMFP** | LiMnFePO₄         | 3.7  | 2.8–4.15 | 85 °C  | Dual Fe/Mn plateau, higher voltage than LFP |
| **LTO**  | Li₄Ti₅O₁₂ anode   | 2.3  | 1.5–2.8  | 95 °C  | Ultra-safe zero-strain anode, very low R₀ |
| **NCA**  | LiNiCoAlO₂        | 3.65 | 3.0–4.2  | 65 °C  | Highest energy density, thermally sensitive |
| **LMO**  | LiMn₂O₄ spinel    | 3.8  | 3.0–4.2  | 55 °C  | Low cost, double plateau, higher self-discharge |
| **SSB**  | Solid-state       | 3.85 | 3.0–4.35 | 150 °C | Li-metal anode, highest safety, poor cold performance |

Each chemistry lives in `bms/chemistry.py` (`CHEMISTRY_PROPS`); request one with
`get_chemistry_props("lfp")` or via the `chemistry=` argument on `PackConfig`,
`HybridFaultDetector`, `OCVSOC.from_chemistry(...)`, `ECMParameters.for_lfp()`, etc.

---

## Project structure

```
battery-management-system-digital-twin/
├── bms/                       # Library (32 modules)
│   ├── chemistry.py           # 7 chemistries: OCV tables, Arrhenius, limits, defaults
│   ├── ocv_soc.py             # OCV–SOC characteristic (PCHIP interpolant, temp coefficient)
│   ├── ecm.py                 # 2-RC equivalent-circuit model, Arrhenius scaling, parameter ID
│   ├── pack.py                # Series × parallel pack, scatter, shared-node parallel currents
│   ├── thermal.py             # 1-D FDM thermal model (CFL-guarded) + PID / predictive cooling
│   ├── balancing.py           # Passive / switched-capacitor / inductor balancing + comparison
│   ├── soc_estimators.py      # Coulomb counter, EKF, UKF, NumPy LSTM + benchmark harness
│   ├── soh_estimator.py       # Joint EKF: online SoC + capacity (SoH) with uncertainty
│   ├── online_id.py           # RLS online R0 identification (tracks drift, no re-fit)
│   ├── estimation.py          # Model-agnostic protocols + registry + BYO adapter + Estimate
│   ├── faults.py              # Fault injection + hybrid rule/ML detector + feature buffer
│   ├── mechanics.py           # Pressure / gas / swelling, venting, internal-short detection
│   ├── _train_detector.py     # Synthetic labelled-data generator for the ML detector
│   ├── fmea.py                # FMEA / RPN table + RUL estimator (capacity & resistance fade)
│   ├── charging.py            # AC/DC, fast/slow charging physics + plating limit
│   ├── aging.py               # Dynamic capacity fade + resistance growth; SoH feedback to pack
│   ├── control.py             # Supervisor FSM + precharge (RC DC-link plant) + SoH-aware control
│   ├── safety.py              # State-of-Safety index (fuses T/V/gas/SoH/imbalance → 0–1)
│   ├── hv_safety.py           # Insulation monitor (IMD) + contactor weld detection
│   ├── sensor_fdi.py          # Sensor fault detect/isolate (V/I/T) + virtual sensor
│   ├── propagation.py         # Cell-to-cell thermal-runaway propagation (cascade + barriers)
│   ├── passport.py            # Battery passport — lifetime EFC / DWC / RTE / throughput
│   ├── sop.py                 # State of Power — multi-horizon traction/regen limits
│   ├── can.py                 # CAN 2.0B telemetry + DBC export, health frame, bus monitor
│   ├── dva.py                 # Differential & incremental capacity analysis (dV/dQ, dQ/dV)
│   ├── diagnostics.py         # EIS (Nyquist) simulation + C-rate capability map
│   ├── range_predictor.py     # Physics-based EV range predictor (weather/traffic/road coupling)
│   ├── interpret.py           # Plain-language explanations, feature importances, fusion
│   ├── agent.py               # Optional LLM diagnostic agent (LangGraph / LangChain / Langfuse)
│   ├── data.py                # Load / CC-CV / power profiles, NASA-like & ageing datasets
│   ├── datasets.py            # Real-dataset loaders (LG / NASA) + estimator leaderboard
│   └── calibration.py         # ECM parameter-ID from data, learned scatter, validation reports
├── app/streamlit_app.py       # Live multi-tab dashboard + range predictor
├── notebooks/                 # Executed end-to-end demo
├── scripts/build_notebook.py  # Reproducible notebook generator
├── bms.dbc                    # Shipped Vector DBC (cantools-validated, matches the encoder)
├── tests/test_bms.py          # 307 unit tests
├── figures/                   # 12 PNGs produced by the notebook
├── docs/architecture.md       # Layered-design notes & invariants
├── docs/estimation.md         # Which model produces each quantity + SoC benchmark
├── scripts/benchmark_estimators.py  # Head-to-head SoC RMSE across chemistries/conditions
├── pyproject.toml • CHANGELOG.md • CONTRIBUTING.md • requirements.txt • LICENSE (MIT)
```

---

## Module-to-capability map

| Capability | Module(s) |
|---|---|
| Cell chemistry library (7 chemistries) | `bms/chemistry.py` |
| OCV–SOC + temperature coefficient | `bms/ocv_soc.py` |
| Second-order RC ECM + parameter ID | `bms/ecm.py` |
| Series × parallel pack (shared-node currents) | `bms/pack.py` |
| 1-D FDM thermal + PID / predictive cooling | `bms/thermal.py` |
| Three balancing strategies + comparison | `bms/balancing.py` |
| SoC estimators — CC / EKF / UKF / LSTM | `bms/soc_estimators.py` |
| **Online SoH — joint EKF (SoC + capacity)** | `bms/soh_estimator.py` |
| **Model-agnostic estimator registry + BYO + uncertainty** | `bms/estimation.py` |
| Fault injection + hybrid (rule + ML) detection | `bms/faults.py`, `bms/_train_detector.py` |
| **Mechanical / gas faults — pressure, swelling, venting, internal short** | `bms/mechanics.py` |
| FMEA with S/O/D/RPN + RUL | `bms/fmea.py` |
| **Charging-method physics (AC/DC, fast/slow) + plating** | `bms/charging.py` |
| **Dynamic aging — capacity fade + resistance growth** | `bms/aging.py` |
| Supervisor + **RC pre-charge plant (inrush/energy)** + **SoH-aware control** | `bms/control.py` |
| Lifetime accounting (passport) | `bms/passport.py` |
| State of Power (multi-horizon limits) | `bms/sop.py` |
| **CAN 2.0B telemetry + shipped `.dbc`, health frame & bus monitor** | `bms/can.py`, `bms.dbc` |
| DVA / ICA, EIS, C-rate map | `bms/dva.py`, `bms/diagnostics.py` |
| Physics-based EV range prediction | `bms/range_predictor.py` |
| **Interpretability — explanations (opt. LLM), importances, fusion** | `bms/interpret.py` |
| **Framework adapters (sklearn / ONNX / BYO)** | `bms/estimation.py` |
| **Optional LLM diagnostic agent (LangChain / LangGraph / Langfuse)** | `bms/agent.py` |
| **Real-dataset validation — loaders + estimator leaderboard** | `bms/datasets.py` |
| **Calibration — ECM parameter-ID, learned scatter, validation reports** | `bms/calibration.py` |
| Visualisation | `app/streamlit_app.py`, `notebooks/` |

---

## Installation

Tested on Python 3.10–3.13. Pip-installable:

```bash
pip install -e .                  # core library only
pip install -e ".[app,notebook]"  # + Streamlit dashboard and notebook tooling
pip install -e ".[dev]"           # + pytest and ruff (for contributors)
pip install -e ".[onnx]"          # + ONNX Runtime (framework-neutral estimators)
pip install -e ".[agent]"         # + LangChain / LangGraph / Langfuse (LLM diagnostic agent)
```

Core runtime dependencies are intentionally minimal (numpy, scipy, pandas, scikit-learn, filterpy);
the `app` / `notebook` extras add streamlit/plotly and jupyter/matplotlib.

**Why no PyTorch / TensorFlow?** The LSTM SoC estimator is implemented from scratch in NumPy
(including BPTT and Adam) to keep dependencies minimal — and you can still plug a Torch/ONNX model
in via `FunctionSocEstimator` (see below).

---

## Quickstart

### Simulate a pack

```python
import bms

pack     = bms.BatteryPack(bms.PackConfig(n_cells=6, n_parallel=2, chemistry="lfp", seed=42))
thermal  = bms.ThermalModel(n_cells=6)
detector = bms.HybridFaultDetector(chemistry="lfp")
sup      = bms.BMSSupervisor(pack, thermal, detector)

for k, ii in enumerate(bms.generate_load_profile(60, mode="drive", c_rate=1.0, capacity_Ah=3.2)):
    out = sup.step(float(ii), 1.0, k=k)

print(sup.passport.summary())            # lifetime EFC / RTE / throughput
print(bms.explain_state(out))            # plain-language state summary
```

### Swap SoC estimators by name (model-agnostic) + uncertainty

```python
est = bms.make_soc_estimator("ukf", params=bms.ECMParameters(), ocv_curve=bms.OCVSOC())
est.reset(0.9); est.run(currents, voltages, dt=1.0)
print(bms.soc_estimate(est))             # Estimate(value=..., sigma=...)  ← 1-σ uncertainty

# Bring any trained model (sklearn / PyTorch / ONNX):
byo = bms.FunctionSocEstimator(my_model.predict, name="onnx")
```

### Online SoH (joint EKF: SoC + capacity)

```python
soh = bms.make_soc_estimator("joint_ekf", capacity_Ah=2.3)
soh.reset(0.95); soh.run(currents, voltages, dt=1.0)
print(f"SoH {soh.soh*100:.1f}% ± {soh.soh_uncertainty_1sigma*100:.1f}%,  Q={soh.capacity_Ah:.3f} Ah")
```

### Charging physics + dynamic aging

```python
from bms import ChargingModel, ChargeProtocol, ChargeMethod, compare_methods
print(compare_methods(pack_energy_kWh=60, q_nom_Ah=2.3, r0_ohm=0.025, v_nom=3.7))
# -> per method: C-rate, charge time, efficiency, peak temp, plating risk, fade/session
```

### SoH-aware control

```python
cfg = bms.SupervisorConfig(soh_aware=True)     # derate + plating cap
sup = bms.BMSSupervisor(pack, thermal, detector, config=cfg)
sup.set_soh(0.75)                              # feed live SoH from the joint EKF / AgingModel
out = sup.step(-30.0, 1.0)                      # aggressive cold charge → capped below plating limit
```

### EV range & diagnostics

```python
from bms import RangePredictor, VehicleParams, WeatherConditions, ROUTE_PROFILES
r = RangePredictor(VehicleParams.suv()).predict(80_000.0, "nmc",
        ROUTE_PROFILES["wltp"], WeatherConditions.cold_winter())
print(f"{r.estimated_range_km:.0f} km (weather penalty {r.weather_penalty_pct:.1f}%)")
```

---

## Design highlights

### Physics
- **OCV–SOC**: monotonic PCHIP interpolant per chemistry with a `temp_coeff_V_per_K` shift.
- **ECM**: 2-RC discrete recurrence with a closed-form EKF Jacobian; `R0` Arrhenius-scaled by temperature.
- **Pack**: per-cell scatter; parallel groups share one terminal node solved from KCL, so mismatched
  cells exchange **circulating currents**.
- **Thermal**: 1-D FDM rod, CFL-guarded (auto sub-steps for any `dt`), with irreversible heat
  `Q = |I·(OCV − Vₜ)|` — correct for both charge and discharge.

### Estimation, SoH & the model-agnostic layer
- `SocEstimator` (batch `run`) and `RecursiveSocEstimator` (adds `update`/`soc`) protocols; the
  built-ins satisfy them unchanged. Pick any by name via the registry, or wrap your own model with
  `FunctionSocEstimator`. `soc_estimate()` returns a value **with 1-σ uncertainty** from the filter
  covariance. The **joint EKF** augments the state with capacity to track SoH online.

### Charging & aging (put a number on it)
- Effective C-rate ≈ `power_kW / pack_energy_kWh`; wall-to-battery efficiency (OBC loss for AC, higher
  I²R heat for DC); cell heating ∝ C²; a temperature/SoC-dependent **lithium-plating** limit. The
  `AgingModel` accumulates capacity fade + resistance growth from C-rate, temperature, DoD, plating,
  and a √-time calendar term, and writes SoH back onto the cells.

### Fault detection (rule + ML, rule-only trip authority)
Rules cover overcharge, undervoltage, short-circuit (over-current), and thermal runaway; only the rule
layer can trip. A `RandomForestClassifier` adds advisory drift detection — with `feature_importances()`
to explain *why* it fired.

### Interpretability
`explain_state()` / `explain_charge()` (plain language), `feature_importances()` (named), 
`estimator_agreement()` (spread + inverse-variance **fused** estimate), `soc_report()` (±kσ band).

---

## Interactive dashboard

`streamlit run app/streamlit_app.py` opens a four-mode Plotly app: **🔬 Simulation** (chemistry, SxP
topology, current/power load, fault injection; tabs for Live Signals, SoH & Aging, Battery Passport,
Diagnostics, Fault Analysis), **🚗 Range Predictor**, **🔋 Charging & Aging**, and
**🔌 Hardware & Agent** — self-contained demos of the CAN/DBC bus (frame table, live integrity
monitor, downloadable `bms.dbc`), the RC pre-charge plant (interactive inrush/energy), the
deterministic diagnostic agent (findings → gated actions), real-data calibration (estimator
leaderboard + pulse parameter-ID), and mechanical gas/pressure sensing (pressure-leads-temperature).

---

## Testing

```bash
pytest -q                                  # 307 tests, ~30 s
pytest --cov=bms --cov-fail-under=85       # coverage gate (CI enforces ≥ 85%; currently 90%)
ruff check .                               # lint — blocking in CI
```

Covers OCV/ECM correctness and parameter recovery, pack/parallel bookkeeping, thermal stability,
balancer energy monotonicity, estimator accuracy + the model-agnostic contract, online SoH
convergence, charging/aging behaviour, fault trip authority, supervisor transitions + SoH-aware
limiting, precharge, passport, SoP, CAN round-trips, DVA/ICA/EIS invariants, interpretability, and
range-predictor energy conservation.

---

## Roadmap

- **Real data** — loaders + an estimator leaderboard ship in `bms/datasets.py`; drop in the
  LG-18650 / NASA PCoE files (see `DATASET_SOURCES`) and it runs unchanged.
- **Online resistance SoH** — estimate R₀ growth alongside capacity.
- **Dashboard expansion** — surface charging, aging, online SoH, SoP, and interpretability.
- **Publish to PyPI**; add rendered API docs.

---

## License

MIT — see [`LICENSE`](LICENSE). © 2026 Akash Preetham.
