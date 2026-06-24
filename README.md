# BMS Digital Twin

A research-grade, **AI-augmented Battery Management System (BMS) digital twin** for multi-cell
Li-ion / solid-state packs. It unifies electrochemical modelling, thermal simulation, cell
balancing, multi-method state-of-charge estimation, hybrid fault detection, lifetime accounting,
electrochemical diagnostics, and a physics-based EV range predictor into one reproducible,
fully-tested framework — every module is independently usable and exercised by unit tests.

<p>
  <img alt="version" src="https://img.shields.io/badge/version-0.4.0-blue">
  <img alt="tests" src="https://img.shields.io/badge/tests-155%20passing-brightgreen">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%E2%80%933.13-blue">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="dashboard" src="https://img.shields.io/badge/dashboard-Streamlit-ff4b4b">
</p>

> **Status.** ✅ 155/155 unit tests pass • 16 library modules • 7 chemistries • 12 figures •
> 5-tab Streamlit dashboard + EV range predictor • executed demo notebook.

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
- [Reproducibility](#reproducibility)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [License](#license)

---

## Highlights

- **Seven cell chemistries** — NMC, LFP, LMFP, LTO, NCA, LMO, and a solid-state (SSB) model,
  each with its own OCV–SOC table, Arrhenius resistance scaling, voltage window, runaway
  temperature, and default ECM. Switch chemistry with a single argument.
- **Configurable pack topology** — arbitrary series × parallel (`SxP`) packs with realistic
  per-cell manufacturing scatter (capacity, R₀, initial SOC).
- **Four SOC estimators, benchmarked** — Coulomb counter, Extended Kalman Filter,
  Unscented Kalman Filter, and an LSTM implemented from scratch in NumPy.
- **Hybrid fault detection** — deterministic rule layer (sole trip authority) OR-fused with a
  Random-Forest ML layer (advisory), across five failure modes.
- **Intelligent supervisor** — a state machine that gates faults, dynamically selects a
  balancing strategy, drives predictive cooling, and supports both current- and power-mode loads.
- **Lifetime accounting** — a battery passport tracking equivalent full cycles, depth-weighted
  cycles, round-trip efficiency, and energy throughput.
- **Electrochemical diagnostics** — DVA/ICA aging fingerprints, simulated EIS (Nyquist), and a
  C-rate capability map over the SOC × temperature envelope.
- **Physics-based EV range predictor** — first-principles traction/regen/HVAC/accessory model
  with weather, traffic, road-quality, and battery-temperature coupling, plus vehicle and route
  presets (including India-specific drive cycles, city routes, and seasonal weather).
- **Reproducible** — every randomness source is seeded; the demo notebook and all 12 figures
  regenerate deterministically.

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
bms_digital_twin/
├── bms/                       # Library (16 modules)
│   ├── chemistry.py           # 7 chemistries: OCV tables, Arrhenius, voltage limits, defaults
│   ├── ocv_soc.py             # OCV–SOC characteristic (PCHIP interpolant, temp coefficient)
│   ├── ecm.py                 # 2-RC equivalent-circuit model, Arrhenius scaling, parameter ID
│   ├── pack.py                # Series × parallel pack with manufacturing scatter
│   ├── thermal.py             # 1-D FDM thermal model + PID + predictive cooling controller
│   ├── balancing.py           # Passive / switched-capacitor / inductor balancing + comparison
│   ├── soc_estimators.py      # Coulomb counter, EKF, UKF, NumPy LSTM + benchmark harness
│   ├── faults.py              # Fault injection + hybrid rule/ML detector + feature buffer
│   ├── _train_detector.py     # Synthetic labelled-data generator for the ML detector
│   ├── fmea.py                # FMEA / RPN table + RUL estimator (capacity & resistance fade)
│   ├── control.py             # Supervisory state machine (BMSSupervisor)
│   ├── passport.py            # Battery passport — lifetime EFC / DWC / RTE / throughput
│   ├── dva.py                 # Differential & incremental capacity analysis (dV/dQ, dQ/dV)
│   ├── diagnostics.py         # EIS (Nyquist) simulation + C-rate capability map
│   ├── range_predictor.py     # Physics-based EV range predictor (weather/traffic/road coupling)
│   └── data.py                # Load / CC-CV / power profiles, NASA-like & ageing datasets
├── notebooks/
│   └── BMS_Digital_Twin_Demo.ipynb   # End-to-end executed walkthrough
├── app/
│   └── streamlit_app.py       # Live multi-tab dashboard + range predictor
├── scripts/
│   └── build_notebook.py      # Reproducible notebook generator
├── tests/
│   └── test_bms.py            # 155 unit tests
├── figures/                   # 12 PNGs produced by the notebook
├── docs/
│   └── architecture.md        # Layered-design notes & invariants
├── requirements.txt
└── LICENSE                    # MIT
```

---

## Module-to-capability map

| Capability | Module(s) |
|---|---|
| Cell chemistry library (7 chemistries) | `bms/chemistry.py` |
| OCV–SOC characteristic + temperature coefficient | `bms/ocv_soc.py` |
| Second-order RC ECM + parameter identification | `bms/ecm.py` |
| Series × parallel pack with scatter | `bms/pack.py` |
| 1-D FDM thermal model + PID / predictive cooling | `bms/thermal.py` |
| Three balancing strategies + comparison | `bms/balancing.py` |
| SOC benchmark — CC / EKF / UKF / LSTM | `bms/soc_estimators.py` |
| Fault injection + hybrid (rule + ML) detection | `bms/faults.py`, `bms/_train_detector.py` |
| FMEA with S/O/D/RPN + RUL | `bms/fmea.py` |
| Supervisory control state machine | `bms/control.py` |
| Lifetime accounting (passport) | `bms/passport.py` |
| Aging diagnostics — DVA / ICA | `bms/dva.py` |
| EIS spectrum + C-rate capability map | `bms/diagnostics.py` |
| Physics-based EV range prediction | `bms/range_predictor.py` |
| Synthetic load / power / ageing datasets | `bms/data.py` |
| Visualisation | `notebooks/`, `app/streamlit_app.py` |
| Modular, tested, reproducible | `tests/`, `scripts/build_notebook.py` |

---

## Installation

Tested on Python 3.10–3.13.

```bash
pip install -r requirements.txt
```

`requirements.txt` is intentionally minimal:

```
numpy>=1.24        scipy>=1.10        pandas>=2.0        matplotlib>=3.7
scikit-learn>=1.3  filterpy>=1.4.5    nbformat>=5.9      jupyter>=1.0
streamlit>=1.30    plotly>=5.18       pytest>=7.4
```

**Why no PyTorch / TensorFlow?** The LSTM SOC estimator is implemented from scratch in NumPy —
including BPTT and Adam — to keep dependencies minimal and to make the recurrent gradient flow
legible for review. Swapping it for `torch.nn.LSTM` is a ~30-line change.

---

## Quickstart

### 1. Run the test suite

```bash
pytest tests/ -q          # 155 tests
```

### 2. Walk through the demo notebook

```bash
jupyter lab notebooks/BMS_Digital_Twin_Demo.ipynb
```

The pre-executed copy already contains every figure. To regenerate from scratch:

```bash
python scripts/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/BMS_Digital_Twin_Demo.ipynb
```

### 3. Launch the live dashboard

```bash
streamlit run app/streamlit_app.py
```

### 4. Use the package as a library

```python
import bms

# A 6S2P LFP pack, with thermal model and a (rule-only until fitted) fault detector
pack     = bms.BatteryPack(bms.PackConfig(n_cells=6, n_parallel=2, chemistry="lfp", seed=42))
thermal  = bms.ThermalModel(n_cells=6)
detector = bms.HybridFaultDetector(chemistry="lfp")
sup      = bms.BMSSupervisor(pack, thermal, detector)

i_load = bms.generate_load_profile(60, mode="drive", c_rate=1.0, capacity_Ah=3.2)
for k, ii in enumerate(i_load):
    out = sup.step(float(ii), 1.0, k=k)
    print(out["state"], "SOC=", out["soc"].round(3))

print(sup.passport.summary())          # lifetime EFC / RTE / throughput
```

### 5. Predict EV range

```python
from bms import RangePredictor, VehicleParams, WeatherConditions, ROUTE_PROFILES

pred   = RangePredictor(VehicleParams.suv())
result = pred.predict(
    pack_energy_Wh=80_000.0, chemistry="nmc",
    route=ROUTE_PROFILES["wltp"], weather=WeatherConditions.cold_winter(),
)
print(f"{result.estimated_range_km:.0f} km, completable={result.route_completable}, "
      f"weather penalty={result.weather_penalty_pct:.1f}%")
```

### 6. Aging & impedance diagnostics

```python
from bms import (synthetic_discharge_for_dva, compute_dva, compute_ica,
                 ECMParameters, simulate_eis, compute_crate_map)

q, v       = synthetic_discharge_for_dva("lfp")
q_ax, dva  = compute_dva(q, v)          # dV/dQ aging fingerprint
v_ax, ica  = compute_ica(q, v)          # dQ/dV — ideal for flat-plateau LFP

p              = ECMParameters.for_nmc()
f, z_re, z_im  = simulate_eis(p, temperature_C=-10.0)        # Nyquist spectrum
soc, T, cmap   = compute_crate_map(p, chemistry="nmc")        # power envelope
```

---

## Design highlights

### Physics

- **OCV–SOC**: PCHIP interpolant per chemistry (e.g. NMC 3.0 V → 4.2 V; LFP's characteristic
  flat plateau). Smooth, monotonic, with a configurable `temp_coeff_V_per_K` shift.
- **ECM**: 2-RC discrete-time recurrence with `a₁,₂ = exp(−dt/τ₁,₂)`; closed-form Jacobian used
  by the EKF. `R0` scales with temperature via a per-chemistry Arrhenius factor.
- **Parameter ID**: `scipy.optimize.least_squares` fits ECM parameters to a noisy I/V trace,
  driving voltage RMSE down toward the noise floor on a pulse profile.
- **Pack**: each cell carries independent capacity, R₀, and initial-SOC scatter; `n_cells`
  series groups × `n_parallel` cells per group.
- **Thermal**: 1-D finite-difference rod with cell-to-cell conduction and a time-varying
  convective coefficient modulated by a PID / predictive cooling controller.
  Heat generation `Q = i²R₀ + i·max(OCV − Vₜ, 0)` (ohmic + over-potential).

### SOC estimators (representative results from the demo notebook)

| Estimator | Accuracy @ ~5 mV noise | Robust to current bias? |
|---|---|---|
| Coulomb counting | drifts with bias | ❌ |
| EKF | sub-percent SOC | ✅ |
| UKF (filterpy) | sub-percent SOC | ✅ |
| LSTM (NumPy) | low single-digit % | depends on training distribution |

### Balancing trade-offs

Three strategies — `PassiveBalancer`, `SwitchedCapacitorBalancer`, `InductorBalancer` —
trade balancing speed against energy lost as heat. The active inductor balancer is the fastest
and most efficient; passive resistive bleed is simplest but dissipates the most energy.
`compare_balancers(...)` runs all three head-to-head.

### Fault detection: rule + ML, with rule-only trip authority

Five failure modes (overcharge, short circuit, thermal runaway, sensor dropout, sensor bias)
are simulated and labelled. The detector OR-fuses two layers:

- **Rule layer** — deterministic thresholds on V, T. *Only this layer can trip the contactor*,
  matching functional-safety convention and preventing ML misclassifications from causing
  nuisance shutdowns.
- **ML layer** — a `RandomForestClassifier` with a confidence threshold. Surfaces subtle drift
  faults (slow sensor bias) within nominal V/T limits — but only as advisory warnings.

### Supervisor (state machine)

```
       request ≠ 0                    rule alarms ≥ N
IDLE ──────────────────► OPERATING ──────────────────► FAULT ──► (manual reset)
  ▲                          ▲
  │ imbalance < 0.5 %        │ imbalance > 0.5 %
  └────────── BALANCING ◄────┘
```

The supervisor accepts either a requested **current** or a requested **power** (converted via the
instantaneous pack voltage, then de-rated against configurable power limits), reports
`peak_power_W` capability at the current SOC/temperature, drives predictive cooling, selects a
balancing strategy by imbalance magnitude, and updates the battery passport every cycle.

### Battery passport

`bms.BatteryPassport` (exposed as `supervisor.passport`) accumulates, over the pack's life:
**equivalent full cycles** (discharge Ah ÷ nominal Ah), **depth-weighted cycles**
(Σ|ΔSOC|/2 half-cycle approximation), **round-trip efficiency**, and total charge/discharge
energy and time. `summary()` returns the full snapshot dict.

### Aging & impedance diagnostics

- **DVA (dV/dQ)** and **ICA (dQ/dV)** with Savitzky-Golay smoothing — peak shifts and height
  loss fingerprint capacity fade and loss of active material; ICA is especially diagnostic for
  flat-plateau LFP/LMFP.
- **EIS** — the 2-RC ECM (plus a Warburg diffusion tail) maps onto an impedance spectrum;
  `simulate_eis` returns a Nyquist curve with the R₀ intercept, two depressed semicircles, and a
  45° diffusion tail.
- **C-rate capability map** — maximum continuous discharge C-rate over a SOC × temperature grid,
  exposing where cold or low SOC limits power.

### EV range predictor

A first-principles consumption model — traction (aero + rolling + grade), stop-and-go with
regen, HVAC (temperature-driven, COP-modelled), and constant accessory load — coupled to the
battery via temperature-dependent capacity and efficiency derating. Weather affects air density
(altitude), headwind, rolling resistance, and HVAC load. Ships with vehicle presets
(compact / sedan / SUV / truck plus India e-scooter / e-motorcycle / e-moped), drive-cycle
profiles (WLTP, city, highway, mixed, mountain, MIDC, India NH), India city routes, and seasonal
weather presets.

---

## Interactive dashboard

`streamlit run app/streamlit_app.py` opens a two-mode app:

- **🔬 Simulation** — chemistry selector, series × parallel topology, current- or power-mode
  load, and sidebar fault injection, with five tabs:
  **📊 Live Signals** · **🔬 SoH & Aging** · **📋 Battery Passport** · **🔭 Diagnostics** ·
  **⚠ Fault Analysis**.
- **🚗 Range Predictor** — interactive range estimation across vehicles, routes, and weather,
  with a per-chemistry comparison view.

All charts are Plotly. This is the closest equivalent to the SCADA view a BMS engineer would use.

---

## Reproducibility

Every randomness source is seeded — `PackConfig.seed`, `np.random.default_rng(seed)` for noise,
`generate_load_profile(seed=…)`, `RandomForestClassifier(random_state=…)`, and
`LSTMEstimator(seed=…)`. Re-running `scripts/build_notebook.py` followed by
`nbconvert --execute` reproduces every figure deterministically on the same NumPy/SciPy versions.

---

## Testing

```bash
pytest tests/ -q          # 155 tests, ~30 s
```

The suite covers OCV/ECM correctness and parameter recovery, pack scatter and series/parallel
bookkeeping, thermal stability, balancer energy monotonicity, estimator accuracy bounds, fault
trip authority, FMEA/RUL, supervisor state transitions, passport accounting, DVA/ICA/EIS shape
invariants, and range-predictor energy conservation.

---

## Roadmap

- **Reinforcement learning** for adaptive cooling-duty policies — the supervisor's `step` already
  returns a reward-shaped state dict.
- **Cloud / streaming integration** — every signal is a `pandas.DataFrame`; emitting to MQTT,
  Kafka, or a Delta Lake table is a one-line `to_*` call.
- **PyTorch LSTM** — drop-in replacement for `bms.LSTMEstimator`, same I/O contract.

---

## License

MIT — see [`LICENSE`](LICENSE).
