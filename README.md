# BMS Digital Twin

A research-grade, **AI-augmented Battery Management System (BMS) digital twin** for a multi-cell
Li-ion pack (4S/6S). It integrates electrochemical modelling, thermal simulation, electrical
balancing, fault detection, and intelligent control into one reproducible framework — every
module is independently usable and covered by unit tests.

> **Status.** ✅ 27/27 unit tests pass • 8 modules • 12 figures • Streamlit dashboard +
> demo notebook • ~2,300 lines of Python.

---

## What's inside

```
bms_digital_twin/
├── bms/                       # Library
│   ├── ocv_soc.py             # OCV-SOC characteristic (PCHIP interpolant)
│   ├── ecm.py                 # 2-RC equivalent-circuit model + parameter ID
│   ├── pack.py                # Series multi-cell pack with manufacturing scatter
│   ├── thermal.py             # 1-D FDM thermal model + PID controller
│   ├── balancing.py           # Passive / SC / inductor balancing strategies
│   ├── soc_estimators.py      # Coulomb counter, EKF, UKF, NumPy-LSTM + benchmark
│   ├── faults.py              # Fault injection, rule + ML hybrid detector
│   ├── fmea.py                # FMEA / RPN table + RUL estimator
│   ├── control.py             # Supervisory state-machine
│   ├── data.py                # Synthetic + NASA-like load profiles & ageing data
│   └── _train_detector.py     # Synthetic labelled-data generator for the ML detector
├── notebooks/
│   └── BMS_Digital_Twin_Demo.ipynb   # End-to-end walkthrough (executed)
├── app/
│   └── streamlit_app.py       # Live dashboard
├── scripts/
│   └── build_notebook.py      # Reproducible notebook generator
├── tests/
│   └── test_bms.py            # 27 unit tests
├── figures/                   # 12 PNGs produced by the notebook
├── docs/
│   └── architecture.md        # High-level design notes
└── requirements.txt
```

### Module-to-spec map

| Spec requirement | Module |
|---|---|
| 1. Second-order RC ECM + OCV-SOC + parameter fit | `bms/ocv_soc.py`, `bms/ecm.py` |
| 2. Three balancing strategies (passive, SC, inductor) | `bms/balancing.py` |
| 3. SOC benchmark — CC / EKF / UKF / LSTM | `bms/soc_estimators.py` |
| 4. FDM thermal model + PID regulation | `bms/thermal.py` |
| 5. Fault injection + hybrid detection | `bms/faults.py`, `bms/_train_detector.py` |
| 5. FMEA with S/O/D/RPN | `bms/fmea.py` |
| 6. Intelligent control layer | `bms/control.py` |
| 7. Visualisation interface | `notebooks/`, `app/streamlit_app.py` |
| 8. Modular, tested, reproducible | `tests/`, `scripts/build_notebook.py` |
| Optional: Predictive maintenance / RUL | `bms/fmea.py::estimate_rul` |

---

## Installation

Tested on Python 3.10–3.12 / Linux + macOS.

```bash
pip install -r requirements.txt
```

`requirements.txt` is intentionally minimal:

```
numpy>=1.24
scipy>=1.10
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
filterpy>=1.4.5      # UKF
streamlit>=1.30      # dashboard
nbformat>=5.9        # notebook builder
jupyter>=1.0         # to run the notebook
pytest>=7.4
```

**Why no PyTorch / TensorFlow?** The LSTM SOC estimator is implemented from scratch in
NumPy — including BPTT and Adam — to keep dependencies minimal and to make the recurrent
gradient flow legible for review. For production, swapping it for `torch.nn.LSTM` is a
~30-line change.

---

## Quickstart

### 1. Run the test suite (≈ 8 s)

```bash
pytest tests/ -v
```

### 2. Walk through the demo notebook (≈ 90 s end-to-end)

```bash
jupyter lab notebooks/BMS_Digital_Twin_Demo.ipynb
```

The pre-executed copy already contains every figure. To regenerate everything from scratch:

```bash
python scripts/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/BMS_Digital_Twin_Demo.ipynb
```

### 3. Launch the live dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard (4S/6S pack, configurable load profile, side-bar fault injection, live SOC /
voltage / temperature / cooling-duty / fault-alert plots) is the closest equivalent to
the SCADA view a real BMS engineer would use.

### 4. Use the package as a library

```python
import bms

pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=42))
thermal = bms.ThermalModel(n_cells=4)
detector = bms.HybridFaultDetector()                  # rule-only until .fit() is called
sup = bms.BMSSupervisor(pack, thermal, detector)

i_load = bms.generate_load_profile(60, mode="drive", c_rate=1.0, capacity_Ah=2.3)
for k, ii in enumerate(i_load):
    out = sup.step(float(ii), 1.0, k=k)
    print(out["state"], "SOC=", out["soc"].round(3))
```

---

## Design highlights

### Physics

- **OCV–SOC**: PCHIP interpolant on 13 anchor points typical of an NMC 18650 (3.0 V at
  0% SOC, 4.2 V at 100%). Smooth, monotonic — well-defined inverse for SOC anchoring.
- **ECM**: 2-RC discrete-time recurrence with `a₁,₂ = exp(−dt/τ₁,₂)`. The Jacobian
  closed-form is used by the EKF.
- **Parameter ID**: `scipy.optimize.least_squares` minimises voltage residual on a noisy
  I/V trace. Voltage RMSE drops to ~5 mV (the noise floor) on a typical pulse profile.
- **Pack**: each cell carries independent capacity (3 % σ), R₀ (5 % σ), and initial SOC
  (5 % σ) — realistic manufacturing scatter.
- **Thermal**: 1-D FDM rod with κ for cell-to-cell conduction, time-varying convective
  coefficient `h(t) ∈ [h_min, h_max]` modulated by PID duty cycle. Mirror BC at the ends.
- **Heat generation**: `Q = i²R₀ + i·max(OCV − V_t, 0)` — Ohmic + over-potential.

### Estimators

| Estimator | RMSE @ 5 mV noise | Robust to current bias? |
|---|---|---|
| Coulomb counting | drifts with bias | ❌ |
| EKF (this project) | < 0.3 % SOC | ✅ |
| UKF (filterpy) | < 0.3 % SOC | ✅ |
| LSTM (NumPy) | ~1.6 % SOC | depends on training distribution |

### Balancing tradeoffs (4S, 13.5 % initial imbalance, 3 h idle)

| Strategy | Time to balance to 0.5 % SOC | Energy lost |
|---|---|---|
| Passive resistive | ≈ 8,800 s | ~ 8,200 J (dissipated as heat) |
| Switched capacitor | did not converge in 3 h | ~ 165 J |
| Inductor active | ≈ 2,400 s | ~ 51 J |

### Fault detection: rule + ML, with rule-only trip authority

Five failure modes (overcharge, short circuit, thermal runaway, sensor dropout, sensor bias)
are simulated and labelled. The detector OR-fuses two layers:

- **Rule layer** — deterministic thresholds on V, T. *Only this layer can trip the
  contactor.* This matches functional-safety convention and prevents ML misclassifications
  from triggering nuisance shutdowns.
- **ML layer** — `RandomForestClassifier` with confidence threshold 0.65. Surfaces subtle
  drift faults (slow sensor bias) that lie within nominal V/T limits, but only as warnings.

### Supervisor (state machine)

```
       request_current ≠ 0           rule alarms ≥ 3
IDLE ──────────────────────► OPERATING ──────────────► FAULT ──► (manual reset)
  ▲                                ▲
  │ imbalance < 0.5 %              │ imbalance > 0.5 %
  └──────────── BALANCING ◄────────┘
```

Balancing strategy is selected dynamically based on the current SOC imbalance:

```
imbalance ≥ 5 %    → InductorBalancer       (high efficiency, fast)
imbalance ≥ 2 %    → SwitchedCapacitor      (medium efficiency)
≥ 0.5 % near top   → PassiveBalancer        (only at near-full SOC)
otherwise          → no balancing
```

---

## Reproducibility

Every randomness source is seeded:

- `PackConfig.seed` for cell scatter
- `np.random.default_rng(seed)` for noise traces
- `generate_load_profile(seed=…)` for drive cycles
- `RandomForestClassifier(random_state=…)` for the ML detector
- `LSTMEstimator(seed=…)` for weight init and training-set sampling

Re-running `scripts/build_notebook.py` followed by `nbconvert --execute` reproduces every
figure bit-for-bit on the same NumPy / SciPy versions.

---

## Optional extensions

### Implemented

- ✅ **Predictive maintenance / RUL** (`bms.estimate_rul`) — square-root-of-cycles capacity-fade
  fit, computes SoH and remaining cycles to 80 % EoL.
- ✅ **AI-augmented control** — the supervisor in `bms.BMSSupervisor` dynamically swaps
  balancing strategies and the detector hybridises rules with a Random Forest.

### Hooks left for future work

- **Reinforcement learning** for adaptive cooling-duty policies. The supervisor's `step` method
  already returns a reward-shaped state dict — wiring in `gymnasium.Env` is mostly plumbing.
- **Cloud / Databricks integration**. Every signal is already a `pandas.DataFrame`; emitting
  to MQTT, Kafka, or a Delta Lake table is a one-line `to_*` call from each scenario.
- **PyTorch LSTM** — drop-in replacement for `bms.LSTMEstimator`, same I/O contract.

---

## License

MIT — see `LICENSE`.
