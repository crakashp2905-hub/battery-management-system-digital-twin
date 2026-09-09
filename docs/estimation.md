# Estimation — which model produces each quantity

This twin is **not a single model**. It is a physics plant (2nd-order RC ECM +
OCV–SOC + thermal FDM) with a bank of estimators on top and a deterministic
safety/diagnostics layer beside them. This page says exactly which component
produces each quantity, and benchmarks the SoC estimators head-to-head.

## TL;DR — quantity → producer

| Quantity | Produced by | Kind | Where it surfaces |
|---|---|---|---|
| **SoC** (online) | `JointEKFSoH` (default) — or `ekf` / `ukf` / `coulomb` / `lstm` via the registry | Recursive state estimate (+1σ) | `BMSSupervisor.step()['soc_estimated','soc_sigma']` |
| **SoH** (online, capacity retention Q/Q₀) | `JointEKFSoH` — **same filter as SoC** | Recursive parameter estimate (+1σ) | `step()['soh_estimated','soh_sigma','capacity_est_Ah']` |
| **SoH / capacity fade** (offline, over cycles) | `aging.AgingModel`, `datasets.soh_curve` | Degradation model / trend fit | SoH & Aging tab |
| **SoE** (state of energy, Wh) | `pack.state_of_energy_Wh()` | Derived (∫ V·I, energy not charge) | `step()['soe_Wh']` |
| **SoP** (power limits, 2/10/30 s) | `sop.StateOfPower` | Analytic limit (v_min/v_max + temp derate) | `BMSSupervisor.state_of_power()` |
| **Peak deliverable power** | `control.py` (v_min-bounded) | Analytic | `step()['peak_power_W']` |
| **Battery health record** (EFC, DWC, round-trip η, throughput) | `passport.BatteryPassport` | Lifetime accounting | Battery Passport tab |
| **RUL** (remaining useful life) | `fmea.estimate_rul[_with_resistance]`, `datasets.soh_curve` | Trend extrapolation to EoL | SoH & Aging / Diagnostics |
| **Thermal runaway & faults** | `faults.HybridFaultDetector` (physics rules + RandomForest) + `mechanics` (pressure leads temperature) | Detection / classification | Fault Analysis tab; `step()['fault_label','fault_source']` |
| **Plain-language explanation** | `interpret.explain_state/explain_charge`, optional `agent.DiagnosticAgent` | Narration (rules; optional LLM) | Hardware & Agent tab |

Key point: **SoC and SoH come from one online filter** (the joint-EKF). SoE is
*derived* from SoC, not separately estimated. Thermal-runaway is a *detector*,
deliberately kept off the Kalman path so a safety trip never depends on a filter
converging.

## The SoC estimator bank

All estimators implement one interface (`bms.estimation.SocEstimator`) and are
built by name from a registry, so you can swap algorithms without touching
callers:

```python
from bms import make_soc_estimator, available_soc_estimators
available_soc_estimators()      # ['coulomb', 'ekf', 'ukf', 'lstm', 'joint_ekf']
est = make_soc_estimator("joint_ekf", params=..., ocv_curve=..., capacity_Ah=2.3)
```

| Name | Model | Needs training | Gives uncertainty | Also gives SoH |
|---|---|---|---|---|
| `coulomb` | Open-loop current integration | no | no | no |
| `ekf` | Extended Kalman filter on the ECM | no | yes (1σ) | no |
| `ukf` | Unscented Kalman filter on the ECM | no | yes (1σ) | no |
| `lstm` | Pure-NumPy LSTM (feature → SoC) | **yes** | no | no |
| `joint_ekf` | **Joint SoC + capacity EKF** | no | yes (1σ each) | **yes** |

The **joint-EKF** augments the state with capacity, `x = [SoC, V_RC1, V_RC2, Q]`,
and tracks `Q` as a slow random walk. Capacity becomes observable whenever SoC
moves appreciably (the *rate* of SoC change depends on `Q`), so `soh = Q/Q₀`
falls out of the same filter — with its own 1σ that stays wide under a flat load
and tightens on a real charge/discharge excursion. This is why it is the default
online estimator in the supervisor and dashboard.

## Benchmark — which SoC filter is most accurate?

Run the head-to-head yourself (uses the built-in `estimator_leaderboard`, whose
ground-truth SoC is the plant ECM's own SoC, so scoring is exact):

```bash
python scripts/benchmark_estimators.py            # per-(chemistry,condition) tables
python scripts/benchmark_estimators.py --markdown  # winner matrix
```

Three conditions per chemistry: **nominal** (clean current, 5 mV noise),
**current bias** (a +0.15 A current-sensor offset), and **voltage noise**
(20 mV). Results (SoC RMSE, lower is better):

### Winner (lowest SoC RMSE) per chemistry × condition

| chemistry | nominal | current_bias | voltage_noise |
|---|---|---|---|
| nmc | coulomb | **ekf** | coulomb |
| lfp | coulomb | **ekf** | coulomb |
| lmfp | coulomb | **ekf** | coulomb |
| nca | coulomb | **ekf** | coulomb |
| lto | coulomb | **ekf** | coulomb |
| lmo | coulomb | coulomb | coulomb |
| ssb | coulomb | coulomb | coulomb |

### Representative magnitudes (SoC RMSE, %)

| condition | coulomb | ekf | ukf | joint_ekf |
|---|---|---|---|---|
| NMC · nominal | **0.001** | 0.173 | 0.216 | 0.168 |
| NMC · current bias (0.15 A) | 1.883 | **0.315** | 0.337 | 0.934 |
| LMO · current bias (0.15 A) | **2.165** | 2.183 | 2.179 | 2.750 |

### How to read this (important, honest caveat)

- **Coulomb "wins" nominal/voltage-noise by construction.** In the synthetic
  fixture the ground-truth SoC is *defined* by the (unbiased) current, and the
  Coulomb counter integrates that same current — so it is near-perfect when the
  current sensor is clean and initial SoC is known. That is rarely true in the
  field, which is the whole reason model-based filters exist.
- **The `current_bias` column is the decision-relevant one.** With a realistic
  current-sensor offset, the **EKF corrects the bias via voltage feedback** and
  beats Coulomb by ~6× on sloped-OCV chemistries (NMC: **0.32 % vs 1.88 %**).
- **Flat-OCV chemistries (LMO, SSB) blunt every voltage-feedback filter.** When
  OCV barely changes with SoC, voltage carries little SoC information, so all
  methods drift together (~2 %). This is a well-known property of LFP-like/flat
  chemistries — plan for better current sensing there, not a cleverer filter.
- **`joint_ekf` trades a little SoC accuracy for online SoH.** Under bias it is
  worse on SoC than plain `ekf` (0.93 % vs 0.32 %) because it also spends
  observability on capacity. You pick it when you need **SoC *and* SoH from one
  filter**, not for SoC alone.
- `lstm` is excluded from the leaderboard because it needs separate training.

### Temperature — the value of a temperature sensor

Ranking estimators *across* temperatures is confounded: as the cell gets colder
both the true SoC trajectory and the physical operating point shift (a 1 C
discharge at −20 °C collapses the terminal voltage below any real cutoff), so a
cross-temperature RMSE table is apples-to-oranges. The fair, decision-relevant
question is instead answered on **one trace at a time**: score the same cold/hot
drive cycle with the filter **told** the temperature vs the filter **assuming
25 °C**. Run it with:

```bash
python scripts/benchmark_estimators.py --temperature            # sweep table
python scripts/benchmark_estimators.py --temperature --markdown  # docs table
```

EKF SoC RMSE (%), current bias 0.05 A, isothermal trace:

| ekf @ | 25 °C | 15 °C | 5 °C | −5 °C | −15 °C |
|---|---|---|---|---|---|
| nmc · **aware** | 0.08 | 0.11 | 0.21 | 0.33 | 0.24 |
| nmc · naive (assumes 25 °C) | 0.08 | 2.50 | 5.85 | 10.83 | **21.95** |
| lfp · **aware** | 1.47 | 1.32 | 1.01 | 0.22 | 0.74 |
| lfp · naive (assumes 25 °C) | 1.47 | 8.19 | 7.95 | 14.35 | **13.23** |

**Reading it:** a temperature-aware filter is essentially *flat* across
temperature (~0.1–0.3 % on NMC) because it uses the Arrhenius-shifted ECM
resistance and the temperature-corrected OCV. A temperature-**blind** filter is
only good near 25 °C and degrades catastrophically in the cold (NMC: 22 % RMSE
at −15 °C). The takeaway is not "which filter" but **"feed the filter a cell
temperature"** — every recursive estimator here (`ekf`, `ukf`, `joint_ekf`)
accepts a per-sample temperature; the supervisor already passes it. LFP is worse
in absolute terms at every temperature (flat OCV → weak voltage feedback), which
again points at current sensing rather than filter choice.

### Choosing

| If you… | Use |
|---|---|
| Trust the current sensor and know SoC₀ | `coulomb` (cheapest) |
| Have real sensor bias/drift and a sloped OCV | `ekf` (best SoC here) |
| Want SoC **and** online SoH from one filter | `joint_ekf` (the default) |
| Operate away from 25 °C | any recursive filter **fed the cell temperature** (naive filters lose 10–20 % RMSE in the cold) |
| Have a flat-OCV chemistry (LFP/LMO/SSB) | invest in current sensing; filter choice matters less |
| Have a trained data-driven model | `lstm`, or bring your own via `model_from_file` / `SklearnSocEstimator` / `OnnxSocEstimator` |

## Wiring notes

- The supervisor runs the joint-EKF each step when
  `SupervisorConfig.estimate_online=True` (the dashboard sets this). With
  `online_feeds_soh=True` (default) and `soh_aware=True`, that filter's SoH also
  drives the control-layer current derating — one source of truth for SoC, SoH,
  and the SoH-aware limits.
- With `estimate_online=False` (library default) the online-estimation fields are
  `nan` and behaviour is unchanged, so existing pipelines are unaffected.
