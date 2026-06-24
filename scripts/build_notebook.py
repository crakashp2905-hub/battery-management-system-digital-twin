"""Build the BMS_Digital_Twin_Demo.ipynb notebook programmatically.

Each cell is added via a small helper so the structure stays readable.
Run this from the repo root::

    python scripts/build_notebook.py

It writes ``notebooks/BMS_Digital_Twin_Demo.ipynb``. To also execute every
cell and embed the figures, follow up with::

    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/BMS_Digital_Twin_Demo.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ======================================================================
# 0. Title & contents
# ======================================================================
md(r"""
# BMS Digital Twin — End-to-End Demo

A research-grade Battery Management System (BMS) digital twin for a multi-cell Li-ion pack.
This notebook walks through every module of the package:

1. Second-order RC equivalent-circuit model (ECM) with OCV–SOC characteristic and parameter ID.
2. Multi-cell pack with manufacturing scatter + three balancing strategies.
3. Benchmarking framework for SOC estimators (CC, EKF, UKF, LSTM).
4. Finite-difference thermal model + PID-regulated cooling.
5. Fault injection and hybrid (rule + ML) detection.
6. FMEA + RUL.
7. Intelligent supervisory control.
8. Visualisation + reproducibility notes.

All figures are saved to `figures/` so they can be embedded in reports.
""")

code(r"""
# Imports & global setup
import sys, os, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")

# Make the package importable when running the notebook from notebooks/
ROOT = Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

import bms
print("bms version:", bms.__version__)
print("Public API:", len(bms.__all__), "symbols")
""")

# ======================================================================
# 1. ECM + OCV-SOC
# ======================================================================
md(r"""
## 1. Equivalent-Circuit Model and OCV-SOC characteristic

The cell is modeled as

$$V_t(k) = \mathrm{OCV}(\mathrm{SOC}(k)) \;-\; V_{\mathrm{RC}_1}(k) \;-\; V_{\mathrm{RC}_2}(k) \;-\; R_0\, i(k)$$

with two RC pairs capturing fast (charge transfer) and slow (diffusion) dynamics.
The OCV–SOC curve is a PCHIP interpolant fit to typical NMC 18650 datapoints.
""")

code(r"""
from bms import OCVSOC, SecondOrderECM, ECMParameters, fit_ecm_parameters

ocv = OCVSOC()
soc_grid = np.linspace(0, 1, 200)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
axes[0].plot(soc_grid, ocv.ocv(soc_grid))
axes[0].scatter(ocv.table[:, 0], ocv.table[:, 1], c="C3", s=20, zorder=3, label="anchor pts")
axes[0].set(xlabel="SOC [-]", ylabel="OCV [V]", title="OCV–SOC characteristic")
axes[0].legend()
axes[1].plot(soc_grid, ocv.docv_dsoc(soc_grid), color="C2")
axes[1].set(xlabel="SOC [-]", ylabel="dOCV/dSOC [V]", title="Slope (used by EKF)")
fig.tight_layout(); fig.savefig(FIG_DIR / "01_ocv_soc.png", bbox_inches="tight")
plt.show()
""")

md(r"""
### 1.1 Forward simulation: 1 C discharge + rest

This validates the dynamic response qualitatively — voltage relaxation after a load step is the
characteristic signature of the two RC pairs.
""")

code(r"""
true_p = ECMParameters(R0=0.030, R1=0.012, C1=2_500, R2=0.025, C2=10_000, Q_nom_Ah=2.3)
ecm = SecondOrderECM(params=true_p, ocv_curve=ocv); ecm.reset(0.95)

dt = 1.0
n = 3600
i = np.zeros(n)
i[200:1500] = 2.3   # ~1 C discharge for ~22 min
out = ecm.simulate(i, dt, soc0=0.95)

fig, axes = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True)
t = np.arange(n) * dt / 60
axes[0].plot(t, out["v_terminal"], color="C0"); axes[0].set_ylabel("V_term [V]")
axes[1].plot(t, out["soc"], color="C1");        axes[1].set_ylabel("SOC [-]")
axes[2].plot(t, i, color="C3");                 axes[2].set(ylabel="i [A]", xlabel="time [min]")
fig.suptitle("Second-order RC ECM — 1C discharge then rest")
fig.tight_layout(); fig.savefig(FIG_DIR / "02_ecm_response.png", bbox_inches="tight")
plt.show()
""")

md(r"""
### 1.2 Parameter identification on a noisy I/V trace

We run the model with known parameters, add 5 mV Gaussian sensor noise, and recover the
parameters via `scipy.optimize.least_squares`. The voltage RMSE — the only quantity we
actually validate against in practice — should fall close to the noise level.

A *single* discharge+rest does not excite both RC pairs equally well, so the individual R/C
values can drift; using an HPPC-style profile dramatically improves identifiability.
""")

code(r"""
rng = np.random.default_rng(0)
v_meas = out["v_terminal"] + rng.normal(0, 0.005, n)
fitted, info = fit_ecm_parameters(i, v_meas, dt=dt, Q_nom_Ah=2.3, ocv_curve=ocv)

# Re-simulate with fitted parameters
ecm_fit = SecondOrderECM(params=fitted, ocv_curve=ocv); ecm_fit.reset(info["soc0"])
out_fit = ecm_fit.simulate(i, dt)

fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(t, v_meas, color="C0", alpha=0.4, label="measured (noisy)")
ax.plot(t, out_fit["v_terminal"], color="C3", label="fitted ECM")
ax.set(xlabel="time [min]", ylabel="V_term [V]", title=f"Voltage fit (RMSE = {info['rmse_v']*1000:.1f} mV)")
ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "03_ecm_fit.png", bbox_inches="tight")
plt.show()

print("True vs fitted parameters:")
print(pd.DataFrame({
    "true":   [true_p.R0, true_p.R1, true_p.C1, true_p.R2, true_p.C2],
    "fitted": [fitted.R0, fitted.R1, fitted.C1, fitted.R2, fitted.C2],
}, index=["R0", "R1", "C1", "R2", "C2"]).round(4))
""")

# ======================================================================
# 2. Pack + balancing
# ======================================================================
md(r"""
## 2. Multi-cell pack and balancing strategies

The pack is a series stack of `N` cells; manufacturing scatter (`σ_capacity = 3 %`,
`σ_R0 = 5 %`) and initial-SOC scatter create realistic imbalance.

We compare three strategies on an idle pack — same initial state, no load — and measure:
- **Final imbalance** (max − min SOC) after a fixed time window.
- **Time to balance** to within 0.5 % SOC.
- **Energy lost** to dissipation or switching.
""")

code(r"""
from bms import BatteryPack, PackConfig, compare_balancers

factory = lambda: BatteryPack(PackConfig(n_cells=4, initial_soc_sigma=0.10, seed=7))
p0 = factory()
print("Initial SOC:", np.round(p0.soc, 4), "  imbalance =", round(p0.soc_imbalance(), 4))

bal_df = compare_balancers(factory, duration_s=3*3600, dt=2.0, pack_current_A=0.0)
display(bal_df)
""")

code(r"""
# Plot SOC convergence
hist = bal_df.attrs["history"]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
for ax, name in zip(axes, hist.keys()):
    soc = hist[name]["soc"]
    t_h = np.arange(soc.shape[0]) * 2 / 3600  # dt=2s
    for c in range(soc.shape[1]):
        ax.plot(t_h, soc[:, c], lw=1.2, label=f"cell {c}")
    ax.set(title=name, xlabel="time [h]")
    ax.legend(fontsize=8, ncol=2)
axes[0].set_ylabel("SOC [-]")
fig.tight_layout(); fig.savefig(FIG_DIR / "04_balancing_soc.png", bbox_inches="tight")
plt.show()

# Bar chart: efficiency comparison
fig, ax = plt.subplots(figsize=(6.5, 3.3))
ax.bar(bal_df.index, bal_df["energy_loss_J"], color=["C3", "C0", "C2"])
ax.set_ylabel("Energy lost [J]")
ax.set_title("Balancing energy loss — same initial scatter")
for i, v in enumerate(bal_df["energy_loss_J"]):
    ax.text(i, v, f"{v:,.0f} J", ha="center", va="bottom", fontsize=9)
fig.tight_layout(); fig.savefig(FIG_DIR / "05_balancing_loss.png", bbox_inches="tight")
plt.show()
""")

md(r"""
**Observations.** Passive resistive balancing is the fastest of the three at large initial
imbalance because the dissipation power is `V²/R` and unaffected by ΔV — but it dumps
~50–100× more energy than the active strategies. Inductor-based balancing has the smallest
loss and converges essentially exactly. Switched-capacitor balancing slows dramatically as
ΔV → 0 because the per-cycle charge transfer scales linearly with ΔV; on this idle scenario
it does not reach the 0.5 % threshold within 3 h.
""")

# ======================================================================
# 3. SOC estimator benchmark
# ======================================================================
md(r"""
## 3. SOC estimator benchmark

We feed the same noisy I/V trace into four estimators:

| Estimator             | Type                       | Notes |
|-----------------------|----------------------------|-------|
| Coulomb counting      | Open-loop integrator       | Fast, no observability — drifts with current bias. |
| EKF                   | Linearised Kalman          | Uses dOCV/dSOC as the measurement Jacobian. |
| UKF                   | Unscented Kalman           | Sigma-point propagation; handles nonlinearity exactly to 2nd order. |
| LSTM (NumPy, BPTT)    | Data-driven                | Single-layer recurrent net trained on synthetic ECM trajectories. |

Each is run with a deliberately *biased* current sensor (+50 mA) and 5 mV / 20 mV voltage
noise to expose the difference between open-loop and closed-loop estimation.
""")

code(r"""
from bms import (CoulombCounter, EKFEstimator, UKFEstimator, LSTMEstimator,
                 benchmark_estimators)

# 1) Truth trace
i_true = np.zeros(3600)
i_true[200:1200] = 1.5
i_true[1500:2500] = 2.5
i_true[2800:3500] = 1.0
m = SecondOrderECM(params=true_p, ocv_curve=ocv)
out_truth = m.simulate(i_true, dt=1.0, soc0=0.95)
v_true, soc_true = out_truth["v_terminal"], out_truth["soc"]
i_meas = i_true + 0.05  # +50 mA current sensor bias

# 2) Train the LSTM on a small synthetic set
print("Training NumPy LSTM (this takes ~12 s)…")
n_seq, T = 8, 500
rng = np.random.default_rng(42)
X, Y = np.zeros((n_seq, T, 3)), np.zeros((n_seq, T))
for k in range(n_seq):
    soc0 = float(rng.uniform(0.4, 0.95))
    ii = np.zeros(T)
    for _ in range(rng.integers(2, 5)):
        s = rng.integers(0, T - 100); L = rng.integers(50, 150)
        ii[s:s + L] = rng.uniform(0.5, 2.5)
    mm = SecondOrderECM(params=true_p, ocv_curve=ocv); mm.reset(soc0)
    oo = mm.simulate(ii, 1.0)
    X[k, :, 0] = oo["v_terminal"]; X[k, :, 1] = ii; X[k, :, 2] = soc0
    Y[k] = oo["soc"]

lstm = LSTMEstimator(input_size=3, hidden_size=10, seed=0)
hist_loss = lstm.fit(X, Y, epochs=15, lr=0.02)
print(f"  final training loss: {hist_loss[-1]:.5f}")

# 3) Benchmark
estimators = {
    "coulomb_counting": CoulombCounter(true_p.Q_nom_Ah, soc0=0.92),
    "ekf":              EKFEstimator(params=true_p, ocv_curve=ocv),
    "ukf":              UKFEstimator(params=true_p, ocv_curve=ocv, dt=1.0),
    "lstm":             lstm,
}
bench_df = benchmark_estimators(estimators, i_meas, v_true, soc_true,
                                dt=1.0, soc0=0.92,
                                noise_levels_v=(0.0, 0.005, 0.020))
display(bench_df.round(5))
""")

code(r"""
# Plot SOC traces under the 5 mV noise condition
rng = np.random.default_rng(1)
v_n = v_true + rng.normal(0, 0.005, len(v_true))

soc_estimates = {}
for name, est in estimators.items():
    try: est.reset(0.92)
    except TypeError: pass
    if isinstance(est, LSTMEstimator):
        soc_estimates[name] = est.run(i_meas, v_n, 1.0, soc0=0.92)
    else:
        soc_estimates[name] = est.run(i_meas, v_n, 1.0)

fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
t = np.arange(len(soc_true)) * 1.0
axes[0].plot(t, soc_true, "k--", lw=1.6, label="truth")
for name, soc_hat in soc_estimates.items():
    axes[0].plot(t, soc_hat, lw=1.1, alpha=0.85, label=name)
axes[0].set_ylabel("SOC"); axes[0].legend(loc="upper right", ncol=5, fontsize=9)
axes[0].set_title("SOC estimation under +50 mA current bias and 5 mV voltage noise")

for name, soc_hat in soc_estimates.items():
    axes[1].plot(t, (soc_hat - soc_true) * 100, lw=1.0, label=name)
axes[1].axhline(0, color="k", lw=0.6); axes[1].set(xlabel="time [s]", ylabel="error [% SOC]")
axes[1].legend(loc="upper right", ncol=5, fontsize=9)
fig.tight_layout(); fig.savefig(FIG_DIR / "06_soc_estimators.png", bbox_inches="tight")
plt.show()
""")

code(r"""
# RMSE bar chart at each noise level
fig, ax = plt.subplots(figsize=(8, 3.6))
bench_pivot = bench_df.pivot(index="estimator", columns="noise_v_sigma", values="rmse") * 100
bench_pivot.plot(kind="bar", ax=ax)
ax.set_ylabel("RMSE [% SOC]"); ax.set_xlabel("")
ax.set_title("SOC estimator robustness vs voltage-sensor noise σ")
ax.legend(title="σ_V [V]", fontsize=8)
plt.xticks(rotation=20)
fig.tight_layout(); fig.savefig(FIG_DIR / "07_soc_rmse_bars.png", bbox_inches="tight")
plt.show()
""")

md(r"""
**Observations.** Coulomb counting drifts monotonically because the +50 mA current bias
integrates without bound — by the end of the trace it is off by ~5 % SOC. EKF and UKF both
correct that drift via the OCV–SOC observation; their errors stay below ~0.3 % over the
whole trace. The LSTM trained on a tiny synthetic set is competitive at low noise but its
robustness to voltage noise depends entirely on what was in its training distribution. The
takeaway is the standard one in the literature: model-based filters dominate when a good
ECM is available; data-driven estimators shine when the model is poor or unavailable.
""")

# ======================================================================
# 4. Thermal + PID
# ======================================================================
md(r"""
## 4. Finite-difference thermal model and PID regulation

The pack is treated as a 1-D rod of `N` nodes with cell-to-cell conduction (`κ`),
convective dissipation `h_eff(t)·(T - T_amb)` modulated by the PID-controlled cooling
duty cycle, and per-cell heat generation
`Q_gen,i = i²·R₀,i + i·max(OCV - V_t, 0)`.

We drive the pack with a 3 C discharge for 30 minutes and watch the PID hold the hottest
cell at the 32 °C setpoint.
""")

code(r"""
from bms import ThermalModel, ThermalParameters, PIDController

n_cells = 4
tm = ThermalModel(n_cells=n_cells, params=ThermalParameters(T_amb_C=25.0,
                                                            h_min_W_per_K=0.15,
                                                            h_max_W_per_K=4.0))
tm.reset(25.0)
pid = PIDController(kp=0.18, ki=0.012, kd=0.4, setpoint=32.0, out_max=1.0)

# 4-cell pack with one slightly higher-resistance cell to create a hotspot
pack = BatteryPack(PackConfig(n_cells=n_cells, seed=11, r0_sigma=0.20))
i_load = np.full(1800, 6.9)  # 3C ≈ 6.9 A — aggressive enough that PID must engage

T_hist  = np.zeros((len(i_load), n_cells))
duty_h  = np.zeros(len(i_load))
for k, ii in enumerate(i_load):
    step = pack.step(float(ii), 1.0)
    ocv_v = np.array([float(pack.ocv_curve.ocv(c.soc)) for c in pack.cells])
    R0    = np.array([c.params.R0 for c in pack.cells])
    heat  = ThermalModel.heat_generation(np.full(n_cells, float(ii)), R0, step["v_cells"], ocv_v)
    duty  = pid.step(float(tm.T.max()), 1.0)
    tm.step(heat, duty, dt=1.0)
    T_hist[k] = tm.T
    duty_h[k] = duty

fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
t = np.arange(len(i_load)) / 60
for c in range(n_cells):
    axes[0].plot(t, T_hist[:, c], lw=1.1, label=f"cell {c}")
axes[0].axhline(32, color="k", ls="--", lw=0.8, label="setpoint")
axes[0].set_ylabel("Temperature [°C]"); axes[0].set_title("Per-cell temperature under 3 C discharge with PID cooling (setpoint = 32 °C)")
axes[0].legend(ncol=5, fontsize=8)

axes[1].plot(t, duty_h, color="C2"); axes[1].set(ylabel="cooling duty", xlabel="time [min]", ylim=(-0.05, 1.05))
fig.tight_layout(); fig.savefig(FIG_DIR / "08_thermal_pid.png", bbox_inches="tight")
plt.show()

print(f"Final spread (max-min): {T_hist[-1].max() - T_hist[-1].min():.2f} °C")
print(f"Peak temperature      : {T_hist.max():.2f} °C")
""")

md(r"""
The hotspot cell heats fastest, the PID ramps up cooling duty until equilibrium, and the
inter-cell spread settles below 2 °C — typical of a pack with adequate cooling. Persistent
spreads above ~5 °C are a known accelerator of ageing-asymmetry between cells.
""")

# ======================================================================
# 5. Faults — injection + hybrid detection
# ======================================================================
md(r"""
## 5. Fault injection and hybrid (rule + ML) detection

Five failure modes are simulated:
**OVERCHARGE** (forced charge current), **SHORT_CIRCUIT** (large extra discharge),
**THERMAL_RUNAWAY** (pre-warm cell + super-linear self-heating beyond 70 °C),
**SENSOR_DROPOUT** (V → 0), **SENSOR_BIAS** (slow voltage drift).

The detector combines a rule layer (deterministic safety net) with a Random-Forest classifier
trained on a fast-to-generate synthetic feature dataset. We OR the two so rules cover labels
the ML model never saw and the ML covers subtle drift faults that lie within nominal voltage
limits.

**Trip authority is rule-only.** Per functional-safety convention, only the deterministic rule
layer can cause the supervisor to open the contactor. ML detections are surfaced as alarms but
treated as warnings — this prevents misclassifications from causing nuisance shutdowns.
""")

code(r"""
from bms import HybridFaultDetector
from bms._train_detector import generate_fault_training_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

t0 = time.time()
X, y = generate_fault_training_data(samples_per_class=200, seed=0)
print(f"Generated {len(X)} samples in {time.time() - t0:.1f}s")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
det = HybridFaultDetector(n_trees=80, ml_min_confidence=0.65)
det.fit(Xtr, ytr)
print("\nClassifier on held-out data:\n")
print(classification_report(yte, det.clf.predict(Xte)))
""")

code(r"""
from bms import FaultInjector, FaultSpec, FaultMode, BMSSupervisor, SupervisorConfig

# Build a fresh supervisor and inject a thermal-runaway scenario
pack2 = BatteryPack(PackConfig(n_cells=4, seed=3))
thermal2 = ThermalModel(n_cells=4)
sup = BMSSupervisor(pack2, thermal2, det, config=SupervisorConfig(T_setpoint_C=35.0))

inj = FaultInjector([
    FaultSpec(mode=FaultMode.THERMAL_RUNAWAY, start_step=120, end_step=400,
              cell_index=2, severity=1.0)
])

n_steps = 400
log = {k: [] for k in ["t", "soc0", "T_hot", "state", "label", "src", "duty"]}
for k in range(n_steps):
    if k == 119:
        thermal2.T[2] = 75.0     # initial trigger
    # Sustain a small heat injection while the fault is "active" — without
    # this, dissipation cools the cell below 70 °C in one step and the rule
    # detector only fires once. Real runaway events are sustained.
    if 120 <= k < 200:
        thermal2.T[2] = max(thermal2.T[2], 72.0)
    thermal2.T[:] = inj.apply_to_temperatures(thermal2.T, k, 1.0)

    out = sup.step(requested_pack_current_A=2.3, dt=1.0, k=k)
    log["t"].append(k)
    log["soc0"].append(out["soc"][0])
    log["T_hot"].append(out["T_cells"].max())
    log["state"].append(out["state"])
    log["label"].append(out["fault_label"])
    log["src"].append(out["fault_source"])
    log["duty"].append(out["cooling_duty"])

log_df = pd.DataFrame(log)
display(log_df.iloc[110:140])
""")

code(r"""
fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
axes[0].plot(log_df["t"], log_df["T_hot"], color="C3"); axes[0].axhline(70, ls="--", color="k", lw=0.8, label="runaway thresh")
axes[0].set(ylabel="hottest T [°C]", title="Thermal-runaway scenario — supervisor reaction"); axes[0].legend()

# Mark fault detections
for k, src, lbl in zip(log_df["t"], log_df["src"], log_df["label"]):
    if src == "rule":
        axes[0].axvline(k, color="C3", lw=0.5, alpha=0.4)
    elif src == "ml":
        axes[0].axvline(k, color="C2", lw=0.5, alpha=0.4)
state_codes = {"idle": 0, "operating": 1, "balancing": 2, "fault": 3, "shutdown": 4}
axes[1].plot(log_df["t"], [state_codes[s] for s in log_df["state"]], drawstyle="steps-post", color="C0")
axes[1].set(xlabel="step", ylabel="state", yticks=list(state_codes.values()),
            yticklabels=list(state_codes.keys()))
fig.tight_layout(); fig.savefig(FIG_DIR / "09_fault_supervisor.png", bbox_inches="tight")
plt.show()

print("Detection sources during the runaway window:")
print(log_df.iloc[120:200]["src"].value_counts())
""")

# ======================================================================
# 6. FMEA + RUL
# ======================================================================
md(r"""
## 6. FMEA and Remaining-Useful-Life

The FMEA table follows AIAG-VDA conventions: each failure mode is scored on **Severity**,
**Occurrence**, and **Detection** (1–10). The Risk Priority Number `RPN = S·O·D` orders
the modes by overall risk and helps prioritise mitigation work.
""")

code(r"""
from bms import build_fmea_table, estimate_rul, load_nasa_like_dataset
fmea = build_fmea_table()
display(fmea)

fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(fmea["failure_mode"], fmea["RPN"], color="C3")
ax.invert_yaxis(); ax.set_xlabel("Risk Priority Number (RPN)")
ax.set_title("FMEA — failure modes ranked by RPN")
fig.tight_layout(); fig.savefig(FIG_DIR / "10_fmea.png", bbox_inches="tight")
plt.show()
""")

code(r"""
# RUL: square-root-of-cycles capacity fade
ageing = load_nasa_like_dataset(cycles=80, capacity_Ah=2.3, seed=3)
rul = estimate_rul(ageing["cycle"].values, ageing["capacity_Ah"].values,
                   nominal_capacity_Ah=2.3)
print(rul)

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.scatter(ageing["cycle"], ageing["capacity_Ah"], s=14, label="observed")
N = np.linspace(0, rul["cycles_to_eol"] * 1.05, 400)
Q_pred = 2.3 * (1 - rul["alpha"] * np.sqrt(N))
ax.plot(N, Q_pred, "C3-", label=f"fit  (α={rul['alpha']:.4f})")
ax.axhline(0.8 * 2.3, color="k", ls="--", lw=0.8, label="80% EoL threshold")
ax.axvline(rul["cycles_to_eol"], color="C2", ls=":", label=f"EoL @ {rul['cycles_to_eol']:.0f} cycles")
ax.set(xlabel="cycle number", ylabel="capacity [Ah]",
       title=f"Capacity-fade RUL  •  SoH={rul['soh']*100:.1f}%, RUL={rul['rul_cycles']:.0f} cycles")
ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "11_rul.png", bbox_inches="tight")
plt.show()
""")

# ======================================================================
# 7. End-to-end scenario
# ======================================================================
md(r"""
## 7. End-to-end scenario: load + imbalance + fault

We tie everything together in a single 20-minute simulated drive cycle. The supervisor
selects a balancing strategy based on imbalance, runs the PID, and the detector flags a
sensor-bias fault injected at minute 8. Outputs are the live signals the Streamlit dashboard
would display in real time.
""")

code(r"""
from bms import generate_load_profile

pack3 = BatteryPack(PackConfig(n_cells=4, seed=21, initial_soc_sigma=0.07))
thermal3 = ThermalModel(n_cells=4)
sup3 = BMSSupervisor(pack3, thermal3, det)

duration = 1200
i_load = generate_load_profile(duration, dt=1.0, mode="drive", c_rate=1.0,
                               capacity_Ah=2.3, seed=2)
inj3 = FaultInjector([
    FaultSpec(mode=FaultMode.SENSOR_BIAS, start_step=480, end_step=1000,
              cell_index=1, severity=1.2)
])

records = []
for k in range(duration):
    cell_currents = np.full(pack3.n_cells, float(i_load[k]))
    cell_currents = inj3.apply_to_currents(cell_currents, k)
    extra = cell_currents - float(i_load[k])
    pack3.step(float(i_load[k]), 1.0, balancing_currents=extra)

    out = sup3.step(float(i_load[k]), 1.0, k=k)
    records.append({
        "t": k, "v_pack": out["v_pack"], "soc_min": out["soc"].min(),
        "soc_max": out["soc"].max(), "T_max": out["T_cells"].max(),
        "duty": out["cooling_duty"], "balancer": out["balancer"],
        "state": out["state"], "label": out["fault_label"], "src": out["fault_source"],
    })

scen = pd.DataFrame(records)
print("State distribution over the run:"); print(scen["state"].value_counts())
print("\nBalancer activity:"); print(scen["balancer"].value_counts())
print(f"\nFault alarms: {(scen['label'] != 'none').sum()} steps")
""")

code(r"""
fig, axes = plt.subplots(4, 1, figsize=(10, 8.4), sharex=True)
axes[0].plot(scen["t"], scen["v_pack"], color="C0"); axes[0].set_ylabel("V_pack [V]")
axes[1].plot(scen["t"], scen["soc_max"], label="max SOC", color="C2")
axes[1].plot(scen["t"], scen["soc_min"], label="min SOC", color="C1")
axes[1].fill_between(scen["t"], scen["soc_min"], scen["soc_max"], alpha=0.15, color="C2")
axes[1].set_ylabel("SOC"); axes[1].legend()
axes[2].plot(scen["t"], scen["T_max"], color="C3"); axes[2].set_ylabel("T_max [°C]")
axes[3].plot(scen["t"], scen["duty"], color="C4"); axes[3].set(xlabel="time [s]", ylabel="cooling duty")

# Mark ML alarms with light bands; rule alarms with bold lines
for k, src in zip(scen["t"], scen["src"]):
    if src == "ml":
        axes[0].axvspan(k - 0.5, k + 0.5, color="C2", alpha=0.06)
    elif src == "rule":
        for ax in axes:
            ax.axvline(k, color="C3", lw=0.6, alpha=0.5)

axes[0].set_title("End-to-end scenario — drive cycle + sensor-bias fault (green = ML alarm, red = rule alarm)")
fig.tight_layout(); fig.savefig(FIG_DIR / "12_end_to_end.png", bbox_inches="tight")
plt.show()
""")

# ======================================================================
# 8. Reproducibility
# ======================================================================
md(r"""
## 8. Reproducibility & next steps

**Determinism.** Every random source is seeded (`PackConfig.seed`, `np.random.default_rng(seed)`,
`generate_load_profile(..., seed=…)`, `RandomForestClassifier(random_state=…)`,
`LSTMEstimator(seed=…)`). Re-running this notebook reproduces every figure bit-for-bit.

**Streamlit dashboard.** Run `streamlit run app/streamlit_app.py` from the repo root to see
a live version of the figures above with interactive controls.

**Tests.** `pytest tests/` runs unit tests on the OCV curve, ECM round-trip, balancing
energy bookkeeping, EKF/UKF convergence, FDM thermal stability, and FMEA RPN ordering.

### Optional extensions implemented
- **Predictive maintenance / RUL** — see Section 6.
- **Hybrid AI control** — Section 7's supervisor selects strategies dynamically.

### Optional extensions left as further work
- **Reinforcement learning** for adaptive cooling-duty policies (the PID could be replaced
  by a small `gymnasium`-style env with the supervisor as the agent).
- **Cloud / Databricks integration** — every signal is already in a tidy DataFrame; emitting
  it to an MQTT broker or a Delta table is a one-line change.
""")


# ======================================================================
# Write notebook
# ======================================================================
nb["cells"] = cells
ROOT = Path(__file__).resolve().parents[1]
out_path = ROOT / "notebooks" / "BMS_Digital_Twin_Demo.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out_path)
print(f"Wrote {out_path.relative_to(ROOT)}  ({len(cells)} cells)")
