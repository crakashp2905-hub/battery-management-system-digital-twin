# Architecture

The digital twin is layered. Each layer depends only on the layer(s) below it, so any
single component can be replaced or instrumented without touching the others.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INTERFACE LAYER                                │
│   notebooks/BMS_Digital_Twin_Demo.ipynb  •  app/streamlit_app.py            │
└─────────────────────────────────────────────────────────────────────────────┘
                                     ▲
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CONTROL LAYER                                  │
│   bms.BMSSupervisor (state machine, strategy selection, PID, fault gating)  │
└─────────────────────────────────────────────────────────────────────────────┘
        ▲                            ▲                            ▲
┌──────────────┐           ┌────────────────────┐          ┌──────────────┐
│  ESTIMATION  │           │     BALANCING      │          │  DIAGNOSTICS │
│              │           │                    │          │              │
│ CoulombCounter            PassiveBalancer                   FaultInjector
│ EKFEstimator              SwitchedCapacitor…                HybridFault…
│ UKFEstimator              InductorBalancer                  build_fmea_table
│ LSTMEstimator                                               estimate_rul
└──────────────┘           └────────────────────┘          └──────────────┘
        ▲                            ▲                            ▲
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PHYSICAL LAYER                                 │
│   bms.OCVSOC  →  bms.SecondOrderECM  →  bms.BatteryPack  ↔  bms.ThermalModel│
└─────────────────────────────────────────────────────────────────────────────┘
```

## Object lifetimes

A typical run looks like:

```python
ocv      = OCVSOC()                                  # static once configured
ecm      = SecondOrderECM(params, ocv)               # one per cell
pack     = BatteryPack(config)                       # owns N ecms
thermal  = ThermalModel(n_cells)                     # parallel to pack
detector = HybridFaultDetector().fit(X, y)           # trained offline
sup      = BMSSupervisor(pack, thermal, detector)    # owns PID + balancers

while True:
    out = sup.step(requested_pack_current_A, dt, k=step_index)
```

The supervisor is the single point of contact between the user (or the dashboard, or a
real load profile) and the physical models. It returns a fully self-describing dict:

```python
{
    "state":      "operating",            # IDLE | OPERATING | BALANCING | FAULT | SHUTDOWN
    "fault_label": "none",                # one of FaultMode values
    "fault_source": "none",               # "rule" | "ml" | "none"
    "v_cells":     ndarray (n_cells,),
    "v_pack":      float,
    "soc":         ndarray (n_cells,),
    "T_cells":     ndarray (n_cells,),
    "cooling_duty":float in [0,1],
    "balancer":    str,                   # name of active strategy or "none"
    "balancing_currents": ndarray,
    "imbalance":   float,
    "cmd_current": float,                 # what was actually applied (zero in FAULT)
}
```

## State machine

```
                   no rule alarms
        ┌──────────────────────────────────┐
        │                                  │
        ▼                                  │
       IDLE ────────► BALANCING ──────► OPERATING
        ▲                ▲                  │
        │                │                  │ rule alarms ≥ N (default 3)
        │                │                  ▼
        │                │              FAULT
        │                │             (open contactor,
        │                │              max cooling duty)
        │                │                  │
        │                └──────────────────┘
        │            (manual reset by re-instantiating)
```

## Key invariants

1. **Pack series current**: every cell sees the same external `pack_current`; per-cell
   variation only enters via balancing currents.
2. **Energy bookkeeping** (balancers): `energy_loss_J` is monotonically non-decreasing.
3. **Trip authority**: only `fault_source == "rule"` increments the supervisor's alarm
   streak. ML alarms are always advisory.
4. **State validity**: SOC ∈ [0, 1] and cooling duty ∈ [0, 1] are clipped at every step.

## Extension points

| Want to | Edit |
|---|---|
| Use a different OCV curve | pass a `(N, 2)` table to `OCVSOC(table=…)` |
| Use a different cell model | subclass `SecondOrderECM` or substitute via duck-typing in `BatteryPack.cells` |
| Add a fourth balancing strategy | subclass `Balancer`, add to `BMSSupervisor._select_balancer` |
| Replace PID with RL | swap `PIDController.step` for any `(measured, dt) → duty` policy |
| Add new fault modes | extend `FaultMode` enum + `FaultInjector.apply_to_*` + `_train_detector` |
| Stream to cloud | add a callback in `BMSSupervisor.step` (e.g., MQTT publish) |

## Why this layering

The interface (notebook / Streamlit) and the physical models share **nothing** except the
supervisor's output contract. That means:

* The dashboard can be rewritten in a different framework without touching physics.
* A unit-test against `EKFEstimator` doesn't need `BatteryPack`.
* A real BMS firmware port can re-implement the supervisor in C without re-deriving the
  Kalman filter — the equations are documented in `bms/soc_estimators.py`.
