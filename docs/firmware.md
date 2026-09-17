# From twin to firmware — the control core

This repository is a **simulation / algorithm-development twin** (NumPy, SciPy,
pandas). Real BMS firmware runs on an MCU and can't use any of that. `bms.control_core`
is the bridge: the deterministic algorithms a real BMS executes each control
cycle, written so they **transliterate to embedded C**, with the twin as their
software-in-the-loop test bench.

## What the core contains (`bms/control_core.py`)

- A **1-state SoC EKF** with the RC over-potentials as deterministic feed-forward.
- **State-of-Power** limits (discharge/charge current & power, voltage-window bounded).
- **Safety threshold checks** as a fault bitmask (over-voltage / under-voltage /
  over-temperature / over-current) with a contactor-open command.

It uses **only `math`** — fixed-size state (a `CoreState` struct), a literal OCV
lookup table with linear interpolation (no PCHIP/SciPy), no dynamic allocation,
no library calls in the hot path. `core_step(state, params, v, i, T, dt)` is the
single control-cycle entry point; each line maps to C.

## The twin as SIL oracle

```python
from bms.control_core import CoreParams, run_sil
# currents/voltages/temperatures from the twin plant (or a data replay)
res = run_sil(CoreParams(...), currents, voltages, temps, dt, soc_truth=plant_soc)
res["soc_rmse"]      # the C-portable core vs the twin's true SoC
res["flags_seen"]    # which safety flags fired
```

In the reference bench the core tracks the twin's SoC to **~0.003 RMSE** and its
safety flags fire on injected over-temp / over-voltage / over-current — the exact
regression you run before, and after, porting to C.

## The boundary

- **Simulation-only** (this package at large): the physics plant, the ML fault
  detector, the LLM agent, the electrochemical SPM, the analytics. High fidelity,
  not real-time, not safety-rated.
- **Firmware-portable** (`control_core`): the deterministic estimator + limits +
  threshold safety. This is the part you would re-implement in C, verified against
  the simulation-only twin.

Nothing here is certified firmware or a substitute for a hardware safety path
(see `docs/safety_case.md`); `control_core` is the *reference algorithm* to port
and re-validate on the target, in the spirit of the foxBMS-style control core.
