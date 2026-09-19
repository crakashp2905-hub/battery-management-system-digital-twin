# Autonomous Battery Intelligence

Most battery twins answer *"what is the battery doing?"*. This one is built to
answer a harder question: *"what don't I know, and how do I learn it fastest?"* —
and then to act on the answer without being told.

It is the closed loop the strategy review pointed at, assembled from four
components that each ship as their own module:

```text
                 REAL BATTERY  (or a higher-fidelity plant / SIL oracle)
                       │
              voltage / current / temp
                       ▼
             ┌────────────────────┐
             │  DIGITAL TWIN       │  bms/twin.py
             │  state + uncertainty│
             └─────────┬──────────┘
                       ▼
             ┌────────────────────┐
             │ OBSERVABILITY ENGINE│  bms/observability.py
             │  "what can't I see?"│  Fisher information / CRLB
             └─────────┬──────────┘
                       ▼
             ┌────────────────────┐
             │ EXPERIMENT DESIGNER │  bms/experiment_design.py
             │ "what probe helps   │  D-optimal / targeted, **safety-gated**
             │  most, safely?"     │
             └─────────┬──────────┘
                       ▼
                 SAFE EXCITATION ──────────────► run on the real cell
                       ▼
             ┌────────────────────┐
             │ SELF-CALIBRATION    │  bms/self_calibration.py
             │ fix the wrong params│  observability-gated + CV-gated
             └─────────┬──────────┘
                       ▼
             predictive fidelity improved ──────► repeat
```

`bms/autonomy.py` (`AutonomousBatteryTwin`) drives the loop.

## The research question

> Can a battery digital twin autonomously identify its own epistemic
> uncertainty, choose the experiments that reduce it most efficiently, update
> from them, and improve its predictive fidelity over time?

## What each round does

1. **Target its uncertainty.** It tracks the best Cramér–Rao bound any experiment
   has yet achieved for each parameter, and goes after the one still worst-pinned
   — dropping a parameter once targeting it stops helping, so it never chases an
   intrinsically hard-to-identify quantity (the slow `R2` branch) forever.
2. **Design a safe probe.** The experiment designer returns the excitation that
   most reduces that parameter's CRLB, after rejecting any candidate that would
   leave the voltage window or exceed a current limit.
3. **Perform it** on the real cell (here a hidden ground-truth plant).
4. **Self-calibrate**, gated two ways: observability decides *which* parameters
   the probe could identify, and a **held-out validation check** commits the
   update only if it does not worsen fidelity. Together these make the loop
   **monotone non-increasing in error — it cannot diverge** on an unlucky probe.

## Reference result

A twin that starts with **fresh-cell parameters** is shown a **hidden aged cell**
(`R0` +120 %, `Q` −12 %). Within a few rounds, and *without being told what was
wrong*:

| | cold start | after learning |
|---|---|---|
| held-out voltage RMSE | ~19 mV | **~0.7 mV** (sensor floor) |
| parameter error (RMS relative) | 0.39 | **0.13** |
| `R0` | 0.025 Ω (wrong) | **0.055 Ω** (truth 0.055) |

`degradation_report()` then attributes the change to its modes and correctly
names **resistance growth** as dominant (`+120 %`, vs the true `+120 %`).

Some parameters (the slow `R2` branch, and the `Q`/`R2` trade-off) remain only
partially identifiable — which the twin reports rather than hides. That honesty
*is* the point: the loop drives the **observable** behaviour to the noise floor
and tells you which internal parameters the data could not separate.

## A note on novelty

This composes standard, well-founded pieces — Fisher-information identifiability,
optimal experiment design, recursive parameter identification, cross-validated
model updates — into one autonomous loop for a battery twin. That combination and
framing is what makes it interesting, but (as the review stressed) **a concept
can be strongly differentiated without being unprecedented.** Any novelty claim
should be checked against the prior-art, patent and literature record before it
is made; treat the wording here as an engineering description, not a priority
claim.
