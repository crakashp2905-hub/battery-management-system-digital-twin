# The digital twin — one authoritative state

A pile of excellent battery algorithms is not a digital twin. A twin is **one
continuously calibrated belief** about the cell that live measurements keep in
sync with reality, and that is honest about how much it trusts itself.

[`bms/twin.py`](https://github.com/crakashp2905-hub/battery-management-system-digital-twin/blob/main/bms/twin.py)
provides that unifying object, `BatteryDigitalTwin`, on top of the engines the
repository already had:

| Concern | Engine | What the twin reads from it |
|---|---|---|
| SoC / capacity / SoH (probabilistic core) | `JointEKFSoH` (`soh_estimator.py`) | estimates **and their 1-σ** |
| Model-vs-measurement residual, drift | `TwinSync` (`twin_sync.py`) | `residual_rms_V`, `drift` |
| Online ohmic resistance | `RLSIdentifier` (`online_id.py`) | `r0_ohm` |

## One call, one snapshot

```python
import bms

twin = bms.BatteryDigitalTwin(params=bms.ECMParameters(Q_nom_Ah=2.3))
twin.reset(soc0=0.9)

for v, i, T in stream:                      # live (voltage, current, temperature)
    state = twin.update(v, i, dt=1.0, temperature_C=T)

print(state.to_dict())
# {'soc': 0.71, 'soc_95_ci': [0.69, 0.73],
#  'soh': 0.94, 'soh_95_ci': [0.88, 1.00],
#  'r0_ohm': 0.026, 'drift': False,
#  'capacity_observable': True, 'confidence': 0.83, ...}
```

`update` returns an immutable `TwinState`; the latest is also kept as
`twin.state`. `run(currents, voltages, dt, temperatures)` assimilates a whole
trace and returns the per-sample snapshots.

Every estimate ships with a **95 % credible interval** (`soc_95_ci`,
`soh_95_ci`), Gaussian from the filter's 1-σ and clipped to physical ranges — so
a consumer gets "here is the estimate *and how sure the twin is*", not a bare
number.

## Observability — is SoH even knowable right now?

Capacity is only identifiable when SoC actually moves: the *rate* of SoC change
is what reveals `Q`. A cell sitting at rest gives the joint EKF nothing to lock
onto, so a twin that keeps reporting a crisp SoH through a long rest is lying.

The twin measures `excitation` (the SoC swing over a trailing window) and sets
`capacity_observable`. During a rest this is `False` and the SoH confidence
collapses, exactly as it should.

## The confidence model

`confidence ∈ [0, 1]` (with per-state `soc_confidence` / `soh_confidence` and a
`confidence_breakdown`) fuses four signals, each mapped to `[0, 1]`:

| Signal | 1 (trust) when… | 0 (distrust) when… |
|---|---|---|
| **freshness** | warmed up (`n ≥ warmup_steps`) | just reset (cold start) |
| **residual** | model matches (`rms` ≪ drift threshold) | residual large / drifted |
| **soc / soh uncertainty** | 1-σ well below its reference | 1-σ at/above the reference |
| **excitation** | recent SoC swing ≥ `excitation_ref` | resting, no movement |

SoC confidence needs a converged filter **and** a matching model; SoH confidence
additionally needs excitation. The overall score weights SoC more heavily
because keeping SoC and operation trustworthy in real time is the twin's primary
job, while SoH is a slow background estimate that legitimately converges over a
cycle.

The payoff is the distinction a monitoring system actually needs:

> SoH 81.7 %, **confidence 0.94**

versus

> SoH 81.7 %, **confidence 0.41 — insufficient excitation for a reliable
> capacity estimate**

## Design notes

- The joint EKF is the estimator; `TwinSync` runs *alongside* it purely as an
  independent drift/health monitor (its Luenberger-corrected SoC is a
  cross-check, not the reported state). Feeding an **aged** cell to a
  fresh-parameter twin trips `drift` and depresses mean confidence — the twin
  knowing it needs recalibration, which a plain forward simulation cannot signal.
- `confidence` is a deliberately interpretable heuristic, not a calibrated
  probability. Every threshold (`excitation_ref`, `soc_sigma_ref`,
  `soh_sigma_ref`, `warmup_steps`) is a constructor argument, so a deployment can
  tune trust to its own duty cycle.
