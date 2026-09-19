# Universal auto-calibration & signal conditioning

The NASA validation (`docs/validation.md`) showed the point plainly: a *generic*
model is only so accurate on a *specific* cell. The generic NMC OCV mis-serves a
LiCoO₂ cell. The answer is not a bigger model — it is **calibrating to the cell in
front of you**, automatically, whatever its chemistry.

## Clean the data first — `bms.signal`

Raw telemetry is noisy and glitchy. `bms.signal` provides the robust cleaners
that matter for BMS data:

| Function | Use |
|---|---|
| `hampel(x, window, n_sigma)` | despike current/voltage (robust median/MAD; flags & repairs) |
| `median_filter` | impulse-noise rejection |
| `savitzky_golay` | smooth **without** distorting peak shape (safe before DVA/ICA) |
| `ewma`, `moving_average` | light low-pass |
| `clean_signal(x)` | recommended pipeline: Hampel despike → gentle Savitzky-Golay |

```python
res = bms.clean_signal(raw_current)      # res.signal, res.n_outliers, res.outlier_mask
```

## Calibrate to any cell — `bms.auto_calibrate`

Give it a raw `(current, voltage)` trace from a cell of **unknown chemistry**; it
returns a `CalibratedCell` ready to drive any estimator:

```python
cal = bms.auto_calibrate(current, voltage, dt)      # no chemistry hint needed
ekf = cal.make_estimator("ekf")                     # ready-tuned for THIS cell
```

How it works, and why it is robust:

1. **Capacity** — estimated from a full (dis)charge's throughput (or supplied).
2. **OCV template selection** — every shipped chemistry OCV is tried; for each,
   the ECM `(R0, R1, C1, R2, C2)` is fit by nonlinear least squares
   (Levenberg-Marquardt / trust-region), and the lowest-residual template wins.
3. **ECM identification** — the winning template's fitted parameters.

Selecting the closest *well-behaved* template and fitting the rest side-steps the
ill-posed problem of inverting an OCV curve from a single constant-current trace
— which, tried directly, fails badly (a 60 % SoC error in testing). Matching to a
template is stable and always converges.

### Validated on real data

Given a **real NASA PCoE discharge** (a LiCoO₂ cell) with **no chemistry hint**,
`auto_calibrate` selects the closest template (NCA/NMC) and fits the ECM; the
resulting EKF drops from the naive **~12.5 %** generic result to **~3.7 %** SoC
RMSE on that trace, and to **~1.5 %** on held-out cycles when calibrated on a
separate one. The same pipeline is what the self-calibrating twin
(`docs/autonomy.md`) runs continuously — here it is a one-shot, chemistry-agnostic
entry point.

> Note: this is honest engineering, not magic. Template-matching + ECM fitting
> gets a new cell to a good operating point automatically; the remaining error is
> dominated by OCV-template mismatch, which a cell-specific OCV (from proper
> low-rate or rest data) would close further.
