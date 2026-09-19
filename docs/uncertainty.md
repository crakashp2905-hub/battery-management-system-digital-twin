# Uncertainty calibration

A twin that reports `SoC = 0.71 ± 0.02` is only trustworthy if that interval is
**calibrated** — over many predictions the truth should land inside the 95 %
interval about 95 % of the time. An unchecked confidence number is decoration.
`bms.uncertainty` makes it checkable.

## Metrics

| Function | Answers |
|---|---|
| `coverage(y, mean, sigma, level)` | what fraction of truths fall in the `level` interval? |
| `picp(y, lower, upper)` / `mpiw` | coverage and mean width of explicit intervals |
| `reliability_curve(y, mean, sigma)` | observed vs nominal coverage across levels (reliability-diagram data) |
| `expected_calibration_error(...)` | average gap between nominal and observed (0 = perfect) |
| `sharpness(sigma)` | mean interval width — tight is good *given* calibration |
| `assess_calibration(...)` | a one-call verdict: calibrated / overconfident / underconfident |
| `calibration_scale(...)` | the σ-correction factor that makes the interval nominal |

The verdict is based on the **signed** miscalibration averaged over the whole
reliability curve, which is robust where 95 % coverage alone saturates at 1.0.

## What it found

Applied to the twin's own SoC intervals over a synthetic ensemble with known
ground truth, the reported σ (≈ 0.012) is far larger than the actual error RMSE
(≈ 0.0006): coverage is 1.00 (not 0.95) and the verdict is **underconfident** —
the joint-EKF covariance is over-cautious on a well-matched cell. That is an
honest finding, not a hidden one, and it is exactly the point of measuring
calibration rather than asserting it.

`calibration_scale` then returns the multiplicative correction that brings the
95 % interval back to nominal — measure, then fix:

```python
c = bms.calibration_scale(soc_true, soc_est, soc_sigma, level=0.95)
sigma_calibrated = c * soc_sigma        # coverage(soc_true, soc_est, sigma_calibrated) ≈ 0.95
```

## Why this matters

Distinguishing *measurement* noise, *parameter* uncertainty and *model* error is
only meaningful if the resulting intervals are honest. Calibrated uncertainty is
what lets the twin's confidence be used in a decision — e.g. gating an action, or
triggering an experiment when confidence is genuinely (not spuriously) low.
