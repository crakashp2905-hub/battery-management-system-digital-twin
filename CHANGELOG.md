# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [0.5.0] - 2026-09-05

### Fixed
- **LSTM training reproducibility** — `LSTMEstimator.fit` shuffles with the
  instance's seeded RNG instead of the global NumPy state, so a fixed `seed`
  reproduces training exactly.
- **Heat generation** — corrected to `Q = |I·(OCV − V_t)|`, valid for both
  charge and discharge. The previous `I²R0 + I·max(OCV − V_t, 0)` double-counted
  the ohmic term on discharge and dropped all polarisation heat on charge.
- **Short-circuit detection** — added a current-based rule with trip authority.
  Previously short circuit had no rule and relied on the advisory ML layer,
  which cannot trip the contactor.
- **Supervisor fault evaluation** — uses the real per-cell currents (was
  hard-coded to zeros), and an optional `FaultInjector` is applied to the
  measured signals so injected faults actually reach the detector.
- **Peak power** — `peak_power_W` reports the deliverable power bounded by the
  min-voltage cutoff (consistent with `diagnostics.compute_crate_map`) instead
  of the unreachable matched-load `V_oc²/(4·R0)`.
- **Kalman filters** — SOC is no longer hard-clipped inside the EKF/UKF
  recursion (which biased the covariance); it is clipped only when reported.
- **Thermal solver** — the explicit-Euler step sub-steps internally to remain
  stable for any `dt` (CFL guard); `dt = 1 s` behaviour is unchanged.
- **NumPy compatibility** — `state_of_energy_Wh` falls back to `np.trapz` when
  `np.trapezoid` (NumPy ≥ 2.0) is unavailable.
- **OCV / filter consistency** — the EKF/UKF measurement models pass current
  into the OCV lookup so they match the plant when hysteresis is enabled.
- Documented the `soe_Wh` field of `BMSSupervisor.step`; removed dead
  hidden-state assignments in `LSTMEstimator.reset`.

### Added
- **Terminal SHUTDOWN + fault recovery** — thermal runaway latches to the
  terminal `SHUTDOWN` state (previously defined but unreachable), and
  `BMSSupervisor.clear_fault()` provides operator reset from the recoverable
  `FAULT` state.
- **Parallel-group circulating currents** — parallel cells share a common
  terminal node solved from KCL, so cells at different SOC exchange circulating
  currents instead of having their voltages averaged.
- **Battery Passport series capacity** — `nominal_capacity_Ah` uses the
  per-group cell capacity (`np.mean(capacities_Ah)`) instead of `capacities_Ah.sum()`,
  so equivalent full cycles (EFC) is physically accurate for series packs.
- **Switched-capacitor balancer directionality** — efficiency attenuation is
  consistently applied to the receiving cell regardless of transfer direction ($v_i > v_{i+1}$
  or $v_i < v_{i+1}$), eliminating unphysical charge creation.
- **Instant short-circuit trip** — short-circuit current spike trips contactor
  immediately to `FAULT` on step 0 without a 3-step debounce delay.
- **Cell under-voltage protection** — added `UNDERVOLTAGE` fault mode and rule
  guarding against over-discharge ($v_{cells} < v_{min}$ and $> v_{dropout}$).
- **Streamlit double-stepping resolved** — removed redundant manual `sup.pack.step`
  call and wired `FaultInjector` directly into `BMSSupervisor`.
- Packaging (`pyproject.toml`, pip-installable), GitHub Actions CI (test matrix
  on Python 3.10–3.13 + advisory ruff), `CONTRIBUTING.md`, and 14 regression
  tests (**169** total).

### Changed
- `heat_generation`, parallel-group physics, and `peak_power_W` change numeric
  outputs for multi-parallel packs and power reporting. Packs with one cell per
  series group at `dt = 1 s` are unaffected.

## [0.4.0]
- Seven chemistries, battery passport, DVA/ICA + EIS diagnostics, and the EV
  range predictor. (Baseline prior to this changelog.)
