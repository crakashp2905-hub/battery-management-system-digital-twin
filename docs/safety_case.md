# Safety case

A demo shows features work once; a *dependable* platform shows that every
serious hazard is analysed, mitigated, and **traced to a test that proves the
mitigation runs**. This page ties the three safety artefacts together.

## 1. FMEA — what can go wrong

`bms.build_fmea_table()` scores each failure mode on Severity × Occurrence ×
Detection (AIAG-VDA), giving a Risk Priority Number (RPN). The highest-RPN modes
(overcharge, internal/external short, thermal runaway, sensor faults, coolant
failure, firmware fault, ageing) are the ones the rest of the case must address.

## 2. Fault Tree Analysis — how the top hazard arises

`bms.thermal_runaway_tree()` (`bms/fta.py`) decomposes the top event **thermal
runaway** into basic events through AND/OR gates:

- **Internal short** (OR of manufacturing defect / dendrite / crush) — each a
  **single-event cut set**: it alone can cause runaway.
- **Uncontrolled overcharge** (AND) — needs *both* a charger overvoltage *and* a
  protection failure (voltage-sensor fault **or** welded contactor).
- **Sustained overtemperature** (AND) — coolant failure *and* a blind temp sensor.
- **Propagation from a neighbour** — a single-event cut once one cell ignites.

`minimal_cut_sets()` lists the smallest event combinations that cause the top
event; `probability()` propagates independent basic-event probabilities to the
top (≈ 7 × 10⁻⁵ in the reference tree). **Every basic event names the twin
detector that watches it** (`faults`, `mechanics`, `sensor_fdi`, `hv_safety`,
`propagation`), so the tree is not decoration — each leaf maps to code.

Render it with `bms.to_mermaid(bms.thermal_runaway_tree())`.

## 3. Traceability — every mitigation is tested

`bms.fmea_traceability()` and `scripts/traceability.py` cross-reference each
failure mode against the test suite (`FMEA_TEST_LINKS`). The rule enforced:

> **every failure mode with RPN ≥ 100 must have ≥ 1 covering test.**

```bash
python scripts/traceability.py            # matrix + PASS/FAIL (exit 1 on a gap)
python scripts/traceability.py --markdown  # matrix for docs
```

It exits non-zero if a high-RPN mode is untested, so it can run as a CI gate —
turning "we have tests" into "we can *prove* every serious hazard is exercised".

## Simulation-only vs safety-critical

This repository is a **simulation / algorithm-development twin**, not certified
firmware. The estimators, limits, and safety logic here are the *reference* to be
re-implemented and validated on a real BMS (see the foxBMS-style "control core"
path). The line is deliberate: nothing in this package should be deployed as the
sole safety path of a real pack.
