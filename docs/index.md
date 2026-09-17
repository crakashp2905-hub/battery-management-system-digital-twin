# BMS Digital Twin

A research-grade, **AI-augmented Battery Management System (BMS) digital twin**
for multi-cell Li-ion / solid-state packs. It unifies electrochemical modelling
(ECM **and** a Single Particle Model), thermal simulation, cell balancing,
**model-agnostic** state estimation (SoC **and** online SoH with uncertainty),
hybrid fault detection, charging-method physics and model-predictive charging,
dynamic aging with fleet reliability, SoH-aware and safety control, a
firmware-portable control core, and a plain-language interpretability layer —
into one reproducible, fully-tested framework.

## Start here

- **[Architecture](architecture.md)** — the layered design and invariants.
- **[Estimation & benchmarks](estimation.md)** — which model produces each
  quantity, and the SoC estimator benchmarks (chemistry, temperature, hysteresis,
  sensor bias, analog front-end).
- **[Validation](validation.md)** — the real-data validation harness and datasets.
- **[Safety case](safety_case.md)** — FMEA, Fault Tree Analysis, and
  FMEA→test traceability.
- **[Firmware & SIL](firmware.md)** — the portable control core and the twin as a
  software-in-the-loop test oracle.

## Install

```bash
pip install bms-digital-twin            # core
pip install "bms-digital-twin[app]"     # + Streamlit dashboard
pip install "bms-digital-twin[api]"     # + FastAPI live-twin service
```

## A one-liner

```python
import bms
pack = bms.BatteryPack(bms.PackConfig(n_cells=4))
sup = bms.BMSSupervisor(pack, bms.ThermalModel(n_cells=4), bms.HybridFaultDetector())
out = sup.step(requested_pack_current_A=2.0, dt=1.0)
print(out["soc"], out["v_pack"], out["state"])
```

See the project [README](https://github.com/crakashp2905-hub/battery-management-system-digital-twin)
for the full feature list, the interactive dashboard, and the changelog.
