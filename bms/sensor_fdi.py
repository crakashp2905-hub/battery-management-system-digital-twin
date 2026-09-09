"""Sensor fault detection & isolation (FDI) with model-based reconfiguration.

Distinct from *cell* faults: this watches the **sensors** themselves — voltage,
current, temperature channels — for the classic failure modes, and when a
channel drops out it can be replaced by a **virtual sensor** derived from the
ECM.  A real ISO 26262 requirement the cell-fault detector does not cover.

Failure modes detected per channel:

* ``dropout``      — NaN/inf or an implausible zero.
* ``out_of_range`` — outside the physically plausible band.
* ``rate``         — a step change larger than any real signal (spike/glitch).
* ``stuck``        — no change at all over a window (a frozen ADC / flat-lined
  channel), suspicious for a live signal that should carry noise.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class SensorMonitor:
    """Residual-based fault monitor for one sensor channel."""

    lo: float                       # plausible minimum
    hi: float                       # plausible maximum
    max_rate: float                 # max plausible |Δ| per update
    stuck_window: int = 25          # flat for this many samples → stuck
    stuck_tol: float = 1e-9         # "no change" tolerance
    _hist: deque = field(default_factory=lambda: deque(maxlen=64))

    def reset(self) -> None:
        self._hist.clear()

    def check(self, value: float) -> str:
        """Classify the latest sample: ``ok`` or a fault label."""
        if value is None or math.isnan(value) or math.isinf(value):
            return "dropout"
        v = float(value)
        label = "ok"
        if not (self.lo <= v <= self.hi):
            label = "out_of_range"
        elif self._hist and abs(v - self._hist[-1]) > self.max_rate:
            label = "rate"
        elif len(self._hist) >= self.stuck_window - 1:
            recent = list(self._hist)[-(self.stuck_window - 1):] + [v]
            if max(recent) - min(recent) <= self.stuck_tol:
                label = "stuck"
        self._hist.append(v)
        return label


@dataclass
class SensorFDI:
    """Bundle of channel monitors for a cell's V / I / T sensors.

    Construct with plausible bounds (usually from chemistry props), then call
    :meth:`check` each step with the measured triple; it returns a per-channel
    status dict.  When the voltage channel is faulted, substitute
    :func:`virtual_cell_voltage` to keep the estimator running.
    """

    voltage: SensorMonitor
    current: SensorMonitor
    temperature: SensorMonitor

    @classmethod
    def for_chemistry(cls, chemistry: str, max_current_A: float = 300.0) -> "SensorFDI":
        from .chemistry import get_chemistry_props
        p = get_chemistry_props(chemistry)
        return cls(
            voltage=SensorMonitor(lo=p["v_dropout"], hi=p["v_overcharge"] + 0.5,
                                  max_rate=1.0),
            current=SensorMonitor(lo=-max_current_A, hi=max_current_A, max_rate=max_current_A),
            temperature=SensorMonitor(lo=-40.0, hi=120.0, max_rate=15.0),
        )

    def check(self, voltage: float, current: float, temperature: float) -> dict:
        status = {"voltage": self.voltage.check(voltage),
                  "current": self.current.check(current),
                  "temperature": self.temperature.check(temperature)}
        status["any_fault"] = any(v != "ok" for v in status.values())
        return status


def virtual_cell_voltage(ecm_params, ocv_curve, soc: float, current: float,
                         v_rc: float = 0.0, temperature_C: float = 25.0) -> float:
    """Model-based virtual voltage sensor: OCV(SoC) − IR − V_RC.

    Replaces a dropped-out voltage measurement with the ECM's prediction, so a
    single sensor failure degrades gracefully instead of blinding the estimator.
    """
    p = ecm_params.at_temperature(temperature_C)
    ocv = float(ocv_curve.ocv(soc, current, T_C=temperature_C))
    return ocv - p.R0 * float(current) - float(v_rc)
