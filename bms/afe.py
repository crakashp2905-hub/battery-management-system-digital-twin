"""Analog front-end (AFE) measurement model.

Estimators are usually fed near-ideal voltage/current/temperature.  A real BMS
measures through an analog front-end — a battery-monitor IC (an LTC/ADBMS-class
AFE) and a current sensor — that adds quantisation, gain/offset error, thermal
noise, and, on the current channel, a finite **bandwidth** that smears
transients.  :class:`AFE` applies that chain so the estimators can be scored
against realistic measurements, not the plant's clean state.

Feed a clean trace through :meth:`AFE.apply_to_drivecycle` and re-run
:func:`bms.estimator_leaderboard` to see how the ranking shifts under real
measurement — the AFE hits the voltage-feedback filters (which trust the ADC)
harder than the open-loop Coulomb counter, so it changes what "best" means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AFEConfig:
    """Analog front-end error budget."""

    v_bits: int = 14                # voltage ADC resolution
    v_range_V: float = 5.0          # voltage ADC full scale (unipolar)
    v_noise_V: float = 8e-4         # voltage thermal-noise std
    v_gain_error: float = 1e-3      # fractional gain error
    v_offset_V: float = 5e-4        # voltage offset error
    i_bits: int = 12                # current ADC resolution
    i_range_A: float = 300.0        # current ADC full scale (bipolar ±)
    i_noise_A: float = 0.05         # current thermal-noise std
    i_bandwidth_hz: float = 10.0    # current-sensor bandwidth (0 = ideal)
    i_offset_A: float = 0.02        # current-sensor offset (bias)
    t_bits: int = 10                # temperature ADC resolution
    t_range_C: float = 120.0        # temperature ADC full scale
    t_noise_C: float = 0.2          # temperature noise std
    seed: int = 0


class AFE:
    """Apply an :class:`AFEConfig` measurement chain to true signals."""

    def __init__(self, config: AFEConfig | None = None):
        self.cfg = config or AFEConfig()
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.cfg.seed)
        self._i_filt: float | None = None      # current low-pass state

    @staticmethod
    def _quantize(value, bits: int, full_scale: float, bipolar: bool):
        span = full_scale * (2.0 if bipolar else 1.0)
        step = span / (2 ** bits)
        return np.round(np.asarray(value) / step) * step

    def measure_voltage(self, v_true: float) -> float:
        c = self.cfg
        v = v_true * (1.0 + c.v_gain_error) + c.v_offset_V
        v = v + self._rng.normal(0.0, c.v_noise_V)
        return float(self._quantize(v, c.v_bits, c.v_range_V, bipolar=False))

    def measure_current(self, i_true: float, dt: float) -> float:
        c = self.cfg
        # First-order bandwidth limit on the current sensor.
        if c.i_bandwidth_hz > 0:
            tau = 1.0 / (2.0 * math.pi * c.i_bandwidth_hz)
            a = dt / (dt + tau)
            self._i_filt = i_true if self._i_filt is None else self._i_filt + a * (i_true - self._i_filt)
            i = self._i_filt
        else:
            i = i_true
        i = i + c.i_offset_A + self._rng.normal(0.0, c.i_noise_A)
        return float(self._quantize(i, c.i_bits, c.i_range_A, bipolar=True))

    def measure_temperature(self, t_true: float) -> float:
        c = self.cfg
        t = t_true + self._rng.normal(0.0, c.t_noise_C)
        return float(self._quantize(t, c.t_bits, c.t_range_C, bipolar=False))

    def measure(self, v_true: float, i_true: float, t_true: float,
                dt: float) -> tuple[float, float, float]:
        return (self.measure_voltage(v_true), self.measure_current(i_true, dt),
                self.measure_temperature(t_true))

    def apply_to_drivecycle(self, data):
        """Return a copy of a :class:`bms.DriveCycleData` measured through the AFE."""
        from dataclasses import replace
        self.reset()
        dt = data.dt
        v = np.array([self.measure_voltage(float(x)) for x in data.voltage_V])
        i = np.array([self.measure_current(float(x), dt) for x in data.current_A])
        t = np.array([self.measure_temperature(float(x)) for x in data.temperature_C])
        return replace(data, voltage_V=v, current_A=i, temperature_C=t,
                       name=f"{data.name}+afe")
