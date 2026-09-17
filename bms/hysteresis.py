"""Dynamic (Plett) one-state OCV hysteresis.

The package's default hysteresis is *static* — ``OCV − M·sign(I)`` — which flips
instantly with the current sign.  Real hysteresis is **dynamic**: the cell moves
between the charge and discharge OCV branches gradually, over a bit of charge
throughput, and a brief current reversal does not fully flip it.  This is Plett's
one-state model,

    h[k+1] = e^(−|γ·ΔAh|)·h[k] + (1 − e^(−|γ·ΔAh|))·(−sign(I))

with the hysteresis voltage ``M·h`` (``h ∈ [−1, 1]``) added to the OCV.  It is
opt-in on :class:`bms.SecondOrderECM` (``hysteresis=PlettHysteresis(...)``);
enabling it replaces the static term so nothing is double-counted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlettHysteresis:
    """One-state dynamic hysteresis.

    Parameters
    ----------
    max_hysteresis_V : float
        ``M`` — the half-width of the charge/discharge OCV gap [V].
    rate_per_Ah : float
        ``γ`` — how fast ``h`` saturates with charge throughput; larger = flips
        to the branch sooner.  In units of 1/Ah of throughput.
    """

    max_hysteresis_V: float = 0.02
    rate_per_Ah: float = 8.0
    h: float = 0.0

    def reset(self, h0: float = 0.0) -> None:
        self.h = float(np.clip(h0, -1.0, 1.0))

    @property
    def voltage(self) -> float:
        """Current hysteresis contribution ``M·h`` [V]."""
        return self.max_hysteresis_V * self.h

    def update(self, current: float, dt: float, capacity_Ah: float) -> float:
        """Advance ``h`` by one step and return the hysteresis voltage [V].

        ``current > 0`` = discharge (drives ``h`` toward −1, a lower OCV branch);
        charge drives it toward +1.
        """
        throughput_frac = abs(current) * dt / (capacity_Ah * 3600.0)
        decay = float(np.exp(-self.rate_per_Ah * throughput_frac))
        target = -float(np.sign(current))          # 0 when current is 0 (h holds)
        self.h = float(np.clip(decay * self.h + (1.0 - decay) * target, -1.0, 1.0))
        return self.voltage
