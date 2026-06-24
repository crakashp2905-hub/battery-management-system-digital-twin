"""
Three cell-balancing strategies for a series Li-ion pack.

All three implement the same interface
    `step(pack, dt) -> (balancing_currents [n_cells], energy_loss_J)`
so the supervisor (or comparison harness) can swap them transparently.

1. Passive resistive — burns the excess charge of high-SOC cells through a
   bleed resistor. Simple, cheap, very lossy. Energy is dissipated as heat.

2. Switched capacitor — periodically transfers charge between adjacent
   cells through a flying capacitor. Lossless in the ideal limit; real
   efficiency is bounded by RC time-constants and switching losses (~95 %).

3. Inductor-based active — bidirectional buck-boost between adjacent cells
   moves charge from the highest-SOC cell to the lowest. Highest efficiency
   (~98 %) but the most expensive hardware.

`compare_balancers` runs the same idle-pack scenario through all three and
returns SOC convergence, energy delivered, and energy lost as a DataFrame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .pack import BatteryPack


# ----------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------
class Balancer(ABC):
    """Abstract balancing strategy."""

    name: str = "base"
    soc_threshold: float = 5e-3   # 0.5 % SOC band considered balanced

    def __init__(self):
        self.energy_loss_J = 0.0

    def reset(self):
        self.energy_loss_J = 0.0

    @abstractmethod
    def step(self, pack: BatteryPack, dt: float) -> np.ndarray:
        """Return a (n_cells,) array of balancing currents (A, +ve = drain)."""

    def is_balanced(self, pack: BatteryPack) -> bool:
        return pack.soc_imbalance() < self.soc_threshold


# ----------------------------------------------------------------------
# 1. Passive (dissipative) resistive shunt
# ----------------------------------------------------------------------
@dataclass
class PassiveBalancer(Balancer):
    """Bleed-resistor shunt across each cell whose SOC exceeds min(SOC) + band.

    Parameters
    ----------
    bleed_resistance : Ω. Smaller R → faster but higher peak power.
    activation_band  : Cells whose SOC exceeds (min SOC + band) are shunted.
    """

    bleed_resistance: float = 33.0
    activation_band: float = 1e-3

    def __post_init__(self):
        super().__init__()
        self.name = "passive_resistive"

    def step(self, pack: BatteryPack, dt: float) -> np.ndarray:
        soc = pack.soc
        v_cells = pack.cell_voltages()
        target = soc.min() + self.activation_band
        currents = np.where(
            soc > target,
            v_cells / self.bleed_resistance,    # discharge through resistor
            0.0,
        )
        # All shunt power is dissipated.
        self.energy_loss_J += float(np.sum(v_cells * currents) * dt)
        return currents


# ----------------------------------------------------------------------
# 2. Switched-capacitor (active, near-lossless)
# ----------------------------------------------------------------------
@dataclass
class SwitchedCapacitorBalancer(Balancer):
    """Adjacent-cell flying-capacitor transfer.

    Each switching cycle the capacitor C_sw is connected first across the
    higher-V cell of an adjacent pair, then across the lower-V cell. The
    charge transferred per cycle is approximately C_sw · ΔV; with switching
    frequency f_sw, the average transfer current is f_sw · C_sw · ΔV.
    """

    C_sw: float = 1e-3                # 1 mF flying cap
    f_sw: float = 1_000.0             # 1 kHz switching
    efficiency: float = 0.95          # account for switch / ESR losses

    def __post_init__(self):
        super().__init__()
        self.name = "switched_capacitor"

    def step(self, pack: BatteryPack, dt: float) -> np.ndarray:
        v = pack.cell_voltages()
        currents = np.zeros(pack.n_cells)

        # Charge transfer between every adjacent pair.
        for i in range(pack.n_cells - 1):
            dv = v[i] - v[i + 1]
            i_avg = self.f_sw * self.C_sw * dv          # signed current
            currents[i] += i_avg                        # leaves cell i
            currents[i + 1] -= self.efficiency * i_avg  # arrives at i+1 (with loss)

        # Energy lost per dt = (1-η) · Σ |V·I| transferred
        v_drop = np.abs(v[:-1] - v[1:])
        i_pair = self.f_sw * self.C_sw * v_drop
        self.energy_loss_J += float((1 - self.efficiency) * np.sum(v[:-1] * i_pair) * dt)
        return currents


# ----------------------------------------------------------------------
# 3. Inductor-based active (highest performance)
# ----------------------------------------------------------------------
@dataclass
class InductorBalancer(Balancer):
    """Bidirectional buck–boost between cells.

    A simplified model: at each step, identify the highest-SOC and lowest-SOC
    cell; transfer up to `max_current` from the donor to the receiver,
    proportional to their SOC gap. Hardware efficiency η = 0.98.
    """

    max_current_A: float = 1.5
    gain: float = 8.0          # SOC-gap → current proportional gain
    efficiency: float = 0.98

    def __post_init__(self):
        super().__init__()
        self.name = "inductor_active"

    def step(self, pack: BatteryPack, dt: float) -> np.ndarray:
        soc = pack.soc
        currents = np.zeros(pack.n_cells)
        donor = int(np.argmax(soc))
        receiver = int(np.argmin(soc))
        if donor == receiver:
            return currents

        gap = soc[donor] - soc[receiver]
        i_xfer = float(np.clip(self.gain * gap, 0.0, self.max_current_A))
        currents[donor] = i_xfer                        # leaves donor
        currents[receiver] = -self.efficiency * i_xfer  # arrives, less loss

        v = pack.cell_voltages()
        self.energy_loss_J += float((1 - self.efficiency) * v[donor] * i_xfer * dt)
        return currents


# ----------------------------------------------------------------------
# Comparison harness
# ----------------------------------------------------------------------
def compare_balancers(
    pack_factory,
    duration_s: float = 7_200.0,
    dt: float = 1.0,
    pack_current_A: float = 0.0,
    strategies: list[Balancer] | None = None,
) -> pd.DataFrame:
    """Run identical idle-pack scenarios through each strategy.

    Parameters
    ----------
    pack_factory : callable -> BatteryPack
        Returns a *fresh* pack with identical initial scatter for fair
        comparison. Typical use:  ``lambda: BatteryPack(PackConfig(seed=1))``.
    """
    if strategies is None:
        strategies = [PassiveBalancer(), SwitchedCapacitorBalancer(), InductorBalancer()]

    n_steps = int(duration_s / dt)
    rows = []
    history = {}

    for strat in strategies:
        pack = pack_factory()
        strat.reset()
        soc_log = np.empty((n_steps + 1, pack.n_cells))
        imb_log = np.empty(n_steps + 1)
        soc_log[0] = pack.soc
        imb_log[0] = pack.soc_imbalance()
        time_balanced = None

        for k in range(n_steps):
            bal = strat.step(pack, dt)
            pack.step(pack_current_A, dt, balancing_currents=bal)
            soc_log[k + 1] = pack.soc
            imb_log[k + 1] = pack.soc_imbalance()
            if time_balanced is None and strat.is_balanced(pack):
                time_balanced = (k + 1) * dt

        history[strat.name] = {"soc": soc_log, "imbalance": imb_log}
        rows.append({
            "strategy": strat.name,
            "final_imbalance": imb_log[-1],
            "time_to_balance_s": time_balanced if time_balanced is not None else np.nan,
            "energy_loss_J": strat.energy_loss_J,
        })

    df = pd.DataFrame(rows).set_index("strategy")
    df.attrs["history"] = history
    return df
