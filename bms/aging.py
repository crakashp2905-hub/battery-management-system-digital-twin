"""
Dynamic state-of-health (SoH): capacity fade and resistance growth.

This closes the loop the twin was missing — cells that actually **age**.  An
:class:`AgingModel` prices each (dis)charge and each calendar interval into a
capacity loss and a resistance rise, accumulated in an :class:`AgingState`
(``soh_capacity = Q/Q0``, ``soh_resistance = R0/R0_0``).  :meth:`apply_to_pack`
writes that state back onto a :class:`bms.pack.BatteryPack`, so subsequent
simulation sees the degraded cell.

Degradation drivers (all multiplicative on a baseline per-throughput rate)
------------------------------------------------------------------------
* **C-rate** — higher current ages faster (``1 + s·(C-1)`` above 1C).
* **Temperature** — Arrhenius-like acceleration above 25 °C.
* **Depth of discharge** — deeper cycles stress the electrodes more.
* **Lithium plating** — a large multiplier when a charge exceeds the plating
  C-rate limit (supplied by :mod:`bms.charging`); the dominant fast/cold-charge
  damage term.
* **Calendar** — √time storage fade, Arrhenius in T and rising with mean SoC.

.. note::
   Coefficients are literature-plausible engineering estimates reproducing the
   well-known trends (fast/hot/cold/high-SoC ages faster), not a fit to a
   specific dataset.  All are :class:`AgingParams` fields for recalibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AgingParams:
    """Semi-empirical capacity-fade and resistance-growth coefficients."""

    # Cycle (throughput) capacity fade, as a fraction of Q0 per equivalent full
    # cycle at the 1C / 25 °C / full-DoD baseline.
    # Baseline chosen so a full 1C / 25 °C / 100 %-DoD cycle loses ~1e-4 of Q0,
    # i.e. ~2000 equivalent full cycles to the 80 % end-of-life.
    k_cycle: float = 1.0e-4
    crate_sensitivity: float = 0.30      # + fraction per C above 1C
    temp_ref_C: float = 25.0
    hot_sensitivity: float = 0.03        # exp growth per °C above ref
    dod_floor: float = 0.5               # f_dod = floor + (1-floor)·DoD
    plating_accel: float = 1.0           # fade multiplier per unit plating risk
    # Resistance growth (fraction of R0_0 per equivalent full cycle, same drivers).
    k_resistance: float = 1.0e-4
    # Calendar fade: fraction of Q0 per √day at 25 °C, SoC 0.5.
    k_calendar: float = 4.0e-4
    cal_arrhenius_K: float = 5000.0
    cal_soc_ref: float = 0.5
    cal_soc_sensitivity: float = 1.5


@dataclass
class AgingState:
    """Accumulated SoH.  ``soh_capacity`` ≤ 1 falls; ``soh_resistance`` ≥ 1 rises."""

    soh_capacity: float = 1.0
    soh_resistance: float = 1.0
    equivalent_full_cycles: float = 0.0
    calendar_days: float = 0.0

    @property
    def soh_pct(self) -> float:
        return 100.0 * self.soh_capacity

    def summary(self) -> dict:
        return {
            "soh_capacity_pct": round(self.soh_pct, 3),
            "resistance_growth_pct": round(100.0 * (self.soh_resistance - 1.0), 3),
            "equivalent_full_cycles": round(self.equivalent_full_cycles, 3),
            "calendar_days": round(self.calendar_days, 2),
        }


@dataclass
class AgingModel:
    """Prices throughput and storage into capacity fade / resistance growth."""

    params: AgingParams = field(default_factory=AgingParams)

    # ---- stress factors ------------------------------------------------
    def _crate_factor(self, c_rate: float) -> float:
        return 1.0 + self.params.crate_sensitivity * max(0.0, c_rate - 1.0)

    def _temp_factor(self, temperature_C: float) -> float:
        p = self.params
        return float(np.exp(p.hot_sensitivity * max(0.0, temperature_C - p.temp_ref_C)))

    def _dod_factor(self, dod: float) -> float:
        p = self.params
        return float(p.dod_floor + (1.0 - p.dod_floor) * np.clip(dod, 0.0, 1.0))

    def _plating_factor(self, plating: float) -> float:
        return 1.0 + self.params.plating_accel * max(0.0, float(plating))

    # ---- per-event fade ------------------------------------------------
    def charge_fade(self, c_rate: float, temperature_C: float, dod: float,
                    soc_avg: float, throughput_efc: float,
                    plating: float = 0.0) -> tuple[float, float]:
        """Return (capacity_fade_frac, resistance_growth_frac) for one event."""
        p = self.params
        f = (self._crate_factor(c_rate) * self._temp_factor(temperature_C)
             * self._dod_factor(dod) * self._plating_factor(plating))
        d_cap = p.k_cycle * f * max(0.0, throughput_efc)
        d_res = p.k_resistance * f * max(0.0, throughput_efc)
        return float(d_cap), float(d_res)

    # ---- state advancement --------------------------------------------
    def cycle(self, state: AgingState, throughput_efc: float, c_rate: float,
              temperature_C: float, dod: float, soc_avg: float = 0.5,
              plating: float = 0.0) -> AgingState:
        """Advance SoH by one (dis)charge of the given throughput."""
        d_cap, d_res = self.charge_fade(c_rate, temperature_C, dod, soc_avg,
                                        throughput_efc, plating)
        return AgingState(
            soh_capacity=max(0.0, state.soh_capacity - d_cap),
            soh_resistance=state.soh_resistance + d_res,
            equivalent_full_cycles=state.equivalent_full_cycles + max(0.0, throughput_efc),
            calendar_days=state.calendar_days,
        )

    def calendar(self, state: AgingState, days: float, temperature_C: float,
                 soc_avg: float) -> AgingState:
        """Advance calendar (storage) aging by ``days`` — a √time law."""
        p = self.params
        t_K = temperature_C + 273.15
        arr = float(np.exp(p.cal_arrhenius_K * (1.0 / 298.15 - 1.0 / t_K)))
        soc_f = max(0.2, 1.0 + p.cal_soc_sensitivity * (soc_avg - p.cal_soc_ref))
        t0 = state.calendar_days
        d_cap = p.k_calendar * arr * soc_f * (np.sqrt(t0 + days) - np.sqrt(t0))
        return AgingState(
            soh_capacity=max(0.0, state.soh_capacity - float(d_cap)),
            soh_resistance=state.soh_resistance,
            equivalent_full_cycles=state.equivalent_full_cycles,
            calendar_days=t0 + max(0.0, days),
        )

    # ---- remaining useful life ----------------------------------------
    def rul_cycles(self, state: AgingState, capacity_fade_per_cycle: float,
                   eol: float = 0.8) -> float:
        """Full cycles remaining to ``eol`` at a given per-cycle fade rate."""
        if capacity_fade_per_cycle <= 0.0:
            return float("inf")
        return float(max(0.0, (state.soh_capacity - eol) / capacity_fade_per_cycle))

    # ---- feedback into the pack ---------------------------------------
    def apply_to_pack(self, pack, state: AgingState) -> None:
        """Scale every cell's capacity and R0 to reflect ``state``.

        Idempotent: the beginning-of-life values are cached on each cell the
        first time this is called, so repeated calls always scale from BOL
        rather than compounding.
        """
        for group in pack.groups:
            for cell in group.cells:
                if not hasattr(cell, "_q_nom_bol"):
                    cell._q_nom_bol = cell.params.Q_nom_Ah
                    cell._r0_bol = cell.params.R0
                cell.params.Q_nom_Ah = cell._q_nom_bol * state.soh_capacity
                cell.params.R0 = cell._r0_bol * state.soh_resistance
