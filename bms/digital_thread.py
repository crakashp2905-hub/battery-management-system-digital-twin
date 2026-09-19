"""Manufacturing → field digital thread — birth certificate to end of life.

A cell's fate is partly written at the factory.  Formation (the first controlled
cycles that build the SEI), capacity grading and end-of-line resistance carry
signatures that correlate with how fast the cell will later fade in the field.
This module threads that manufacturing data forward into a **predicted field
degradation trajectory**, so a pack builder can flag the cells likely to fail
early *before* they are ever deployed.

The couplings are physically motivated and deliberately simple:

* **Formation coulombic efficiency** — a low first-cycle efficiency means more
  lithium was consumed building a poorer SEI, which keeps consuming lithium in
  the field → a **capacity-fade** multiplier.
* **Capacity grade** (initial / nominal Ah) — a low-grade cell starts closer to
  end-of-life, so it reaches it in fewer cycles.
* **End-of-line resistance** — a cell that leaves the factory with high `R0`
  tends to grow resistance (and heat) faster → a **resistance-growth** multiplier.
* **Self-discharge grade** — excess leakage hints at micro-defects; it adds a
  small capacity-fade penalty.

:func:`project_field_trajectory` turns a record + a usage profile into a SoH
curve and a cycles/days-to-EOL; :class:`DigitalThread` ranks a whole batch by
field risk and reports its life distribution (B10), linking the factory to
:mod:`bms.reliability`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .aging import AgingModel, AgingParams


@dataclass
class ManufacturingRecord:
    """A cell's factory birth certificate."""

    cell_id: str
    formation_efficiency: float = 0.92        # first-cycle coulombic efficiency
    initial_capacity_Ah: float = 2.3
    nominal_capacity_Ah: float = 2.3
    initial_r0_ohm: float = 0.025
    nominal_r0_ohm: float = 0.025
    self_discharge_pct_per_month: float = 2.0

    @property
    def capacity_grade(self) -> float:
        return float(self.initial_capacity_Ah / max(self.nominal_capacity_Ah, 1e-9))


@dataclass
class FieldUsage:
    """How the cell will be used in the field (per-cycle stressors)."""

    c_rate: float = 1.0
    temperature_C: float = 30.0
    dod: float = 0.8                          # depth of discharge per cycle
    soc_avg: float = 0.5
    plating: float = 0.0
    cycles_per_day: float = 1.0


@dataclass(frozen=True)
class FormationCoupling:
    """Multipliers linking manufacturing metrics to field aging rates."""

    capacity_fade_mult: float
    resistance_growth_mult: float


def link_formation_to_aging(record: ManufacturingRecord, *,
                            ce_ref: float = 0.92, r0_ref: float | None = None,
                            sd_ref: float = 2.0,
                            ce_sensitivity: float = 4.0,
                            r0_sensitivity: float = 1.0,
                            sd_sensitivity: float = 0.05) -> FormationCoupling:
    """Map a birth certificate to field-aging-rate multipliers (≥ 1 = worse)."""
    r0_ref = record.nominal_r0_ohm if r0_ref is None else r0_ref
    ce_deficit = max(0.0, ce_ref - record.formation_efficiency)
    r0_excess = max(0.0, (record.initial_r0_ohm - r0_ref) / max(r0_ref, 1e-12))
    sd_excess = max(0.0, record.self_discharge_pct_per_month - sd_ref)
    cap_mult = 1.0 + ce_sensitivity * ce_deficit + sd_sensitivity * sd_excess
    res_mult = 1.0 + r0_sensitivity * r0_excess
    return FormationCoupling(capacity_fade_mult=float(cap_mult),
                             resistance_growth_mult=float(res_mult))


@dataclass(frozen=True)
class FieldProjection:
    """Predicted field trajectory for one cell."""

    cell_id: str
    cycles_to_eol: float
    days_to_eol: float
    capacity_fade_mult: float
    resistance_growth_mult: float
    start_soh: float
    fade_per_cycle: float

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id, "cycles_to_eol": self.cycles_to_eol,
            "days_to_eol": self.days_to_eol,
            "capacity_fade_mult": self.capacity_fade_mult,
            "resistance_growth_mult": self.resistance_growth_mult,
            "start_soh": self.start_soh, "fade_per_cycle": self.fade_per_cycle,
        }


def project_field_trajectory(record: ManufacturingRecord, usage: FieldUsage, *,
                             aging_params: AgingParams | None = None,
                             eol: float = 0.8) -> FieldProjection:
    """Project a cell's SoH forward to end-of-life under a usage profile."""
    coupling = link_formation_to_aging(record)
    base = aging_params or AgingParams()
    params = replace(base,
                     k_cycle=base.k_cycle * coupling.capacity_fade_mult,
                     k_resistance=base.k_resistance * coupling.resistance_growth_mult)
    model = AgingModel(params)
    d_cap, _ = model.charge_fade(usage.c_rate, usage.temperature_C, usage.dod,
                                 usage.soc_avg, throughput_efc=usage.dod,
                                 plating=usage.plating)
    start_soh = record.capacity_grade
    if d_cap <= 0.0:
        cycles = float("inf")
    else:
        cycles = max(0.0, (start_soh - eol) / d_cap)
    days = cycles / max(usage.cycles_per_day, 1e-9)
    return FieldProjection(
        cell_id=record.cell_id, cycles_to_eol=float(cycles), days_to_eol=float(days),
        capacity_fade_mult=coupling.capacity_fade_mult,
        resistance_growth_mult=coupling.resistance_growth_mult,
        start_soh=float(start_soh), fade_per_cycle=float(d_cap),
    )


@dataclass
class DigitalThread:
    """A batch of manufacturing records, threaded forward to field risk."""

    records: list[ManufacturingRecord] = field(default_factory=list)

    def project(self, usage: FieldUsage, *, eol: float = 0.8) -> list[FieldProjection]:
        return [project_field_trajectory(r, usage, eol=eol) for r in self.records]

    def rank_field_risk(self, usage: FieldUsage, *, eol: float = 0.8
                        ) -> list[FieldProjection]:
        """Projections sorted worst-first (fewest cycles to EOL) — the at-risk list."""
        return sorted(self.project(usage, eol=eol), key=lambda p: p.cycles_to_eol)

    def life_distribution(self, usage: FieldUsage, *, eol: float = 0.8) -> dict:
        """Batch cycles-to-EOL distribution, incl. B10 (10th-percentile life)."""
        lives = np.array([p.cycles_to_eol for p in self.project(usage, eol=eol)], float)
        finite = lives[np.isfinite(lives)]
        if finite.size == 0:
            return {"n": 0}
        return {
            "n": int(finite.size),
            "mean_cycles": float(np.mean(finite)),
            "std_cycles": float(np.std(finite)),
            "min_cycles": float(np.min(finite)),
            "b10_cycles": float(np.percentile(finite, 10)),
            "worst_cell": min(self.project(usage, eol=eol),
                              key=lambda p: p.cycles_to_eol).cell_id,
        }

    def formation_life_correlation(self, usage: FieldUsage, *, eol: float = 0.8
                                   ) -> float:
        """Pearson correlation between formation efficiency and projected life.

        A strong positive value is the digital thread's core claim: better-formed
        cells last longer, so the birth certificate is predictive.
        """
        proj = self.project(usage, eol=eol)
        ce = np.array([r.formation_efficiency for r in self.records], float)
        life = np.array([p.cycles_to_eol for p in proj], float)
        mask = np.isfinite(life)
        if mask.sum() < 2 or np.std(ce[mask]) == 0 or np.std(life[mask]) == 0:
            return float("nan")
        return float(np.corrcoef(ce[mask], life[mask])[0, 1])
