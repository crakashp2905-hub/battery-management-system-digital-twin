"""Fleet digital twins — from one battery to a population.

One twin per vehicle is useful; a **fleet** of them is where the value compounds.
Aggregating many per-vehicle states reveals what a single twin cannot: the health
distribution across the population, the outliers degrading faster than their
peers, and the vehicles heading for early failure — the signal a warranty or
maintenance team actually acts on.

:class:`FleetTwin` takes a list of :class:`VehicleState` roll-ups and produces a
:class:`FleetInsight`: population SoH statistics (mean, spread, bottom-decile),
an **at-risk list** (vehicles whose health is a low outlier *or* below an absolute
floor *or* degrading unusually fast), the fastest degraders, and a coarse
fast/normal/slow **degradation clustering**.  It pairs naturally with
:mod:`bms.digital_thread` (birth data → per-vehicle projection) and
:mod:`bms.reliability` (population life statistics).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VehicleState:
    """A per-vehicle health roll-up (one row of the fleet)."""

    vehicle_id: str
    soh: float                          # capacity retention (Q / Q0)
    resistance_growth_pct: float = 0.0
    equivalent_full_cycles: float = 0.0
    mean_temperature_C: float = 25.0

    @property
    def fade_per_cycle(self) -> float:
        return (1.0 - self.soh) / max(self.equivalent_full_cycles, 1.0)


@dataclass(frozen=True)
class FleetInsight:
    """Population-level view of a fleet's battery health."""

    n: int
    mean_soh: float
    std_soh: float
    p10_soh: float                      # bottom-decile SoH
    worst_vehicle: str
    at_risk: list                       # vehicle_ids flagged for attention
    fastest_degraders: list             # (vehicle_id, fade_per_cycle), worst first
    clusters: dict                      # vehicle_id -> "fast"/"normal"/"slow"

    def to_dict(self) -> dict:
        return {
            "n": self.n, "mean_soh": self.mean_soh, "std_soh": self.std_soh,
            "p10_soh": self.p10_soh, "worst_vehicle": self.worst_vehicle,
            "at_risk": self.at_risk, "fastest_degraders": self.fastest_degraders,
            "clusters": self.clusters,
        }


@dataclass
class FleetTwin:
    """A population of per-vehicle battery states with fleet analytics."""

    vehicles: list = field(default_factory=list)

    def add(self, v: VehicleState) -> None:
        self.vehicles.append(v)

    def insight(self, *, z_threshold: float = 1.5, soh_floor: float = 0.80
                ) -> FleetInsight:
        """Summarise the fleet and flag the vehicles needing attention.

        A vehicle is *at risk* if its SoH is a low outlier (below
        ``mean − z·std``), below the absolute ``soh_floor``, or its per-cycle
        fade is in the top decile of the fleet.
        """
        if not self.vehicles:
            return FleetInsight(0, 1.0, 0.0, 1.0, "none", [], [], {})
        soh = np.array([v.soh for v in self.vehicles], float)
        rates = np.array([v.fade_per_cycle for v in self.vehicles], float)
        mean, std = float(np.mean(soh)), float(np.std(soh))
        p10 = float(np.percentile(soh, 10))
        rate_p90 = float(np.percentile(rates, 90))
        outlier_cut = mean - z_threshold * std

        big_fleet = len(self.vehicles) >= 10
        at_risk = [v.vehicle_id for v in self.vehicles
                   if (v.soh < outlier_cut) or (v.soh < soh_floor)
                   or (big_fleet and v.fade_per_cycle >= rate_p90)]

        order = sorted(self.vehicles, key=lambda v: v.fade_per_cycle, reverse=True)
        fastest = [(v.vehicle_id, float(round(v.fade_per_cycle, 8))) for v in order[:5]]

        # Coarse fast/normal/slow clustering on the per-cycle fade terciles.
        lo, hi = np.percentile(rates, [33.0, 66.0])
        clusters = {}
        for v in self.vehicles:
            r = v.fade_per_cycle
            clusters[v.vehicle_id] = ("fast" if r > hi else
                                      "slow" if r < lo else "normal")

        worst = min(self.vehicles, key=lambda v: v.soh).vehicle_id
        return FleetInsight(
            n=len(self.vehicles), mean_soh=mean, std_soh=std, p10_soh=p10,
            worst_vehicle=worst, at_risk=at_risk, fastest_degraders=fastest,
            clusters=clusters,
        )

    def at_risk_vehicles(self, **kwargs) -> list:
        return self.insight(**kwargs).at_risk
