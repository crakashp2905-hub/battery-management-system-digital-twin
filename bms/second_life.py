"""Second-life assessment and simple battery economics.

When an EV pack falls below its automotive end-of-life (~80 % SoH) it is not
waste — it still holds most of its energy and can serve a less-demanding
**stationary** second life before recycling.  This module makes the retire /
repurpose / recycle call, estimates the **residual value** and the **second-life
years** remaining, and rolls a first+second-life **levelised cost** (a simple
TCO per kWh delivered over the whole life).
"""

from __future__ import annotations

from dataclasses import dataclass

from .aging import AgingModel, AgingParams, AgingState


@dataclass
class SecondLifeAssessment:
    verdict: str                     # "first_life" | "second_life" | "recycle"
    soh: float
    residual_value: float            # $ recoverable now
    second_life_years: float         # estimated years of stationary service left
    recommendation: str


def assess_second_life(soh: float, capacity_kWh: float = 60.0,
                       new_price_per_kWh: float = 130.0,
                       first_eol: float = 0.80, second_eol: float = 0.60,
                       recycle_value_per_kWh: float = 12.0,
                       stationary_cycles_per_year: float = 150.0,
                       temperature_C: float = 25.0,
                       params: AgingParams | None = None) -> SecondLifeAssessment:
    """Assess a pack at ``soh`` for second life.

    Above ``first_eol`` it belongs in first life; between the two thresholds it is
    a second-life candidate; at/below ``second_eol`` it goes to recycling.  The
    residual value scales with the usable SoH headroom above the recycle floor;
    the second-life horizon is projected from the aging model at a gentle
    stationary duty.
    """
    soh = float(soh)
    if soh > first_eol:
        verdict, rec = "first_life", "Keep in first-life (automotive) service."
    elif soh > second_eol:
        verdict, rec = "second_life", "Repurpose for stationary storage (gentle duty)."
    else:
        verdict, rec = "recycle", "Below second-life floor — send to recycling."

    if verdict == "recycle":
        residual = capacity_kWh * soh * recycle_value_per_kWh
        return SecondLifeAssessment(verdict, soh, round(residual, 2), 0.0, rec)

    # Usable-value fraction between the recycle floor and beginning-of-life.
    headroom = max(0.0, (soh - second_eol) / (1.0 - second_eol))
    residual = capacity_kWh * new_price_per_kWh * headroom * 0.5   # discounted vs new

    # Project years of stationary service until the second-life EoL.
    model = AgingModel(params or AgingParams())
    st = AgingState(soh_capacity=soh)
    years = 0.0
    while st.soh_capacity > second_eol and years < 30:
        st = model.cycle(st, stationary_cycles_per_year, c_rate=0.5,
                         temperature_C=temperature_C, dod=0.6, soc_avg=0.5)
        st = model.calendar(st, 365.0, temperature_C, soc_avg=0.5)
        years += 1
    return SecondLifeAssessment(verdict, soh, round(residual, 2), float(years), rec)


def levelized_cost_per_kWh(capacity_kWh: float, new_price_per_kWh: float,
                           first_life_cycles: float, second_life_cycles: float = 0.0,
                           depth_of_discharge: float = 0.8) -> float:
    """Rough levelised cost of energy [$/kWh] over the battery's whole life.

    ``(purchase − nothing) / total energy throughput``; a second life adds cycles
    at no extra purchase cost, so it strictly lowers the levelised cost.
    """
    purchase = capacity_kWh * new_price_per_kWh
    energy = capacity_kWh * depth_of_discharge * (first_life_cycles + second_life_cycles)
    return float(purchase / max(energy, 1e-9))
