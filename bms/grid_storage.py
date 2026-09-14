"""Stationary / grid energy-storage dispatch and degradation-aware scheduling.

The EV range predictor covers the mobility use case; this covers the *stationary*
one — a battery serving the grid.  It works at the **energy/power** level
(kWh / kW) with a round-trip efficiency, generates the common duty cycles
(**peak shaving**, **energy arbitrage**), simulates the SoC trajectory and
throughput, and prices the cycling into fade via :class:`bms.aging.AgingModel`
so dispatch can be made **degradation-aware** (don't cycle when the aging cost
outweighs the value).

Also provides :func:`optimal_storage_soc` — the storage SoC that minimises
calendar fade over a horizon, the counterpart of the cycling models for a battery
that is mostly *sitting* (a warehouse or a seasonal reserve).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aging import AgingModel, AgingParams


@dataclass
class StationaryStorage:
    """An energy-level battery: capacity, power limits, round-trip efficiency.

    Sign convention (grid side): ``power_kW > 0`` = **discharge** (to grid),
    ``< 0`` = **charge** (from grid).
    """

    capacity_kWh: float = 100.0
    max_power_kW: float = 50.0
    round_trip_efficiency: float = 0.90
    soc: float = 0.5
    soc_min: float = 0.05
    soc_max: float = 0.95
    throughput_kWh: float = field(default=0.0)

    def reset(self, soc0: float = 0.5) -> None:
        self.soc = float(np.clip(soc0, self.soc_min, self.soc_max))
        self.throughput_kWh = 0.0

    def dispatch(self, power_kW: float, dt_h: float) -> float:
        """Serve a power request for ``dt_h`` hours; returns the *actual* power
        delivered after power/energy limits.  One-way efficiency √RTE is applied
        on each of charge and discharge."""
        p = float(np.clip(power_kW, -self.max_power_kW, self.max_power_kW))
        eff = np.sqrt(self.round_trip_efficiency)
        if p >= 0:                                   # discharge: draw more than delivered
            energy_out = p * dt_h
            e_from_batt = energy_out / eff
            e_from_batt = min(e_from_batt, (self.soc - self.soc_min) * self.capacity_kWh)
            self.soc -= e_from_batt / self.capacity_kWh
            actual = e_from_batt * eff / max(dt_h, 1e-9)
        else:                                        # charge: store less than drawn
            energy_in = -p * dt_h
            e_to_batt = energy_in * eff
            e_to_batt = min(e_to_batt, (self.soc_max - self.soc) * self.capacity_kWh)
            self.soc += e_to_batt / self.capacity_kWh
            actual = -(e_to_batt / eff) / max(dt_h, 1e-9)
        self.throughput_kWh += abs(actual) * dt_h
        return float(actual)

    @property
    def equivalent_full_cycles(self) -> float:
        return self.throughput_kWh / (2.0 * max(self.capacity_kWh, 1e-9))


def peak_shaving_dispatch(load_kW: np.ndarray, threshold_kW: float,
                          recharge_kW: float | None = None) -> np.ndarray:
    """Battery power schedule that caps grid load at ``threshold_kW``.

    Discharges to shave load above the threshold; charges (up to ``recharge_kW``)
    from the headroom below it.  Returns the requested battery power series
    (``>0`` discharge, ``<0`` charge).
    """
    load = np.asarray(load_kW, float)
    recharge_kW = recharge_kW if recharge_kW is not None else threshold_kW * 0.3
    power = np.where(load > threshold_kW, load - threshold_kW,
                     -np.minimum(recharge_kW, threshold_kW - load))
    return power


def arbitrage_schedule(price: np.ndarray, buy_below: float, sell_above: float,
                       power_kW: float) -> np.ndarray:
    """Charge when price < ``buy_below``, discharge when price > ``sell_above``."""
    price = np.asarray(price, float)
    power = np.zeros_like(price)
    power[price > sell_above] = power_kW          # discharge (sell)
    power[price < buy_below] = -power_kW          # charge (buy)
    return power


def simulate_dispatch(storage: StationaryStorage, power_request_kW: np.ndarray,
                      dt_h: float = 0.25, soc0: float = 0.5,
                      aging: AgingModel | None = None,
                      degradation_aware: bool = False,
                      value_per_kWh: float = 0.15,
                      fade_cost_per_pct: float = 50.0) -> dict:
    """Run a dispatch profile; return SoC/power trajectories and a summary.

    With ``degradation_aware`` and an ``aging`` model, each discharge is skipped
    when its marginal fade cost (``fade_cost_per_pct``) exceeds the energy value
    (``value_per_kWh``) — degradation-priced dispatch.
    """
    storage.reset(soc0)
    req = np.asarray(power_request_kW, float)
    soc_traj, actual = np.empty(len(req)), np.empty(len(req))
    served = unmet = 0.0
    for k, p in enumerate(req):
        if degradation_aware and aging is not None and p > 0:
            # Marginal throughput of this discharge (as a fraction of a full cycle).
            efc = (p * dt_h) / (2.0 * storage.capacity_kWh)
            fade_pct = getattr(aging.params, "k_cycle", 0.02) * efc * 100.0
            if fade_pct * fade_cost_per_pct > (p * dt_h) * value_per_kWh:
                p = 0.0                              # not worth the wear
        a = storage.dispatch(float(p), dt_h)
        actual[k] = a
        soc_traj[k] = storage.soc
        if p > 0:
            served += a * dt_h
            unmet += max(0.0, p - a) * dt_h
    return {
        "soc": soc_traj, "power_kW": actual,
        "energy_served_kWh": float(served),
        "energy_unmet_kWh": float(unmet),
        "throughput_kWh": storage.throughput_kWh,
        "equivalent_full_cycles": storage.equivalent_full_cycles,
    }


def optimal_storage_soc(days: float, temperature_C: float = 25.0,
                        aging: AgingModel | None = None,
                        socs: np.ndarray | None = None) -> dict:
    """SoC that minimises calendar fade over ``days`` at ``temperature_C``.

    Sweeps candidate storage SoCs through the calendar-aging model and returns
    the best one plus the fade at each — the answer to "what SoC should a battery
    sit at in storage?".
    """
    from .aging import AgingState
    aging = aging or AgingModel(AgingParams())
    socs = socs if socs is not None else np.linspace(0.1, 0.9, 17)
    fades = []
    for s in socs:
        st = aging.calendar(AgingState(), days=days, temperature_C=temperature_C,
                            soc_avg=float(s))
        fades.append(1.0 - st.soh_capacity)
    fades = np.array(fades)
    best = int(np.argmin(fades))
    return {"best_soc": float(socs[best]),
            "fade_at_best_pct": float(fades[best] * 100.0),
            "socs": np.asarray(socs, float), "fade_pct": fades * 100.0}
