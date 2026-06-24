"""
Battery Passport — lifetime operational record.

The :class:`BatteryPassport` accumulates energy throughput, equivalent full
cycles, round-trip efficiency, and depth-weighted cycles for a single pack
over its operational life.  It is integrated into :class:`BMSSupervisor`
and updated every control cycle.

Key metrics
-----------
* **Equivalent full cycles (EFC)** — total discharge Ah / nominal capacity Ah.
  Most widely used cycle-life metric.
* **Depth-weighted cycles (DWC)** — simplified rain-flow half-cycle counter:
  Σ|ΔSOC| / 2.  Weighted by actual depth-of-discharge, so a shallow pulse
  accumulates proportionally less than a full discharge.
* **Round-trip efficiency (RTE)** — energy out / energy in (0–1).
  Integrates over the entire operational history.

Usage
-----
::

    sup = BMSSupervisor(pack, thermal, detector)
    for k in range(n_steps):
        out = sup.step(I, 1.0, k=k)

    print(sup.passport.summary())
"""

from __future__ import annotations


class BatteryPassport:
    """Lifetime operational log for a battery pack.

    Parameters
    ----------
    nominal_capacity_Ah : float
        Pack-level nominal capacity [Ah] (sum of all parallel branches across
        all series groups).  Used as the denominator for EFC.
    nominal_voltage_V : float
        Pack nominal voltage [V].  Stored for reference only.
    chemistry : str
        Chemistry label — stored verbatim in :meth:`summary`.
    """

    def __init__(
        self,
        nominal_capacity_Ah: float,
        nominal_voltage_V: float,
        chemistry: str = "nmc",
    ) -> None:
        self.nominal_capacity_Ah = float(nominal_capacity_Ah)
        self.nominal_voltage_V = float(nominal_voltage_V)
        self.chemistry = chemistry

        self.total_charge_Ah: float = 0.0
        self.total_discharge_Ah: float = 0.0
        self.total_energy_in_Wh: float = 0.0
        self.total_energy_out_Wh: float = 0.0
        self.total_time_s: float = 0.0
        self._dod_accum: float = 0.0
        self._prev_soc: float | None = None

    # ------------------------------------------------------------------
    def update(
        self,
        current_A: float,
        v_pack_V: float,
        dt_s: float,
        soc_mean: float | None = None,
    ) -> None:
        """Record one supervisor time-step.

        Parameters
        ----------
        current_A : float
            Pack current (positive = discharge, negative = charge) [A].
        v_pack_V : float
            Pack terminal voltage [V].
        dt_s : float
            Step duration [s].
        soc_mean : float, optional
            Mean pack SOC — used for depth-weighted cycle counting.
        """
        dq = abs(current_A) * dt_s / 3600.0
        dE = abs(current_A) * abs(v_pack_V) * dt_s / 3600.0

        if current_A > 1e-3:        # discharge
            self.total_discharge_Ah += dq
            self.total_energy_out_Wh += dE
        elif current_A < -1e-3:     # charge
            self.total_charge_Ah += dq
            self.total_energy_in_Wh += dE

        self.total_time_s += dt_s

        if soc_mean is not None:
            if self._prev_soc is not None:
                self._dod_accum += abs(soc_mean - self._prev_soc)
            self._prev_soc = soc_mean

    # ------------------------------------------------------------------
    @property
    def equivalent_full_cycles(self) -> float:
        """Discharge Ah integrated / nominal capacity [full-cycle equivalents]."""
        return self.total_discharge_Ah / max(self.nominal_capacity_Ah, 1e-9)

    @property
    def depth_weighted_cycles(self) -> float:
        """Σ|ΔSOC| / 2 — half-cycle rain-flow approximation."""
        return self._dod_accum / 2.0

    @property
    def round_trip_efficiency(self) -> float:
        """Energy out / energy in [0–1].  Returns 1.0 if no charge occurred."""
        if self.total_energy_in_Wh < 1e-6:
            return 1.0
        return min(1.0, self.total_energy_out_Wh / self.total_energy_in_Wh)

    @property
    def total_time_h(self) -> float:
        return self.total_time_s / 3600.0

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Snapshot dict of all passport metrics."""
        return {
            "chemistry": self.chemistry,
            "nominal_capacity_Ah": round(self.nominal_capacity_Ah, 4),
            "total_discharge_Ah": round(self.total_discharge_Ah, 4),
            "total_charge_Ah": round(self.total_charge_Ah, 4),
            "total_energy_out_Wh": round(self.total_energy_out_Wh, 4),
            "total_energy_in_Wh": round(self.total_energy_in_Wh, 4),
            "equivalent_full_cycles": round(self.equivalent_full_cycles, 4),
            "depth_weighted_cycles": round(self.depth_weighted_cycles, 4),
            "round_trip_efficiency": round(self.round_trip_efficiency, 4),
            "total_time_h": round(self.total_time_h, 4),
        }

    def __repr__(self) -> str:
        return (
            f"BatteryPassport(chemistry={self.chemistry!r}, "
            f"EFC={self.equivalent_full_cycles:.3f}, "
            f"RTE={self.round_trip_efficiency * 100:.1f}%, "
            f"time={self.total_time_h:.2f} h)"
        )
