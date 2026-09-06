"""
Charging-method physics: how AC vs DC and fast vs slow charging affect a pack.

This module puts **numbers** on the charging question — given a charging method
(AC level-1/2 or DC fast/ultra), it derives the effective C-rate, wall-to-battery
efficiency, cell heating, charge time, and lithium-plating risk, and (via
:mod:`bms.aging`) the resulting capacity fade and resistance growth per charge.

Key relationships
-----------------
* **C-rate from power.**  Charge C-rate ``C = P_batt_kW / E_pack_kWh`` because
  ``P = V·I`` and ``E = V·Q`` give ``P/E = I/Q = C``.  This is what makes DC
  fast charging "fast": a 50 kW charger into a 50 kWh pack is ~1C; 250 kW is ~5C.
* **AC vs DC is really a C-rate and a loss-location story.**  AC charging runs
  through the on-board charger (OBC), which caps power (≈1.4-11 kW) and loses
  ~10 % as heat *in the charger* (off the cell).  DC fast charging bypasses the
  OBC and feeds the pack directly at high power, so conversion loss is small but
  the cell sees a much higher C-rate → more I²R heat *in the cell* and higher
  aging.  Net wall-to-battery efficiency is often similar; the aging is not.
* **Cell ohmic loss** grows with C-rate: fractional loss ≈ ``C·Q·R0 / V_nom``.
* **Peak temperature rise** scales with I² (∝ C²); we use ``ΔT ≈ k_thermal·C²``
  with a configurable ``k_thermal`` standing in for cooling quality.
* **Lithium plating** — the dominant fast-charge damage mechanism — is triggered
  when the charge C-rate exceeds a temperature- and SoC-dependent limit that
  *falls* at low temperature and high SoC (why DC chargers taper above 80 %).

.. note::
   The coefficients here are literature-plausible **engineering estimates**
   chosen to reproduce well-known qualitative trends (fast/cold/high-SoC charging
   ages cells faster); they are not fitted to a specific proprietary dataset.
   All are exposed as parameters so they can be recalibrated to real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .aging import AgingModel, AgingParams


class ChargeMethod(str, Enum):
    """Supported charging methods with representative nominal power."""

    AC_LEVEL1 = "ac_level1"   # 120 V ~1.4 kW domestic outlet
    AC_LEVEL2 = "ac_level2"   # 240 V ~7-11 kW wallbox / public AC
    DC_FAST = "dc_fast"       # ~50 kW DC fast charger
    DC_ULTRA = "dc_ultra"     # ~150-350 kW ultra-rapid DC


# Nominal charger power [kW] and whether the path is DC (bypasses the OBC).
METHOD_POWER_KW: dict[str, float] = {
    "ac_level1": 1.4,
    "ac_level2": 11.0,
    "dc_fast": 50.0,
    "dc_ultra": 250.0,
}
METHOD_IS_DC: dict[str, bool] = {
    "ac_level1": False, "ac_level2": False, "dc_fast": True, "dc_ultra": True,
}


@dataclass
class ChargeProtocol:
    """A charging session specification.

    Parameters
    ----------
    method : ChargeMethod
        Charging method; sets default power and AC/DC path.
    power_kW : float, optional
        Override the method's nominal power [kW].
    ambient_C : float
        Ambient / initial cell temperature [°C].
    soc_start, soc_end : float
        SoC window charged over [0, 1].
    cc_cutoff_soc : float
        SoC at which constant-current gives way to constant-voltage taper.
    """

    method: ChargeMethod = ChargeMethod.AC_LEVEL2
    power_kW: float | None = None
    ambient_C: float = 25.0
    soc_start: float = 0.2
    soc_end: float = 0.9
    cc_cutoff_soc: float = 0.8

    @property
    def is_dc(self) -> bool:
        return METHOD_IS_DC[self.method.value]

    @property
    def wall_power_kW(self) -> float:
        return float(self.power_kW if self.power_kW is not None
                     else METHOD_POWER_KW[self.method.value])


@dataclass
class ChargeResult:
    """Quantified outcome of one charge session (all magnitudes positive)."""

    method: str
    c_rate: float                 # effective constant-current C-rate
    duration_min: float
    energy_delivered_kWh: float   # into the pack terminals
    energy_stored_kWh: float      # actually retained (after cell loss)
    wall_energy_kWh: float        # drawn from the wall (after conversion loss)
    efficiency: float             # energy_stored / wall_energy
    peak_cell_temp_C: float
    plating_risk: float           # 0 (none) … 1+ (severe)
    capacity_fade_pct: float      # this session, % of present capacity
    resistance_growth_pct: float  # this session
    throughput_efc: float         # equivalent full cycles this session


@dataclass
class ChargingModel:
    """Physics of a charge session and the degradation it causes.

    Parameters
    ----------
    obc_efficiency : float
        On-board-charger (AC path) conversion efficiency.
    dc_conversion_efficiency : float
        Off-board rectifier (DC path) conversion efficiency.
    k_thermal_C_per_C2 : float
        Peak temperature rise per (C-rate)² [°C]; a proxy for cooling quality.
    aging : AgingModel
        Degradation model used to price each charge into fade / resistance.
    """

    obc_efficiency: float = 0.90
    dc_conversion_efficiency: float = 0.97
    k_thermal_C_per_C2: float = 2.0
    aging: AgingModel = field(default_factory=lambda: AgingModel(AgingParams()))

    # ------------------------------------------------------------------
    def effective_c_rate(self, protocol: ChargeProtocol,
                         pack_energy_kWh: float) -> float:
        """Constant-current C-rate = power-into-pack / pack energy."""
        conv = (self.dc_conversion_efficiency if protocol.is_dc
                else self.obc_efficiency)
        p_batt = protocol.wall_power_kW * conv
        return float(p_batt / max(pack_energy_kWh, 1e-9))

    def _cell_ohmic_efficiency(self, c_rate: float, r0_ohm: float,
                               q_nom_Ah: float, v_nom: float) -> float:
        """Fraction of delivered energy retained after cell I²R loss."""
        i = c_rate * q_nom_Ah
        loss_frac = i * r0_ohm / max(v_nom, 1e-6)
        return float(np.clip(1.0 - loss_frac, 0.5, 1.0))

    def peak_temperature_C(self, c_rate: float, ambient_C: float) -> float:
        """Peak cell temperature during charge: ambient + k·C² heating."""
        return float(ambient_C + self.k_thermal_C_per_C2 * c_rate ** 2)

    def plating_c_limit(self, temperature_C: float, soc: float) -> float:
        """Max safe charge C-rate before plating (see module :func:`plating_c_limit`)."""
        return plating_c_limit(temperature_C, soc)

    def plating_risk(self, c_rate: float, temperature_C: float,
                     soc_end: float) -> float:
        """Dimensionless plating severity = excess C-rate over the SoC-end limit."""
        limit = self.plating_c_limit(temperature_C, soc_end)
        return float(max(0.0, c_rate - limit) / max(limit, 1e-6))

    # ------------------------------------------------------------------
    def simulate(self, protocol: ChargeProtocol, pack_energy_kWh: float,
                 q_nom_Ah: float, r0_ohm: float, v_nom: float,
                 soh_capacity: float = 1.0) -> ChargeResult:
        """Quantify one charge session end-to-end.

        Parameters
        ----------
        pack_energy_kWh : float
            Rated pack energy at 25 °C [kWh].
        q_nom_Ah, r0_ohm, v_nom : float
            Representative *cell* nominal capacity, DC resistance, and nominal
            voltage — used for ohmic loss, plating, and throughput.
        soh_capacity : float
            Present capacity retention (scales usable energy this session).
        """
        c_rate = self.effective_c_rate(protocol, pack_energy_kWh)
        d_soc = max(0.0, protocol.soc_end - protocol.soc_start)
        soc_avg = 0.5 * (protocol.soc_start + protocol.soc_end)

        # ---- Duration: CC segment at C, then a CV taper above the cutoff ----
        cc_span = max(0.0, min(protocol.soc_end, protocol.cc_cutoff_soc)
                      - protocol.soc_start)
        cv_span = max(0.0, protocol.soc_end - max(protocol.soc_start,
                                                  protocol.cc_cutoff_soc))
        # CV current tapers ~exponentially → ~2.5× the equivalent CC time.
        hours = (cc_span + 2.5 * cv_span) / max(c_rate, 1e-9)
        duration_min = float(hours * 60.0)

        # ---- Energy accounting ----------------------------------------------
        usable_kWh = pack_energy_kWh * soh_capacity
        energy_stored = float(d_soc * usable_kWh)
        eta_cell = self._cell_ohmic_efficiency(c_rate, r0_ohm, q_nom_Ah, v_nom)
        eta_conv = (self.dc_conversion_efficiency if protocol.is_dc
                    else self.obc_efficiency)
        energy_delivered = energy_stored / max(eta_cell, 1e-6)   # at pack terminals
        wall_energy = energy_delivered / max(eta_conv, 1e-6)     # from the wall
        efficiency = float(energy_stored / max(wall_energy, 1e-9))

        # ---- Thermal + plating ----------------------------------------------
        peak_T = self.peak_temperature_C(c_rate, protocol.ambient_C)
        plating = self.plating_risk(c_rate, protocol.ambient_C, protocol.soc_end)

        # ---- Degradation this session ---------------------------------------
        throughput_efc = float(d_soc)   # one partial cycle of depth d_soc
        d_cap, d_res = self.aging.charge_fade(
            c_rate=c_rate, temperature_C=peak_T, dod=d_soc,
            soc_avg=soc_avg, throughput_efc=throughput_efc, plating=plating,
        )

        return ChargeResult(
            method=protocol.method.value,
            c_rate=c_rate,
            duration_min=duration_min,
            energy_delivered_kWh=energy_delivered,
            energy_stored_kWh=energy_stored,
            wall_energy_kWh=wall_energy,
            efficiency=efficiency,
            peak_cell_temp_C=peak_T,
            plating_risk=plating,
            capacity_fade_pct=100.0 * d_cap,
            resistance_growth_pct=100.0 * d_res,
            throughput_efc=throughput_efc,
        )


def compare_methods(pack_energy_kWh: float = 60.0,
                    q_nom_Ah: float = 2.3, r0_ohm: float = 0.025,
                    v_nom: float = 3.7, ambient_C: float = 25.0,
                    soc_start: float = 0.2, soc_end: float = 0.9,
                    model: ChargingModel | None = None):
    """Return a per-method comparison table (requires pandas).

    One row per :class:`ChargeMethod` with C-rate, charge time, efficiency,
    peak temperature, plating risk, and per-session capacity fade — the
    head-to-head "AC vs DC, fast vs slow" numbers.
    """
    import pandas as pd

    model = model or ChargingModel()
    rows = []
    for method in ChargeMethod:
        proto = ChargeProtocol(method=method, ambient_C=ambient_C,
                               soc_start=soc_start, soc_end=soc_end)
        r = model.simulate(proto, pack_energy_kWh, q_nom_Ah, r0_ohm, v_nom)
        rows.append({
            "method": r.method,
            "power_kW": proto.wall_power_kW,
            "path": "DC" if proto.is_dc else "AC",
            "C_rate": round(r.c_rate, 2),
            "charge_time_min": round(r.duration_min, 1),
            "efficiency_%": round(100.0 * r.efficiency, 1),
            "peak_temp_C": round(r.peak_cell_temp_C, 1),
            "plating_risk": round(r.plating_risk, 2),
            "fade_%/session": round(r.capacity_fade_pct, 4),
        })
    return pd.DataFrame(rows).set_index("method")


def plating_c_limit(temperature_C: float, soc: float) -> float:
    """Max safe charge C-rate before lithium-plating onset.

    Falls with low temperature (slow Li diffusion) and high SoC (anode nearly
    full): ~2.5C at 25 °C / mid-SoC, dropping to ~0.3C at 0 °C or SoC ≈ 1.  Used
    both by the charging model and by SoH-aware control to cap charge current.
    """
    t_factor = float(np.clip((temperature_C + 10.0) / 35.0, 0.1, 1.3))
    soc_factor = float(np.clip(1.0 - soc, 0.1, 1.0))
    return float(2.5 * t_factor * (0.4 + 0.6 * soc_factor))
