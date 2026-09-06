"""
Mechanical / gas failure modes — the class of faults that voltage- and
temperature-only BMS logic misses: internal pressure build-up, cell swelling,
gas venting, microfracture internal shorts, and electrolyte leaks.

Why pressure leads temperature
------------------------------
Gas generation accelerates with temperature (Arrhenius, above an onset), so the
headspace pressure ``P ≈ P₀ + n_gas·R·T / V`` climbs **super-linearly** as a cell
heats — *both* the gas amount and ``T`` rise together.  Its rate ``dP/dt`` spikes
**before** the temperature reaches the runaway threshold, which is exactly why
pressure / gas sensing gives earlier warning than temperature alone.  Crossing a
safety-vent threshold expels gas (a pressure drop + an H₂/VOC puff) and releases
an exotherm back into the thermal model.

Detection signatures
--------------------
* pressure / ``dP/dt`` / H₂ ppm → gas venting & imminent runaway (leads temperature)
* swelling / strain            → gas build-up
* coulombic efficiency < 1     → microfracture internal short (self-discharge)
* capacity fade + R₀ rise      → electrolyte leak / dry-out (use EIS + passport)

Parameters are documented, physically-motivated engineering estimates tuned for
simulation on second-scale steps — recalibrate against real pressure/gas data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .faults import FaultMode

_T0_K = 273.15
_T_REF_K = 298.15
_P_ATM_KPA = 101.325


# ======================================================================
@dataclass
class MechanicalParams:
    """Gas-generation, pressure, swelling, and venting parameters (per cell)."""

    p0_kPa: float = _P_ATM_KPA
    # Gas generation: baseline rate, an Arrhenius factor, and an extra
    # acceleration once past the onset temperature (SEI / electrolyte breakdown).
    gas_base_rate: float = 1.0e-6          # mol/s at 25 °C
    gas_onset_C: float = 35.0
    gas_arrhenius_K: float = 7000.0
    gas_accel_per_C: float = 0.08          # extra fractional rate per °C above onset
    # Pressure gain P = p0 + gas_moles · gain · (T/T_ref)  (gain folds in R·T_ref/V).
    gas_pressure_gain_kPa_per_mol: float = 1.3e5
    h2_fraction: float = 0.25              # H₂ share of generated gas
    h2_ppm_gain: float = 8.0e6             # ppm per mol of H₂ (lumped headspace)
    # Venting.
    vent_pressure_kPa: float = 350.0
    vent_release_frac: float = 0.7
    vent_exotherm_J: float = 500.0
    # Swelling.
    swell_mm_per_kPa: float = 0.004        # from overpressure
    swell_mm_per_soc: float = 0.4          # reversible with SoC


@dataclass
class CellMechanicalState:
    """Evolving mechanical state of one cell."""

    pressure_kPa: float = _P_ATM_KPA
    gas_moles: float = 0.0
    swelling_mm: float = 0.0
    h2_ppm: float = 0.0
    vented: bool = False
    vent_event: bool = False               # True only on the step a vent occurs


# ======================================================================
@dataclass
class PressureModel:
    """Couples cell temperature/SoC to internal pressure, gas, and swelling."""

    params: MechanicalParams = field(default_factory=MechanicalParams)

    def gas_rate_mol_s(self, temperature_C: float) -> float:
        """Arrhenius gas-generation rate, accelerated above the onset temperature."""
        p = self.params
        arrhenius = np.exp(p.gas_arrhenius_K * (1.0 / _T_REF_K - 1.0 / (temperature_C + _T0_K)))
        rate = p.gas_base_rate * float(arrhenius)
        if temperature_C > p.gas_onset_C:
            rate *= 1.0 + p.gas_accel_per_C * (temperature_C - p.gas_onset_C)
        return float(rate)

    def _pressure(self, gas_moles: float, temperature_C: float) -> float:
        p = self.params
        return float(p.p0_kPa + gas_moles * p.gas_pressure_gain_kPa_per_mol
                     * (temperature_C + _T0_K) / _T_REF_K)

    def update(self, state: CellMechanicalState, temperature_C: float,
               soc: float, dt: float, extra_gas_mol: float = 0.0) -> CellMechanicalState:
        """Advance one step.

        ``extra_gas_mol`` injects additional gas this step (e.g. an internal
        short or abuse event dumping gas straight into the headspace).
        """
        p = self.params
        state.vent_event = False
        state.gas_moles += self.gas_rate_mol_s(temperature_C) * dt + max(0.0, extra_gas_mol)
        state.pressure_kPa = self._pressure(state.gas_moles, temperature_C)

        over_p = max(0.0, state.pressure_kPa - p.p0_kPa)
        state.swelling_mm = p.swell_mm_per_kPa * over_p + p.swell_mm_per_soc * soc
        state.h2_ppm = p.h2_fraction * state.gas_moles * p.h2_ppm_gain

        if state.pressure_kPa >= p.vent_pressure_kPa and not state.vented:
            state.gas_moles *= (1.0 - p.vent_release_frac)
            state.pressure_kPa = self._pressure(state.gas_moles, temperature_C)
            state.h2_ppm = p.h2_fraction * state.gas_moles * p.h2_ppm_gain
            state.vented = True
            state.vent_event = True
        return state

    def vent_heat_J(self, state: CellMechanicalState) -> float:
        """Exotherm released on the step a vent occurs (feed to the thermal model)."""
        return self.params.vent_exotherm_J if state.vent_event else 0.0


# ======================================================================
@dataclass
class MechanicalFaultDetector:
    """Rule detector for mechanical / gas faults (safety-critical → rule-based).

    ``predict_step`` returns ``(FaultMode, "rule")``.  Because pressure and gas
    rise before temperature, this trips earlier than a temperature-threshold
    rule — the whole point of the module.
    """

    warn_pressure_kPa: float = 220.0
    dpdt_warn_kPa_s: float = 6.0
    h2_ppm_limit: float = 4.0e5
    swelling_limit_mm: float = 1.6
    ce_short_threshold: float = 0.99

    def predict_step(self, state: CellMechanicalState,
                     prev_pressure_kPa: float | None = None, dt: float = 1.0,
                     coulombic_efficiency: float | None = None) -> tuple[str, str]:
        # Gas venting / imminent runaway — pressure level, rate, or H₂.
        dpdt = (0.0 if prev_pressure_kPa is None
                else (state.pressure_kPa - prev_pressure_kPa) / max(dt, 1e-9))
        if (state.pressure_kPa >= self.warn_pressure_kPa
                or dpdt >= self.dpdt_warn_kPa_s
                or state.h2_ppm >= self.h2_ppm_limit
                or state.vent_event):
            return FaultMode.GAS_VENTING.value, "rule"
        # Microfracture internal short — abnormal self-discharge.
        if (coulombic_efficiency is not None
                and coulombic_efficiency < self.ce_short_threshold):
            return FaultMode.INTERNAL_SHORT.value, "rule"
        # Mechanical over-swelling.
        if state.swelling_mm >= self.swelling_limit_mm:
            return FaultMode.SWELLING.value, "rule"
        return FaultMode.NONE.value, "none"


def coulombic_efficiency(charge_Ah: float, discharge_Ah: float) -> float:
    """Coulombic efficiency = discharge Ah / charge Ah (from the passport totals).

    A soft internal short leaks charge, pulling CE below 1.  Returns 1.0 when no
    charge has been recorded yet.
    """
    if charge_Ah <= 1e-9:
        return 1.0
    return float(min(1.0, discharge_Ah / charge_Ah))
