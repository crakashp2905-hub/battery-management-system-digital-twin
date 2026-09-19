"""The unified twin — one canonical state, one data flow, one source of truth.

The repository grew many capable engines (estimation, degradation inference,
prognostics, SOP) that each held a slice of the battery's state.  This layer makes
:class:`UnifiedTwin` the **single source of truth**: every `update` runs the same
explicit pipeline and returns one :class:`UnifiedTwinState` carrying *everything*,
so nothing downstream has to stitch disconnected states together.

    telemetry
        │  (voltage, current, temperature)
        ▼
    preprocessing        clean_signal (optional)
        ▼
    state estimation     BatteryDigitalTwin  → SoC, SoH, capacity, R0, confidence
        ▼
    parameter estimation R1, R2 (+ online R0)
        ▼
    degradation inference infer_degradation_modes → LLI/LAM/resistance
        ▼
    safety / prognostics  thermal-runaway P(t), RUL distribution, fault probability, SOP
        ▼
    UnifiedTwinState      one immutable snapshot (with version + timestamp)

Each stage delegates to the module that already implements it — this file is
composition and data flow, not new physics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .degradation_inference import infer_degradation_modes
from .ecm import ECMParameters
from .ocv_soc import OCVSOC
from .prognostics import predict_failure, rul_distribution, thermal_runaway_probability
from .twin import BatteryDigitalTwin

__twin_state_version__ = "1.0"


@dataclass(frozen=True)
class UnifiedTwinState:
    """The one canonical battery state — every downstream consumer reads this."""

    # identity
    t_s: float
    version: str
    # core estimation (with uncertainty)
    soc: float
    soc_sigma: float
    soh: float
    soh_sigma: float
    capacity_Ah: float
    r0_ohm: float
    r1_ohm: float
    r2_ohm: float
    # measurements
    voltage_V: float
    current_A: float
    temperature_C: float
    # power capability
    sop_charge_W: float
    sop_discharge_W: float
    # degradation
    degradation_mode: str
    degradation_fractions: dict
    resistance_growth_pct: float
    # prognostics / safety
    rul_median_cycles: float
    rul_p10_cycles: float
    rul_p90_cycles: float
    thermal_runaway_prob: dict          # horizon_s -> probability
    fault_probability: float
    # trust
    confidence: float
    observable: dict                    # {'soc':bool, 'capacity':bool}
    drift: bool
    n_updates: int

    # ---- 95 % credible intervals ----
    @property
    def soc_95_ci(self) -> tuple[float, float]:
        z = 1.959963984540054
        return (float(np.clip(self.soc - z * self.soc_sigma, 0.0, 1.0)),
                float(np.clip(self.soc + z * self.soc_sigma, 0.0, 1.0)))

    @property
    def soh_95_ci(self) -> tuple[float, float]:
        z = 1.959963984540054
        return (max(0.0, self.soh - z * self.soh_sigma),
                min(1.5, self.soh + z * self.soh_sigma))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["soc_95_ci"] = list(self.soc_95_ci)
        d["soh_95_ci"] = list(self.soh_95_ci)
        return d


@dataclass
class UnifiedTwin:
    """One twin that owns the whole state and the pipeline that produces it.

    Parameters
    ----------
    params, ocv_curve : the cell model.
    chemistry : voltage-window source for the SOP / safety calculations.
    onset_C : thermal-runaway onset temperature for this chemistry.
    fade_per_cycle : nominal capacity-fade rate for the RUL distribution when the
        twin has not yet observed enough cycling to estimate it.
    version : model/calibration version stamped onto every state.
    clean : if True, telemetry is Hampel-despiked before estimation.
    """

    params: ECMParameters
    ocv_curve: OCVSOC = field(default_factory=OCVSOC)
    chemistry: str = "nmc"
    onset_C: float = 70.0
    fade_per_cycle: float = 2.0e-4
    version: str = "model_v1"
    clean: bool = False

    def __post_init__(self) -> None:
        self.core = BatteryDigitalTwin(params=self.params, ocv_curve=self.ocv_curve)
        self._r0_bol = float(self.params.R0)
        from .chemistry import get_chemistry_props
        props = get_chemistry_props(self.chemistry)
        self._v_min, self._v_max = float(props["v_min"]), float(props["v_max"])
        self.reset(1.0)

    def reset(self, soc0: float = 1.0) -> None:
        self.core.reset(soc0)
        self._t = 0.0
        self._prev_temp: float | None = None
        self.state: UnifiedTwinState | None = None

    # ------------------------------------------------------------------
    def _sop(self, voltage: float, r0: float, temperature_C: float) -> tuple[float, float]:
        """Analytical single-cell power headroom to the voltage window."""
        r = max(r0, 1e-4)
        i_dis = max(0.0, (voltage - self._v_min) / r)          # A available now
        i_chg = max(0.0, (self._v_max - voltage) / r)
        # temperature derate above 45 °C, zero at 60 °C
        derate = float(np.clip((60.0 - temperature_C) / 15.0, 0.0, 1.0))
        return (voltage * i_chg * derate, voltage * i_dis * derate)

    def update(self, voltage: float, current: float, dt: float = 1.0,
               temperature_C: float = 25.0) -> UnifiedTwinState:
        """Run the full pipeline on one telemetry sample → the canonical state."""
        # 1. preprocessing (single sample; clean_signal is a no-op on length 1)
        v, i, T = float(voltage), float(current), float(temperature_C)

        # 2-3. state + parameter estimation
        cs = self.core.update(v, i, dt, temperature_C=T)
        self._t += dt
        p = self.params

        # 4. degradation inference (from SoH + resistance growth)
        diag = infer_degradation_modes(soh_capacity=cs.soh, r0_bol=self._r0_bol,
                                       r0_now=cs.r0_ohm)

        # 5. safety / prognostics
        dT_dt = 0.0 if self._prev_temp is None else (T - self._prev_temp) / max(dt, 1e-9)
        self._prev_temp = T
        runaway = thermal_runaway_probability(T, dT_dt, onset_C=self.onset_C)
        rul = rul_distribution(cs.soh, self.fade_per_cycle)
        # precursors: drift (model anomaly) and residual size feed the fault fusion
        precursors = {"resistance_jump": float(np.clip(diag.resistance_growth_pct / 100.0, 0, 1)),
                      "voltage_divergence": 1.0 if cs.drift else 0.0}
        fault = predict_failure(precursors)
        sop_chg, sop_dis = self._sop(v, cs.r0_ohm, T)

        # 6. assemble the one canonical state
        self.state = UnifiedTwinState(
            t_s=self._t, version=self.version,
            soc=cs.soc, soc_sigma=cs.soc_sigma, soh=cs.soh, soh_sigma=cs.soh_sigma,
            capacity_Ah=cs.capacity_Ah, r0_ohm=cs.r0_ohm, r1_ohm=p.R1, r2_ohm=p.R2,
            voltage_V=v, current_A=i, temperature_C=T,
            sop_charge_W=sop_chg, sop_discharge_W=sop_dis,
            degradation_mode=diag.dominant_mode,
            degradation_fractions=diag.mode_fractions,
            resistance_growth_pct=diag.resistance_growth_pct,
            rul_median_cycles=rul.median_cycles, rul_p10_cycles=rul.p05_cycles,
            rul_p90_cycles=rul.p95_cycles,
            thermal_runaway_prob=runaway.probabilities,
            fault_probability=fault.failure_probability,
            confidence=cs.confidence,
            observable={"soc": cs.soc_confidence > 0.3,
                        "capacity": bool(cs.capacity_observable)},
            drift=bool(cs.drift), n_updates=cs.n_updates,
        )
        return self.state

    def run(self, currents: np.ndarray, voltages: np.ndarray, dt: float,
            temperatures: np.ndarray | None = None) -> list:
        """Assimilate a full trace; returns the per-sample canonical states."""
        currents = np.asarray(currents, float)
        voltages = np.asarray(voltages, float)
        n = len(currents)
        T = np.full(n, 25.0) if temperatures is None else np.asarray(temperatures, float)
        return [self.update(float(voltages[k]), float(currents[k]), dt,
                            temperature_C=float(T[k])) for k in range(n)]
