"""State-of-power (SOP) limits for a series battery pack.

The calculator derives conservative traction and regenerative-braking limits
at multiple time horizons.  It accounts for each series group's measured
voltage, present current, temperature-adjusted 2-RC impedance, configured
current limits, and a temperature derating envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pack import BatteryPack


@dataclass(frozen=True)
class SOPLimit:
    """Allowed pack power/current for one horizon (positive magnitudes)."""

    horizon_s: float
    traction_power_W: float
    regen_power_W: float
    traction_current_A: float
    regen_current_A: float
    temperature_derate: float


@dataclass
class SOPConfig:
    """Electrical and thermal constraints used by :class:`StateOfPower`."""

    horizons_s: tuple[float, ...] = (2.0, 10.0, 30.0)
    max_discharge_current_A: float = float("inf")
    max_charge_current_A: float = float("inf")
    temperature_warning_C: float = 45.0
    temperature_limit_C: float = 60.0


@dataclass
class StateOfPower:
    """Calculate voltage-, current-, and temperature-constrained SOP.

    Current is positive on discharge.  The supplied ``cell_voltages_V`` are
    the present group terminal voltages at ``pack_current_A``; this permits
    the calculation to project from an already-loaded pack rather than
    assuming a rested battery.
    """

    config: SOPConfig = field(default_factory=SOPConfig)

    def _temperature_derate(self, temperatures_C: np.ndarray) -> float:
        t_max = float(np.max(temperatures_C))
        cfg = self.config
        if t_max <= cfg.temperature_warning_C:
            return 1.0
        if t_max >= cfg.temperature_limit_C:
            return 0.0
        return float((cfg.temperature_limit_C - t_max)
                     / (cfg.temperature_limit_C - cfg.temperature_warning_C))

    @staticmethod
    def _horizon_resistance(group, temperature_C: float, horizon_s: float) -> float:
        """Step-response resistance of a temperature-adjusted 2-RC ECM."""
        p = group.params.at_temperature(float(temperature_C))
        r1 = p.R1 * (1.0 - np.exp(-horizon_s / max(p.tau1, 1e-9)))
        r2 = p.R2 * (1.0 - np.exp(-horizon_s / max(p.tau2, 1e-9)))
        return float(p.R0 + r1 + r2)

    def calculate(
        self,
        pack: BatteryPack,
        temperatures_C: np.ndarray | None = None,
        pack_current_A: float = 0.0,
        cell_voltages_V: np.ndarray | None = None,
    ) -> dict[float, SOPLimit]:
        """Return limits keyed by horizon in seconds.

        ``traction_*`` and ``regen_*`` are non-negative usable magnitudes.
        Regen current is reported as a magnitude even though the underlying
        pack-current convention represents charging with a negative value.
        """
        n = pack.n_cells
        temperatures = (np.full(n, 25.0) if temperatures_C is None
                        else np.asarray(temperatures_C, dtype=float))
        if temperatures.shape != (n,):
            raise ValueError("temperatures_C must have one value per series group")
        # When terminal voltages are not supplied, evaluate them at the present
        # pack current (not no-load) so the projection stays self-consistent
        # with pack_current_A.  At pack_current_A == 0 this equals the rested
        # terminal voltage, so the common case is unchanged.
        if cell_voltages_V is None:
            voltages = np.array([
                pack.groups[i].terminal_voltage(
                    pack.ocv_curve, float(temperatures[i]),
                    group_current=pack_current_A)
                for i in range(n)
            ])
        else:
            voltages = np.asarray(cell_voltages_V, dtype=float)
        if voltages.shape != (n,):
            raise ValueError("cell_voltages_V must have one value per series group")

        from .chemistry import get_chemistry_props
        props = get_chemistry_props(pack.cfg.chemistry)
        v_min, v_max = float(props["v_min"]), float(props["v_max"])
        derate = self._temperature_derate(temperatures)
        i_dis_cap = self.config.max_discharge_current_A * derate
        i_chg_cap = self.config.max_charge_current_A * derate

        limits: dict[float, SOPLimit] = {}
        for horizon in self.config.horizons_s:
            if horizon <= 0:
                raise ValueError("SOP horizons must be positive")
            r = np.array([self._horizon_resistance(g, temperatures[i], horizon)
                          for i, g in enumerate(pack.groups)])

            # V_future = V_now - R_h * (I_future - I_now).  The weakest
            # series group establishes both discharge and charge limits.
            i_dis_voltage = float(np.min(pack_current_A + (voltages - v_min) / r))
            i_chg_voltage = float(np.min((v_max - voltages) / r - pack_current_A))
            i_dis = max(0.0, min(i_dis_voltage, i_dis_cap))
            i_chg = max(0.0, min(i_chg_voltage, i_chg_cap))

            v_dis = float(np.sum(voltages - r * (i_dis - pack_current_A)))
            v_chg = float(np.sum(voltages + r * (i_chg + pack_current_A)))
            limits[float(horizon)] = SOPLimit(
                horizon_s=float(horizon),
                traction_power_W=max(0.0, v_dis * i_dis),
                regen_power_W=max(0.0, v_chg * i_chg),
                traction_current_A=i_dis,
                regen_current_A=i_chg,
                temperature_derate=derate,
            )
        return limits
