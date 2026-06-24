"""
Synthetic training-data generator for the ML fault detector.

Runs a short simulation under each fault condition (and the nominal case)
and emits a 14-element feature/label table compatible with
`HybridFaultDetector.fit`.  Each training row is:

    [10 base features from extract_features] + [4 rolling stats from RollingFeatureBuffer]

The 4 rolling stats (max V std, max T std, max V drift, max T drift) capture
slow-onset anomalies that are invisible in instantaneous features and make the
classifier substantially more robust to sensor-bias faults.
"""

from __future__ import annotations

import numpy as np

from .data import generate_load_profile
from .faults import (FaultInjector, FaultMode, FaultSpec,
                     RollingFeatureBuffer, extract_features)
from .pack import BatteryPack, PackConfig
from .thermal import ThermalModel, ThermalParameters


def generate_fault_training_data(samples_per_class: int = 500,
                                 dt: float = 1.0, seed: int = 0,
                                 buffer_window: int = 30,
                                 chemistry: str = "nmc",
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Run short scenarios with each fault label and collect 14-dim features.

    Parameters
    ----------
    samples_per_class : int
        Number of labelled time-steps to collect per fault mode.
    dt : float
        Simulation time step [s].
    seed : int
        Master RNG seed for reproducibility.
    buffer_window : int
        Rolling-buffer window length (must match ``HybridFaultDetector``).
    chemistry : str
        Cell chemistry for the training pack (``"nmc"`` or ``"lfp"``).
        Fault-injection thresholds (overcharge voltage, runaway temperature)
        are scaled to the chemistry automatically.
    """
    from .chemistry import get_chemistry_props
    props = get_chemistry_props(chemistry)
    T_runaway_start = props["T_runaway_C"] + 2.0  # just above threshold

    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[str] = []

    fault_modes = [
        FaultMode.NONE,
        FaultMode.OVERCHARGE,
        FaultMode.SHORT_CIRCUIT,
        FaultMode.THERMAL_RUNAWAY,
        FaultMode.SENSOR_DROPOUT,
        FaultMode.SENSOR_BIAS,
    ]
    n_per = samples_per_class

    for mode in fault_modes:
        collected = 0
        attempt = 0
        while collected < n_per and attempt < n_per * 5:
            attempt += 1
            cfg = PackConfig(n_cells=4, chemistry=chemistry,
                             seed=int(rng.integers(0, 1_000_000)))
            pack = BatteryPack(cfg)
            thermal = ThermalModel(n_cells=4, params=ThermalParameters(T_amb_C=25.0))
            buf = RollingFeatureBuffer(window=buffer_window)

            cell_idx = int(rng.integers(0, pack.n_cells))
            specs = []
            if mode != FaultMode.NONE:
                specs.append(FaultSpec(mode=mode, start_step=20, end_step=200,
                                       cell_index=cell_idx,
                                       severity=float(rng.uniform(0.6, 1.4))))
            inj = FaultInjector(specs, chemistry=chemistry)

            if mode == FaultMode.THERMAL_RUNAWAY:
                thermal.T[cell_idx] = float(rng.uniform(T_runaway_start,
                                                          T_runaway_start + 8.0))

            capacity_Ah = pack.capacities_Ah[0]
            T_steps = 220
            i_load = generate_load_profile(T_steps * dt, dt=dt, mode="pulse",
                                           c_rate=1.0, capacity_Ah=capacity_Ah,
                                           seed=int(rng.integers(0, 1_000_000)))

            prev_v = pack.cell_voltages()
            prev_T = thermal.T.copy()
            for k in range(T_steps):
                pack_cur = float(i_load[k])
                cell_currents = np.full(pack.n_cells, pack_cur)
                cell_currents = inj.apply_to_currents(cell_currents, k)
                bal = cell_currents - pack_cur
                step = pack.step(pack_cur, dt, balancing_currents=bal,
                                 cell_temperatures_C=thermal.T)

                # Use group-level params (works correctly for n_parallel≥1)
                ocv = np.array([
                    float(pack.ocv_curve.ocv(g.soc, T_C=float(thermal.T[i])))
                    for i, g in enumerate(pack.groups)
                ])
                R0 = np.array([g.params.R0 for g in pack.groups])

                from .thermal import ThermalModel as _TM
                heat = _TM.heat_generation(cell_currents, R0,
                                           step["v_cells"], ocv)
                thermal.step(heat, cooling_duty=0.0, dt=dt)
                thermal.T[:] = inj.apply_to_temperatures(thermal.T, k, dt)

                v_meas = inj.apply_to_voltage_meas(step["v_cells"], k)
                dv = v_meas - prev_v
                dT = thermal.T - prev_T

                base_feats = extract_features(v_meas, cell_currents, thermal.T, dv, dT)
                buf.push(v_meas, thermal.T)
                full_feats = np.concatenate([base_feats, buf.stats_vector()])

                if 30 <= k < 200 and inj.label(k) == mode and collected < n_per:
                    rows.append(full_feats)
                    labels.append(mode.value)
                    collected += 1
                if mode == FaultMode.NONE and 30 <= k < 200 and collected < n_per:
                    rows.append(full_feats)
                    labels.append(mode.value)
                    collected += 1

                prev_v = v_meas
                prev_T = thermal.T.copy()

    X = np.array(rows)
    y = np.array(labels)
    return X, y
