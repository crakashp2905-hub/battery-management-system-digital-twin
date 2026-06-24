"""
Multi-cell battery pack with configurable series × parallel topology.

Topology
--------
``PackConfig.n_cells``    — number of series groups (was "n_cells" in v1).
``PackConfig.n_parallel`` — number of cells in parallel per series group.

Examples
--------
* 4S1P  → ``PackConfig(n_cells=4)``           (default, backward-compat)
* 6S1P  → ``PackConfig(n_cells=6)``
* 4S2P  → ``PackConfig(n_cells=4, n_parallel=2)``  total 8 cells
* 2S3P  → ``PackConfig(n_cells=2, n_parallel=3)``  total 6 cells

Pack-level quantities
---------------------
* ``pack_voltage()`` — sum of series-group terminal voltages.
* ``pack.soc``       — per-group capacity-weighted average SOC (length n_series).
* ``cell_voltages()`` — per-group terminal voltage (length n_series).
* ``balancing_currents`` must have length n_series (= n_cells) regardless of n_parallel.

Parallel-group physics
----------------------
Within each parallel group the group current is split among cells inversely
proportional to their effective DC resistance (conductance-weighted), which is
a good approximation when cell voltages are close.  The group terminal voltage
is the mean of individual cell terminal voltages.

Chemistry
---------
Pass ``chemistry="nmc"`` (default) or ``chemistry="lfp"`` to automatically
configure OCV tables, Arrhenius constants, and ECM defaults for that chemistry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .ecm import SecondOrderECM, ECMParameters
from .ocv_soc import OCVSOC


# ── Internal parallel-group class ───────────────────────────────────────
class _ParallelGroup:
    """n_parallel cells connected in parallel, exposed as one series node.

    All cells share the same terminal voltage (approximately).  Group current
    is split by conductance weighting (1/R0_i at operating temperature).
    """

    def __init__(self, cells: list[SecondOrderECM]) -> None:
        if not cells:
            raise ValueError("_ParallelGroup requires at least one cell")
        self.cells = cells

    # ── Aggregate state ──────────────────────────────────────────────
    @property
    def n_parallel(self) -> int:
        return len(self.cells)

    @property
    def soc(self) -> float:
        """Capacity-weighted average SOC of all parallel cells."""
        caps = np.array([c.params.Q_nom_Ah for c in self.cells])
        socs = np.array([c.soc for c in self.cells])
        return float(np.dot(socs, caps) / caps.sum())

    @property
    def v_rc1(self) -> float:
        return float(np.mean([c.v_rc1 for c in self.cells]))

    @property
    def v_rc2(self) -> float:
        return float(np.mean([c.v_rc2 for c in self.cells]))

    @property
    def params(self) -> ECMParameters:
        """Effective ECM parameters for the parallel combination.

        * R0_eff = 1 / Σ(1/R0_i)   (parallel resistance, less than any single cell)
        * C1_eff = Σ C1_i           (parallel capacitances add)
        * Q_eff  = Σ Q_nom_i        (total capacity)
        """
        n = self.n_parallel
        if n == 1:
            return self.cells[0].params
        r0_eff = 1.0 / sum(1.0 / max(c.params.R0, 1e-9) for c in self.cells)
        r1_eff = 1.0 / sum(1.0 / max(c.params.R1, 1e-9) for c in self.cells)
        r2_eff = 1.0 / sum(1.0 / max(c.params.R2, 1e-9) for c in self.cells)
        c1_eff = sum(c.params.C1 for c in self.cells)
        c2_eff = sum(c.params.C2 for c in self.cells)
        q_eff = sum(c.params.Q_nom_Ah for c in self.cells)
        p0 = self.cells[0].params
        return ECMParameters(
            R0=r0_eff, R1=r1_eff, C1=c1_eff,
            R2=r2_eff, C2=c2_eff, Q_nom_Ah=q_eff,
            chemistry=p0.chemistry,
        )

    # ── Terminal voltage ─────────────────────────────────────────────
    def terminal_voltage(self, ocv_curve: OCVSOC,
                         temperature_C: float = 25.0) -> float:
        """Mean terminal voltage of all parallel cells [V]."""
        return float(np.mean([
            ocv_curve.ocv(c.soc, T_C=temperature_C) - c.v_rc1 - c.v_rc2
            for c in self.cells
        ]))

    # ── Time step ────────────────────────────────────────────────────
    def step(self, group_current: float, dt: float,
             temperature_C: float = 25.0) -> float:
        """Advance all parallel cells by dt.

        The group current is split among cells in proportion to their
        conductance (1/R0 at *temperature_C*).  Returns the mean terminal
        voltage of all cells after the step.
        """
        n = self.n_parallel
        if n == 1:
            return self.cells[0].step(group_current, dt, temperature_C)

        g = np.array([
            1.0 / max(c.params.at_temperature(temperature_C).R0, 1e-9)
            for c in self.cells
        ])
        fracs = g / g.sum()
        cell_currents = fracs * group_current

        voltages = np.array([
            c.step(float(I_c), dt, temperature_C)
            for c, I_c in zip(self.cells, cell_currents)
        ])
        return float(voltages.mean())


# ── Pack configuration ──────────────────────────────────────────────────
@dataclass
class PackConfig:
    """Configuration for a series × parallel battery pack.

    Parameters
    ----------
    n_cells : int
        Number of *series* groups (backward-compat name; one group per
        thermal/balancing node).  Default 4.
    n_parallel : int
        Number of cells in parallel within each series group.  Default 1
        (pure-series pack — identical to the v1 behaviour).
    chemistry : str
        ``"nmc"`` (default) or ``"lfp"``.  Sets OCV table, Arrhenius
        constant, and ECM defaults for that chemistry.
    nominal_capacity_Ah : float
        Per-cell capacity used when scattering around the nominal.
        Defaults to the chemistry default (2.3 Ah for NMC, 3.2 Ah for LFP)
        when left at 0.0.
    """
    n_cells: int = 4
    n_parallel: int = 1
    nominal_capacity_Ah: float = 0.0       # 0 ⇒ use chemistry default
    capacity_sigma: float = 0.03            # 3 % capacity scatter
    r0_sigma: float = 0.05                  # 5 % resistance scatter
    initial_soc_mean: float = 0.85
    initial_soc_sigma: float = 0.05
    seed: int | None = 42
    chemistry: str = "nmc"

    def __post_init__(self) -> None:
        if self.nominal_capacity_Ah <= 0.0:
            from .chemistry import get_chemistry_props
            self.nominal_capacity_Ah = get_chemistry_props(self.chemistry)[
                "default_capacity_Ah"
            ]


# ── Main pack class ─────────────────────────────────────────────────────
class BatteryPack:
    """Series-parallel stack of SecondOrderECM cells with manufacturing scatter.

    Internally organises cells as *n_series* (= ``n_cells``) parallel groups,
    each containing *n_parallel* individual cells.  All public properties and
    methods expose the series-group view, so existing code written for
    ``n_parallel=1`` continues to work without modification.
    """

    def __init__(self, config: PackConfig | None = None,
                 base_params: ECMParameters | None = None,
                 ocv_curve: OCVSOC | None = None):
        self.cfg = config or PackConfig()

        # ── OCV curve ──────────────────────────────────────────────────
        if ocv_curve is None:
            self.ocv_curve = OCVSOC.from_chemistry(self.cfg.chemistry)
        else:
            self.ocv_curve = ocv_curve

        # ── Base ECM parameters ────────────────────────────────────────
        if base_params is None:
            from .chemistry import get_chemistry_props
            props = get_chemistry_props(self.cfg.chemistry)
            d = props["default_ecm"]
            base = ECMParameters(
                R0=d["R0"], R1=d["R1"], C1=d["C1"],
                R2=d["R2"], C2=d["C2"],
                Q_nom_Ah=self.cfg.nominal_capacity_Ah,
                chemistry=self.cfg.chemistry,
            )
        else:
            base = base_params

        # ── Build cells into series groups ─────────────────────────────
        rng = np.random.default_rng(self.cfg.seed)
        n_s = self.cfg.n_cells
        n_p = self.cfg.n_parallel

        self._groups: list[_ParallelGroup] = []
        for _ in range(n_s):
            group_cells: list[SecondOrderECM] = []
            for _ in range(n_p):
                cap = float(np.clip(
                    rng.normal(self.cfg.nominal_capacity_Ah,
                               self.cfg.capacity_sigma * self.cfg.nominal_capacity_Ah),
                    0.5 * self.cfg.nominal_capacity_Ah,
                    1.5 * self.cfg.nominal_capacity_Ah,
                ))
                r0 = float(max(1e-4, rng.normal(base.R0, self.cfg.r0_sigma * base.R0)))
                p = ECMParameters(
                    R0=r0, R1=base.R1, C1=base.C1,
                    R2=base.R2, C2=base.C2,
                    Q_nom_Ah=cap, chemistry=self.cfg.chemistry,
                )
                soc0 = float(np.clip(
                    rng.normal(self.cfg.initial_soc_mean, self.cfg.initial_soc_sigma),
                    0.05, 1.0,
                ))
                cell = SecondOrderECM(params=p, ocv_curve=self.ocv_curve)
                cell.reset(soc0)
                group_cells.append(cell)
            self._groups.append(_ParallelGroup(group_cells))

    # ── Public properties ───────────────────────────────────────────────
    @property
    def n_cells(self) -> int:
        """Number of series groups (= ``cfg.n_cells``).

        Matches the historical meaning of ``n_cells`` when ``n_parallel=1``.
        Use ``total_cells`` for the count of all individual cells.
        """
        return len(self._groups)

    @property
    def n_parallel(self) -> int:
        """Number of parallel cells per series group."""
        return self.cfg.n_parallel

    @property
    def total_cells(self) -> int:
        """Total number of individual cells (n_series × n_parallel)."""
        return sum(len(g.cells) for g in self._groups)

    @property
    def groups(self) -> list[_ParallelGroup]:
        """Series groups — the primary per-node view."""
        return list(self._groups)

    @property
    def cells(self) -> list[SecondOrderECM]:
        """Representative cells — first cell of each series group.

        For ``n_parallel=1`` this returns all cells (identical to v1
        behaviour).  For ``n_parallel>1`` it returns one representative cell
        per group; use ``groups`` for full per-cell access.
        """
        return [g.cells[0] for g in self._groups]

    @property
    def soc(self) -> np.ndarray:
        """Per-series-group SOC (capacity-weighted for parallel groups)."""
        return np.array([g.soc for g in self._groups])

    @property
    def capacities_Ah(self) -> np.ndarray:
        """Total capacity per series group [Ah] (sum of parallel cells)."""
        return np.array([sum(c.params.Q_nom_Ah for c in g.cells) for g in self._groups])

    # ── Voltage ─────────────────────────────────────────────────────────
    def cell_voltages(self, temperatures_C: np.ndarray | None = None) -> np.ndarray:
        """Instantaneous terminal voltage of each series group [V].

        Parameters
        ----------
        temperatures_C : (n_cells,) array, optional
            Per-group temperature [°C] for OCV correction.  Defaults to 25 °C.
        """
        if temperatures_C is None:
            return np.array([g.terminal_voltage(self.ocv_curve) for g in self._groups])
        return np.array([
            g.terminal_voltage(self.ocv_curve, float(temperatures_C[i]))
            for i, g in enumerate(self._groups)
        ])

    def pack_voltage(self, temperatures_C: np.ndarray | None = None) -> float:
        return float(self.cell_voltages(temperatures_C).sum())

    # ── Time step ────────────────────────────────────────────────────────
    def step(self, pack_current: float, dt: float,
             balancing_currents: Sequence[float] | None = None,
             cell_temperatures_C: np.ndarray | None = None) -> dict:
        """Advance every series group by *dt*.

        Parameters
        ----------
        pack_current : float
            Series current through every group (positive = discharge) [A].
        balancing_currents : (n_cells,) sequence, optional
            Per-group additional current applied by the balancing circuit.
            Length must equal ``n_cells`` (n_series groups).
        cell_temperatures_C : (n_cells,) array, optional
            Per-group temperature [°C].  Defaults to 25 °C.
        """
        n_s = len(self._groups)
        bal = (np.zeros(n_s) if balancing_currents is None
               else np.asarray(balancing_currents, float))
        if len(bal) != n_s:
            raise ValueError(
                f"balancing_currents length mismatch: got {len(bal)}, "
                f"expected {n_s} (n_series groups)"
            )

        T_arr = (np.full(n_s, 25.0) if cell_temperatures_C is None
                 else np.asarray(cell_temperatures_C, float))

        v_cells = np.empty(n_s)
        for i, group in enumerate(self._groups):
            v_cells[i] = group.step(pack_current + bal[i], dt, float(T_arr[i]))

        return {
            "v_cells": v_cells,
            "v_pack": float(v_cells.sum()),
            "soc": self.soc,
        }

    # ── Metrics ──────────────────────────────────────────────────────────
    def soc_imbalance(self) -> float:
        """Max minus min series-group SOC — a simple imbalance metric."""
        s = self.soc
        return float(s.max() - s.min())

    def state_of_energy_Wh(self) -> float:
        """Remaining discharge energy [Wh] estimated from the OCV integral.

        For each series group, integrates V_OC(s) over s ∈ [0, SOC_group]
        and multiplies by the group capacity.  Uses trapezoidal quadrature
        over 100 points; error is typically < 0.1 %.
        """
        n_pts = 100
        total_Wh = 0.0
        for g in self._groups:
            soc_pts = np.linspace(0.0, float(g.soc), n_pts)
            ocv_pts = np.array([float(self.ocv_curve.ocv(s)) for s in soc_pts])
            cap_Ah = sum(c.params.Q_nom_Ah for c in g.cells)
            total_Wh += float(np.trapezoid(ocv_pts, soc_pts)) * cap_Ah
        return total_Wh

    def __repr__(self) -> str:
        return (
            f"BatteryPack({self.cfg.n_cells}S{self.cfg.n_parallel}P "
            f"chemistry={self.cfg.chemistry!r}, "
            f"SOC={np.round(self.soc, 3).tolist()}, "
            f"V_pack={self.pack_voltage():.3f} V)"
        )
