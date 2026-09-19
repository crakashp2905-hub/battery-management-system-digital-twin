"""Cell-level pack twin — *which* cell is the problem, not just a pack number.

A pack-average "SoH 82 %" hides the fact a real pack is only as good as its
weakest cell.  :class:`PackTwin` maintains a state **per individual cell** across
the series/parallel stack and surfaces the two cells an engineer actually cares
about:

* the **limiting cell** — the lowest-SoC cell, which reaches empty (or full)
  first and therefore bounds the pack's usable capacity right now; and
* the **weakest cell** — the lowest-capacity cell, the most-aged link that will
  drive pack end-of-life.

It also reports per-cell SoH, resistance, voltage and how far each cell deviates
from the pack mean (the imbalance fingerprint), so a diagnosis can say
*"cell 17 is becoming the limiting cell"* instead of *"pack SoH = 82 %"*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pack import BatteryPack


@dataclass(frozen=True)
class CellState:
    """State of one individual cell within the pack."""

    index: int                # global cell index (0-based, across all groups)
    group: int                # series-group index it belongs to
    soc: float
    soh: float                # capacity retention vs nominal (Q / Q_nominal)
    capacity_Ah: float
    r0_ohm: float
    voltage_V: float
    soc_deviation: float      # this cell's SoC minus the pack-mean SoC
    is_limiting: bool         # lowest SoC → bounds usable capacity now
    is_weakest: bool          # lowest capacity → most-aged link


@dataclass(frozen=True)
class PackTwinState:
    """A snapshot of every cell plus the pack-level roll-up."""

    cells: list
    n_cells: int
    pack_soc: float
    pack_soh: float           # limited by the weakest cell (min capacity retention)
    soc_imbalance: float      # max − min cell SoC
    limiting_cell: int        # global index of the lowest-SoC cell
    weakest_cell: int         # global index of the lowest-capacity cell

    def to_dict(self) -> dict:
        return {
            "n_cells": self.n_cells, "pack_soc": self.pack_soc,
            "pack_soh": self.pack_soh, "soc_imbalance": self.soc_imbalance,
            "limiting_cell": self.limiting_cell, "weakest_cell": self.weakest_cell,
            "cells": [c.__dict__ for c in self.cells],
        }


@dataclass
class PackTwin:
    """A per-cell twin over a :class:`bms.BatteryPack`.

    Parameters
    ----------
    pack : BatteryPack
    nominal_capacity_Ah : float, optional
        Beginning-of-life per-cell capacity for SoH normalisation. Defaults to
        the pack's configured nominal.
    """

    pack: BatteryPack
    nominal_capacity_Ah: float | None = None

    def __post_init__(self) -> None:
        self.nominal_capacity_Ah = float(
            self.nominal_capacity_Ah if self.nominal_capacity_Ah is not None
            else self.pack.cfg.nominal_capacity_Ah)

    def update(self, temperatures_C: np.ndarray | None = None) -> PackTwinState:
        """Read the pack's current state into a per-cell snapshot."""
        groups = self.pack.groups
        v_groups = self.pack.cell_voltages(temperatures_C)

        socs, caps = [], []
        raw = []            # (group_idx, soc, cap, r0, voltage)
        for gi, g in enumerate(groups):
            for c in g.cells:
                socs.append(float(c.soc))
                caps.append(float(c.params.Q_nom_Ah))
                raw.append((gi, float(c.soc), float(c.params.Q_nom_Ah),
                            float(c.params.R0), float(v_groups[gi])))
        socs = np.asarray(socs, float)
        caps = np.asarray(caps, float)
        mean_soc = float(np.mean(socs))
        limiting = int(np.argmin(socs))          # lowest SoC → empties first
        weakest = int(np.argmin(caps))           # lowest capacity → most aged
        nom = self.nominal_capacity_Ah

        cells = []
        for i, (gi, soc, cap, r0, v) in enumerate(raw):
            cells.append(CellState(
                index=i, group=gi, soc=soc, soh=float(cap / max(nom, 1e-9)),
                capacity_Ah=cap, r0_ohm=r0, voltage_V=v,
                soc_deviation=float(soc - mean_soc),
                is_limiting=(i == limiting), is_weakest=(i == weakest),
            ))

        return PackTwinState(
            cells=cells, n_cells=len(cells), pack_soc=mean_soc,
            pack_soh=float(np.min(caps) / max(nom, 1e-9)),   # weakest link
            soc_imbalance=float(socs.max() - socs.min()),
            limiting_cell=limiting, weakest_cell=weakest,
        )

    def limiting_cell(self, temperatures_C: np.ndarray | None = None) -> CellState:
        """The cell that currently bounds the pack's usable capacity."""
        state = self.update(temperatures_C)
        return state.cells[state.limiting_cell]
