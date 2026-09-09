"""Cell-to-cell thermal-runaway propagation in a module.

Single-cell venting (:mod:`bms.mechanics`) answers *does this cell run away?*.
The module-level question is *does one cell's runaway cascade to its
neighbours?* — the failure that turns a single defect into a module fire.

Each cell is a lumped thermal mass coupled to its neighbours and to ambient::

    C · dT_i/dt = Q_exo,i(t) + Σ_j k·(T_j − T_i) − h·(T_i − T_amb)

A cell **ignites** when its temperature crosses ``runaway_onset_C``; from then it
releases ``exotherm_J`` over ``ignite_duration_s``, heating its neighbours.  With
enough coupling the front propagates down the module; a **thermal barrier** (low
``coupling_W_per_K``) or more spacing arrests it — which is exactly the
mitigation this model lets you size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PropagationParams:
    """Lumped cell-to-cell thermal-runaway parameters (per 18650-class cell)."""

    runaway_onset_C: float = 150.0          # ignition temperature
    exotherm_J: float = 40_000.0            # total energy released per cell
    ignite_duration_s: float = 30.0         # release window
    coupling_W_per_K: float = 1.5           # adjacent-cell heat transfer
    cell_heat_capacity_J_per_K: float = 40.0
    convection_W_per_K: float = 0.2         # cell → ambient
    ambient_C: float = 25.0
    peak_clamp_C: float = 800.0             # numerical clamp on runaway temperature


class RunawayPropagation:
    """Simulate runaway propagation across a line of ``n_cells``."""

    def __init__(self, n_cells: int, params: PropagationParams | None = None):
        self.n = int(n_cells)
        self.p = params or PropagationParams()
        self.reset()

    def reset(self) -> None:
        p = self.p
        self.T = np.full(self.n, p.ambient_C, float)
        self._ignited = np.zeros(self.n, bool)
        self._ignite_t = np.full(self.n, np.nan)      # time each cell ignited
        self._t = 0.0

    def _neighbours_heat(self) -> np.ndarray:
        """Conductive exchange with line neighbours [W]."""
        k = self.p.coupling_W_per_K
        q = np.zeros(self.n)
        q[:-1] += k * (self.T[1:] - self.T[:-1])       # from right neighbour
        q[1:] += k * (self.T[:-1] - self.T[1:])        # from left neighbour
        return q

    def step(self, dt: float) -> np.ndarray:
        p = self.p
        # Exotherm power from each currently-igniting cell.
        q_exo = np.zeros(self.n)
        active = self._ignited & (self._t - self._ignite_t < p.ignite_duration_s)
        q_exo[active] = p.exotherm_J / p.ignite_duration_s
        q = q_exo + self._neighbours_heat() - p.convection_W_per_K * (self.T - p.ambient_C)
        self.T = np.minimum(self.T + q * dt / p.cell_heat_capacity_J_per_K, p.peak_clamp_C)
        self._t += dt
        # Newly-ignited cells (crossed onset this step).
        newly = (~self._ignited) & (self.T >= p.runaway_onset_C)
        self._ignite_t[newly] = self._t
        self._ignited |= newly
        return self.T

    def trigger(self, cell_index: int) -> None:
        """Force a cell into runaway (the initiating defect)."""
        self.T[cell_index] = self.p.runaway_onset_C + 1.0
        self._ignite_t[cell_index] = self._t
        self._ignited[cell_index] = True

    def simulate(self, trigger_cell: int = 0, duration_s: float = 600.0,
                 dt: float = 0.5) -> dict:
        """Trigger a cell and run; report which cells ignited and when.

        Returns ``temperatures`` (steps × n), ``ignited`` (bool per cell),
        ``ignition_times_s`` (per cell, ``nan`` if it never ignited),
        ``n_ignited``, and ``propagated`` (did it spread beyond the trigger).
        """
        self.reset()
        self.trigger(trigger_cell)
        n_steps = int(duration_s / dt)
        hist = np.empty((n_steps, self.n))
        for k in range(n_steps):
            hist[k] = self.step(dt)
        n_ignited = int(self._ignited.sum())
        return {
            "temperatures": hist,
            "ignited": self._ignited.copy(),
            "ignition_times_s": self._ignite_t.copy(),
            "n_ignited": n_ignited,
            "propagated": bool(n_ignited > 1),
            "peak_C": float(hist.max()),
        }


def propagation_arrested_below(n_cells: int = 6,
                               params: PropagationParams | None = None,
                               couplings=None) -> float:
    """Sweep coupling and return the largest value at which propagation stops.

    A design aid: below this cell-to-cell coupling (i.e. with enough spacing or a
    good enough barrier) a single-cell runaway does **not** cascade.  Returns the
    highest tested coupling that keeps ``n_ignited == 1`` (``0`` if none does).
    """
    from dataclasses import replace
    base = params or PropagationParams()
    couplings = couplings if couplings is not None else np.linspace(0.05, base.coupling_W_per_K, 12)
    safe = 0.0
    for k in couplings:
        sim = RunawayPropagation(n_cells, replace(base, coupling_W_per_K=float(k)))
        if sim.simulate(trigger_cell=0)["n_ignited"] == 1:
            safe = float(k)
    return safe
