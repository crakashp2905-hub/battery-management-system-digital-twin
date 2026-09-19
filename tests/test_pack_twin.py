"""Tests: the cell-level pack twin (which cell is the problem)."""

from __future__ import annotations

import bms


def _pack():
    # Scatter guarantees heterogeneous cells so a limiting/weakest cell exists.
    return bms.BatteryPack(bms.PackConfig(n_cells=6, capacity_sigma=0.05,
                                          r0_sigma=0.1, initial_soc_sigma=0.06, seed=1))


class TestPackTwin:
    def test_per_cell_state_covers_every_cell(self):
        pack = _pack()
        state = bms.PackTwin(pack).update()
        assert state.n_cells == pack.total_cells
        assert len(state.cells) == pack.total_cells
        assert all(0.0 <= c.soc <= 1.0 for c in state.cells)

    def test_limiting_cell_is_the_lowest_soc(self):
        pack = _pack()
        twin = bms.PackTwin(pack)
        state = twin.update()
        limiting = state.cells[state.limiting_cell]
        assert limiting.is_limiting
        assert limiting.soc == min(c.soc for c in state.cells)
        assert twin.limiting_cell().index == state.limiting_cell

    def test_weakest_cell_is_the_lowest_capacity(self):
        state = bms.PackTwin(_pack()).update()
        weakest = state.cells[state.weakest_cell]
        assert weakest.is_weakest
        assert weakest.capacity_Ah == min(c.capacity_Ah for c in state.cells)
        # Pack SoH is bounded by the weakest link, not the average.
        assert abs(state.pack_soh - weakest.soh) < 1e-9

    def test_imbalance_and_deviations_are_consistent(self):
        state = bms.PackTwin(_pack()).update()
        socs = [c.soc for c in state.cells]
        assert abs(state.soc_imbalance - (max(socs) - min(socs))) < 1e-9
        # SoC deviations are measured from the pack mean and roughly cancel.
        assert abs(sum(c.soc_deviation for c in state.cells)) < 1e-6

    def test_serialises(self):
        d = bms.PackTwin(_pack()).update().to_dict()
        for key in ("n_cells", "pack_soc", "pack_soh", "limiting_cell", "cells"):
            assert key in d
