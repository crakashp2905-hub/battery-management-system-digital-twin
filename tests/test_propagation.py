"""Tests: thermal-runaway propagation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from bms.propagation import PropagationParams, RunawayPropagation, propagation_arrested_below


class TestRunawayPropagation:
    def test_strong_coupling_cascades_down_the_module(self):
        sim = RunawayPropagation(6, PropagationParams(coupling_W_per_K=1.5))
        out = sim.simulate(trigger_cell=0, duration_s=600, dt=0.5)
        assert out["n_ignited"] == 6           # whole module ignites
        assert out["propagated"] is True
        # Ignition is sequential away from the trigger (a travelling front).
        times = out["ignition_times_s"]
        assert times[0] == 0.0
        assert np.all(np.diff(times) > 0)      # each cell later than the last

    def test_thermal_barrier_arrests_propagation(self):
        sim = RunawayPropagation(6, PropagationParams(coupling_W_per_K=0.1))
        out = sim.simulate(trigger_cell=0, duration_s=600, dt=0.5)
        assert out["n_ignited"] == 1           # only the triggered cell
        assert out["propagated"] is False

    def test_only_triggered_cell_ignites_with_no_coupling(self):
        sim = RunawayPropagation(4, PropagationParams(coupling_W_per_K=0.0))
        out = sim.simulate(trigger_cell=1, duration_s=400, dt=0.5)
        assert out["ignited"].tolist() == [False, True, False, False]

    def test_arrested_below_is_a_positive_threshold(self):
        base = PropagationParams()
        k_safe = propagation_arrested_below(6, base)
        assert 0.0 <= k_safe < base.coupling_W_per_K
        # At the reported safe coupling, propagation really is arrested.
        if k_safe > 0.0:
            out = RunawayPropagation(6, replace(base, coupling_W_per_K=k_safe)).simulate()
            assert out["n_ignited"] == 1

    def test_temperatures_are_bounded_by_the_clamp(self):
        base = PropagationParams(peak_clamp_C=800.0)
        out = RunawayPropagation(5, base).simulate(trigger_cell=0, duration_s=300, dt=0.5)
        assert out["temperatures"].max() <= 800.0 + 1e-6
        assert out["peak_C"] > base.runaway_onset_C   # it did get hot
