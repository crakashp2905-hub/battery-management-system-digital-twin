"""Tests: fleet digital twins + fleet intelligence."""

from __future__ import annotations

import numpy as np

import bms


def _healthy_fleet(n=20, seed=0):
    rng = np.random.default_rng(seed)
    fleet = bms.FleetTwin()
    for i in range(n):
        fleet.add(bms.VehicleState(
            vehicle_id=f"veh{i:02d}",
            soh=float(np.clip(rng.normal(0.92, 0.02), 0.7, 1.0)),
            equivalent_full_cycles=float(rng.uniform(300, 600))))
    return fleet


class TestFleetTwin:
    def test_insight_summarises_population(self):
        ins = _healthy_fleet().insight()
        assert ins.n == 20
        assert 0.8 < ins.mean_soh < 1.0
        assert ins.p10_soh <= ins.mean_soh
        assert ins.worst_vehicle in {v.vehicle_id for v in _healthy_fleet().vehicles}

    def test_flags_a_low_outlier_as_at_risk(self):
        fleet = _healthy_fleet()
        fleet.add(bms.VehicleState("sick", soh=0.68, equivalent_full_cycles=400))
        ins = fleet.insight(soh_floor=0.80)
        assert "sick" in ins.at_risk                # below floor and a low outlier
        assert ins.worst_vehicle == "sick"

    def test_fastest_degrader_is_ranked_first(self):
        fleet = bms.FleetTwin()
        fleet.add(bms.VehicleState("slow", soh=0.95, equivalent_full_cycles=1000))
        fleet.add(bms.VehicleState("fast", soh=0.85, equivalent_full_cycles=200))
        ins = fleet.insight()
        assert ins.fastest_degraders[0][0] == "fast"   # highest fade per cycle

    def test_clusters_label_every_vehicle(self):
        ins = _healthy_fleet().insight()
        assert set(ins.clusters.values()) <= {"fast", "normal", "slow"}
        assert len(ins.clusters) == 20

    def test_empty_fleet_is_safe(self):
        ins = bms.FleetTwin().insight()
        assert ins.n == 0 and ins.at_risk == []

    def test_serialises(self):
        d = _healthy_fleet().insight().to_dict()
        for key in ("mean_soh", "at_risk", "fastest_degraders", "clusters"):
            assert key in d
