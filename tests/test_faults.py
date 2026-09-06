"""Tests: faults."""

from __future__ import annotations

import numpy as np

import bms


class TestFMEAandRUL:
    def test_rpn_descending(self):
        fmea = bms.build_fmea_table()
        assert (fmea["RPN"].values[:-1] >= fmea["RPN"].values[1:]).all()

    def test_rul_non_negative(self):
        df = bms.load_nasa_like_dataset(cycles=40, capacity_Ah=2.3, seed=1)
        rul = bms.estimate_rul(df["cycle"].values, df["capacity_Ah"].values,
                                nominal_capacity_Ah=2.3)
        assert rul["rul_cycles"] >= 0
        assert 0 <= rul["soh"] <= 1.0

    def test_synthetic_dataset_shape(self):
        df = bms.load_nasa_like_dataset(cycles=10, seed=0)
        assert {"cycle", "capacity_Ah", "temperature_C", "source"}.issubset(df.columns)
        assert len(df) == 10



class TestFaults:
    def test_rule_detects_overcharge(self):
        det = bms.HybridFaultDetector()
        v = np.array([3.7, 4.30, 3.7, 3.7])  # cell 1 over-charged
        T = np.array([25, 25, 25, 25.])
        feats = bms.extract_features(v, np.zeros(4), T, np.zeros(4), np.zeros(4))
        label, src = det.predict_step(feats, v, T)
        assert label == "overcharge"
        assert src == "rule"

    def test_rule_detects_runaway(self):
        det = bms.HybridFaultDetector()
        v = np.array([3.7, 3.7, 3.7, 3.7])
        T = np.array([25, 75., 25, 25])
        feats = bms.extract_features(v, np.zeros(4), T, np.zeros(4), np.zeros(4))
        label, src = det.predict_step(feats, v, T)
        assert label == "thermal_runaway"

    def test_injector_no_active_fault_passthrough(self):
        inj = bms.FaultInjector()
        c = np.array([1.0, 1.0, 1.0, 1.0])
        assert np.allclose(inj.apply_to_currents(c, 0), c)

    def test_supervisor_shutdown_on_thermal_runaway(self):
        det = bms.HybridFaultDetector()                    # not fitted → rule-only
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=3))
        thermal = bms.ThermalModel(n_cells=4)
        sup = bms.BMSSupervisor(pack, thermal, det)

        # Sustained over-temperature above the runaway threshold is a critical
        # fault → latch to terminal SHUTDOWN (not the recoverable FAULT).
        for k in range(10):
            thermal.T[1] = 80.0
            sup.step(0.5, 1.0, k=k)
        assert sup.state == bms.BMSState.SHUTDOWN
        assert any(e["source"] == "rule" for e in sup.fault_log)
        # SHUTDOWN is terminal: clear_fault() refuses and leaves it latched.
        assert sup.clear_fault() is False
        assert sup.state == bms.BMSState.SHUTDOWN

    def test_supervisor_faults_and_recovers_on_noncritical_rule(self):
        # A sensor dropout — a non-critical rule alarm injected through the
        # supervisor's FaultInjector — should trip to the recoverable FAULT
        # state, and clear_fault() should return the pack to IDLE.
        det = bms.HybridFaultDetector()
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=3))
        thermal = bms.ThermalModel(n_cells=4)
        inj = bms.FaultInjector([bms.FaultSpec(bms.FaultMode.SENSOR_DROPOUT,
                                               start_step=0, cell_index=0)])
        sup = bms.BMSSupervisor(pack, thermal, det, injector=inj)

        for k in range(5):
            sup.step(0.5, 1.0, k=k)
        assert sup.state == bms.BMSState.FAULT
        assert any(e["mode"] == "sensor_dropout" and e["source"] == "rule"
                   for e in sup.fault_log)
        # Operator reset recovers from FAULT.
        assert sup.clear_fault() is True
        assert sup.state == bms.BMSState.IDLE

