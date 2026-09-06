"""Tests: can."""

from __future__ import annotations

import pytest

import bms


class TestCanTelemetry:
    def test_can_broadcast_is_classic_8_byte_and_round_trips(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, seed=2))
        thermal = bms.ThermalModel(n_cells=4)
        sup = bms.BMSSupervisor(pack, thermal, bms.HybridFaultDetector())
        result = sup.step(2.0, 1.0)
        sop = bms.StateOfPower().calculate(
            pack, result["T_cells"], result["cmd_current"], result["v_cells"],
        )
        bus = bms.BMSCanBus()
        frames = bus.broadcast(result, sop)
        assert len(frames) == 6
        assert all(len(frame.data) == 8 and not frame.is_extended_id for frame in frames)
        parsed = bus.parse_all(frames)
        status = parsed[0]
        assert status["message"] == "BMS_Status"
        assert status["pack_voltage_V"] == pytest.approx(result["v_pack"], abs=0.01)
        assert status["pack_current_A"] == pytest.approx(result["cmd_current"], abs=0.1)
        assert {item["horizon_s"] for item in parsed if "horizon_s" in item} == {
            2.0, 10.0, 30.0,
        }

    def test_can_frame_rejects_non_classic_payload_length(self):
        with pytest.raises(ValueError, match="exactly 8 bytes"):
            bms.CANFrame(0x180, b"\x00" * 7)

