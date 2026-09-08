"""Tests: can."""

from __future__ import annotations

import pytest

import bms
from bms.can import BMSCanBus, CanBusMonitor, CANFrame, can_checksum


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
        # status + cell + thermal + 3×SOP + health = 7
        assert len(frames) == 7
        assert all(len(frame.data) == 8 and not frame.is_extended_id for frame in frames)
        parsed = bus.parse_all(frames)
        status = parsed[0]
        assert status["message"] == "BMS_Status"
        assert status["pack_voltage_V"] == pytest.approx(result["v_pack"], abs=0.01)
        assert status["pack_current_A"] == pytest.approx(result["cmd_current"], abs=0.1)
        assert {item["horizon_s"] for item in parsed if "horizon_s" in item} == {
            2.0, 10.0, 30.0,
        }
        assert parsed[-1]["message"] == "BMS_Health"

    def test_can_frame_rejects_non_classic_payload_length(self):
        with pytest.raises(ValueError, match="exactly 8 bytes"):
            bms.CANFrame(0x180, b"\x00" * 7)


class TestCanHealth:
    def test_health_frame_carries_checksum_of_status_and_round_trips(self):
        bus = BMSCanBus()
        status = bus.status_frame(
            {"v_pack": 360.0, "cmd_current": 5.0, "soc": [0.5],
             "state": "operating", "fault_label": "none"})
        health = bus.health_frame(status.data)
        decoded = bus.parse(health)
        assert decoded["message"] == "BMS_Health"
        assert decoded["status_checksum"] == can_checksum(status.data)
        assert decoded["bus_off"] is False

    def test_alive_counter_and_tx_count_advance_each_broadcast(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=3, seed=1))
        thermal = bms.ThermalModel(n_cells=3)
        sup = bms.BMSSupervisor(pack, thermal, bms.HybridFaultDetector())
        bus = BMSCanBus()
        alives, txs = [], []
        for _ in range(3):
            result = sup.step(1.0, 1.0)
            sop = bms.StateOfPower().calculate(
                pack, result["T_cells"], result["cmd_current"], result["v_cells"])
            frames = bus.broadcast(result, sop)
            health = bus.parse(frames[-1])
            alives.append(health["alive_counter"])
            txs.append(health["tx_count"])
        assert alives == [1, 2, 3]          # rolls forward, receiver detects a stall
        assert txs[0] < txs[1] < txs[2]     # monotone frame count

    def test_checksum_is_byte_additive(self):
        assert can_checksum(b"\x01\x02\x03") == 6
        assert can_checksum(bytes([200, 100])) == (300 & 0xFF)


class TestDbcExport:
    def test_to_dbc_has_all_messages_and_signals(self):
        dbc = BMSCanBus().to_dbc()
        for name in ("BMS_Status", "BMS_CellExtrema", "BMS_Thermal",
                     "BMS_SOP_2s", "BMS_SOP_10s", "BMS_SOP_30s", "BMS_Health"):
            assert f" {name}:" in dbc
        assert dbc.count("BO_ ") == 7                     # 7 messages
        assert 'SG_ alive_counter' in dbc
        assert dbc.startswith('VERSION ""')

    def test_committed_dbc_matches_encoder(self):
        """The shipped bms.dbc must not drift from the encoder that emits it."""
        from pathlib import Path
        repo_dbc = Path(__file__).resolve().parents[1] / "bms.dbc"
        assert repo_dbc.read_text().replace("\r\n", "\n") == BMSCanBus().to_dbc()

    def test_dbc_round_trips_through_cantools_if_available(self):
        cantools = pytest.importorskip("cantools")
        from pathlib import Path
        repo_dbc = Path(__file__).resolve().parents[1] / "bms.dbc"
        db = cantools.database.load_file(str(repo_dbc))
        assert len(db.messages) == 7
        bus = BMSCanBus()
        frame = bus.status_frame(
            {"v_pack": 366.2, "cmd_current": -12.5, "soc": [0.55, 0.56],
             "state": "operating", "fault_label": "none"})
        decoded = db.decode_message(frame.arbitration_id, frame.data)
        mine = bus.parse(frame)
        assert float(decoded["pack_voltage_V"]) == pytest.approx(mine["pack_voltage_V"])
        assert float(decoded["pack_current_A"]) == pytest.approx(mine["pack_current_A"])


class TestCanBusMonitor:
    def test_freshness_timeout(self):
        mon = CanBusMonitor(timeout_s=0.5)
        assert mon.is_stale(BMSCanBus.STATUS_ID, now=0.0)     # never seen → stale
        mon.receive(CANFrame(BMSCanBus.STATUS_ID, b"\x00" * 8), t=1.0)
        assert not mon.is_stale(BMSCanBus.STATUS_ID, now=1.4)
        assert mon.is_stale(BMSCanBus.STATUS_ID, now=1.6)     # older than timeout

    def test_alive_continuity_detects_skips(self):
        mon = CanBusMonitor()
        assert mon.alive_ok(5)        # first sample always accepted
        assert mon.alive_ok(6)        # +1 → good
        assert not mon.alive_ok(9)    # skipped 7,8 → flagged
        assert mon.alive_ok(10)       # resynced (9 → 10) → good again

    def test_alive_counter_wraps_at_255(self):
        mon = CanBusMonitor()
        assert mon.alive_ok(254)      # first sample accepted
        assert mon.alive_ok(255)      # +1 → good
        assert mon.alive_ok(0)        # 255 → 0 wraps cleanly
        assert mon.alive_ok(1)        # continues past the wrap

    def test_checksum_validation(self):
        mon = CanBusMonitor()
        data = b"\x10\x20\x30\x00\x00\x00\x00\x00"
        assert mon.checksum_ok(data, can_checksum(data))
        assert not mon.checksum_ok(data, can_checksum(data) ^ 0xFF)

    def test_bus_off_on_sustained_errors_and_recovery(self):
        mon = CanBusMonitor(bus_off_threshold=256)
        # 32 consecutive errors × +8 = 256 → bus-off.
        for _ in range(32):
            mon.checksum_ok(b"\x01" * 8, 0x00)   # wrong checksum
        assert mon.bus_off
        assert mon.error_count >= 256
        mon.recover()
        assert not mon.bus_off and mon.error_count == 0

    def test_success_decrements_error_counter(self):
        mon = CanBusMonitor()
        mon.checksum_ok(b"\x01" * 8, 0x00)       # error → +8
        assert mon.error_count == 8
        for _ in range(8):
            mon.checksum_ok(b"\x00" * 8, 0x00)   # success → -1 each
        assert mon.error_count == 0              # cannot go negative
