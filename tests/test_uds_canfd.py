"""Tests: CAN FD frames + UDS diagnostic server."""

from __future__ import annotations

import struct

import pytest

import bms
from bms.can import CANFD_DLCS, CANFDFrame
from bms.uds import DID_PACK_VOLTAGE, DID_SOC, DID_SOH, DID_TEMP_MAX


class TestCANFD:
    def test_accepts_valid_dlcs_up_to_64(self):
        for dlc in (0, 8, 12, 32, 64):
            assert len(CANFDFrame(0x180, bytes(dlc)).data) == dlc

    def test_rejects_invalid_dlc(self):
        with pytest.raises(ValueError, match="not a valid"):
            CANFDFrame(0x180, bytes(9))
        with pytest.raises(ValueError):
            CANFDFrame(0x180, bytes(65))

    def test_pad_to_dlc_rounds_up(self):
        assert len(CANFDFrame.pad_to_dlc(bytes(9))) == 12
        assert len(CANFDFrame.pad_to_dlc(bytes(33))) == 48
        assert CANFDFrame.pad_to_dlc(b"\x01\x02") == b"\x01\x02"   # 2 is already valid
        assert 12 in CANFD_DLCS


class TestUDSServer:
    def _server(self):
        s = bms.UDSServer()
        s.set_live(v_pack=366.2, soc=0.55, soh=0.92, t_max=31.4)
        return s

    def test_read_data_by_identifier(self):
        s = self._server()
        for did, scale, expect in ((DID_SOC, 0.1, 55.0), (DID_SOH, 0.1, 92.0),
                                   (DID_PACK_VOLTAGE, 0.01, 366.2)):
            r = s.request(bytes([0x22]) + struct.pack(">H", did))
            assert r[0] == 0x62                                   # positive response
            assert struct.unpack(">H", r[3:5])[0] * scale == pytest.approx(expect, abs=0.1)

    def test_temperature_is_signed(self):
        s = bms.UDSServer()
        s.set_live(t_max=-12.5)
        r = s.request(bytes([0x22]) + struct.pack(">H", DID_TEMP_MAX))
        assert struct.unpack(">h", r[3:5])[0] * 0.1 == pytest.approx(-12.5, abs=0.1)

    def test_read_and_clear_dtcs(self):
        s = self._server()
        s.add_dtc("thermal_runaway")
        s.add_dtc("overcharge")
        s.add_dtc("overcharge")                                  # dedup
        r = s.request(bytes([0x19, 0x02, 0xFF]))
        assert r[0] == 0x59 and r[1] == 0x02
        assert (len(r) - 3) // 4 == 2                            # two 4-byte DTC records
        assert s.request(bytes([0x14, 0xFF, 0xFF, 0xFF]))[0] == 0x54
        assert s.dtcs == []                                      # cleared

    def test_unknown_did_and_service_are_negative_responses(self):
        s = self._server()
        bad_did = s.request(bytes([0x22, 0xDE, 0xAD]))
        assert bad_did[0] == 0x7F and bad_did[1] == 0x22
        bad_svc = s.request(bytes([0x99]))
        assert bad_svc == bytes([0x7F, 0x99, 0x11])              # serviceNotSupported

    def test_session_control_and_tester_present(self):
        s = self._server()
        assert s.request(bytes([0x10, 0x03])) == bytes([0x50, 0x03])
        assert s.session == 0x03
        assert s.request(bytes([0x3E, 0x00]))[0] == 0x7E

    def test_only_known_faults_become_dtcs(self):
        s = bms.UDSServer()
        s.add_dtc("not_a_real_fault")
        assert s.dtcs == []                                      # ignored
        assert "thermal_runaway" in bms.FAULT_TO_DTC
