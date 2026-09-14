"""Minimal UDS (ISO 14229) diagnostic server for the BMS.

Real BMS ECUs answer diagnostic requests over UDS — a tester reads live data,
reads and clears Diagnostic Trouble Codes (DTCs), and manages sessions.  This is
a small, in-memory UDS server covering the services a battery ECU actually uses,
so the twin can be exercised by a UDS client and its faults surface as DTCs.

Supported services
------------------
* ``0x10`` DiagnosticSessionControl
* ``0x3E`` TesterPresent
* ``0x22`` ReadDataByIdentifier — live pack voltage / SoC / SoH / temperature
* ``0x19`` ReadDTCInformation (sub-function 0x02, by status mask)
* ``0x14`` ClearDiagnosticInformation

A request is ``bytes([service, ...])``; a positive response is
``bytes([service + 0x40, ...])``; a negative response is
``bytes([0x7F, service, nrc])``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# Negative Response Codes
NRC_SERVICE_NOT_SUPPORTED = 0x11
NRC_SUBFUNCTION_NOT_SUPPORTED = 0x12
NRC_REQUEST_OUT_OF_RANGE = 0x31

# Data identifiers (DIDs) for ReadDataByIdentifier (0x22).
DID_PACK_VOLTAGE = 0xF010      # u16, 0.01 V
DID_SOC = 0xF011              # u16, 0.1 %
DID_SOH = 0xF012             # u16, 0.1 %
DID_TEMP_MAX = 0xF013        # i16, 0.1 °C

# Map the twin's fault labels → 3-byte DTCs (ISO 14229 / SAE J2012 style).
FAULT_TO_DTC: dict[str, int] = {
    "overcharge": 0x900100,
    "short_circuit": 0x900200,
    "thermal_runaway": 0x900300,
    "sensor_dropout": 0x900400,
    "sensor_bias": 0x900500,
    "undervoltage": 0x900600,
    "gas_venting": 0x900700,
    "internal_short": 0x900800,
    "swelling": 0x900900,
    "electrolyte_leak": 0x900A00,
}
DTC_STATUS_CONFIRMED = 0x08    # "confirmedDTC" bit


@dataclass
class UDSServer:
    """In-memory UDS server exposing live BMS data and stored DTCs."""

    live: dict = field(default_factory=dict)   # {"v_pack","soc","soh","t_max"}
    dtcs: list = field(default_factory=list)    # list of fault-label strings
    session: int = 0x01                          # default session

    def set_live(self, *, v_pack=None, soc=None, soh=None, t_max=None) -> None:
        for k, val in (("v_pack", v_pack), ("soc", soc), ("soh", soh), ("t_max", t_max)):
            if val is not None:
                self.live[k] = float(val)

    def add_dtc(self, fault_label: str) -> None:
        if fault_label in FAULT_TO_DTC and fault_label not in self.dtcs:
            self.dtcs.append(fault_label)

    # ------------------------------------------------------------------
    def request(self, payload: bytes) -> bytes:
        """Handle one UDS request; returns the response bytes."""
        if not payload:
            return bytes([0x7F, 0x00, NRC_SERVICE_NOT_SUPPORTED])
        service = payload[0]
        handler = {
            0x10: self._session_control,
            0x3E: self._tester_present,
            0x22: self._read_data_by_id,
            0x19: self._read_dtc,
            0x14: self._clear_dtc,
        }.get(service)
        if handler is None:
            return bytes([0x7F, service, NRC_SERVICE_NOT_SUPPORTED])
        return handler(payload)

    # ---- services ----------------------------------------------------
    def _session_control(self, payload: bytes) -> bytes:
        if len(payload) < 2:
            return bytes([0x7F, 0x10, NRC_SUBFUNCTION_NOT_SUPPORTED])
        self.session = payload[1]
        return bytes([0x50, payload[1]])

    def _tester_present(self, payload: bytes) -> bytes:
        return bytes([0x7E, payload[1] if len(payload) > 1 else 0x00])

    def _read_data_by_id(self, payload: bytes) -> bytes:
        if len(payload) < 3:
            return bytes([0x7F, 0x22, NRC_REQUEST_OUT_OF_RANGE])
        did = struct.unpack(">H", payload[1:3])[0]
        if did == DID_PACK_VOLTAGE:
            val = int(round(self.live.get("v_pack", 0.0) / 0.01)) & 0xFFFF
            return bytes([0x62]) + payload[1:3] + struct.pack(">H", val)
        if did == DID_SOC:
            val = int(round(self.live.get("soc", 0.0) * 100 / 0.1)) & 0xFFFF
            return bytes([0x62]) + payload[1:3] + struct.pack(">H", val)
        if did == DID_SOH:
            val = int(round(self.live.get("soh", 0.0) * 100 / 0.1)) & 0xFFFF
            return bytes([0x62]) + payload[1:3] + struct.pack(">H", val)
        if did == DID_TEMP_MAX:
            val = int(round(self.live.get("t_max", 0.0) / 0.1))
            return bytes([0x62]) + payload[1:3] + struct.pack(">h", val)
        return bytes([0x7F, 0x22, NRC_REQUEST_OUT_OF_RANGE])

    def _read_dtc(self, payload: bytes) -> bytes:
        # 0x19 0x02 <status_mask>: report DTCs by status mask.
        if len(payload) < 2 or payload[1] != 0x02:
            return bytes([0x7F, 0x19, NRC_SUBFUNCTION_NOT_SUPPORTED])
        out = bytearray([0x59, 0x02, DTC_STATUS_CONFIRMED])
        for label in self.dtcs:
            out += struct.pack(">I", FAULT_TO_DTC[label])[1:]   # 3-byte DTC
            out.append(DTC_STATUS_CONFIRMED)
        return bytes(out)

    def _clear_dtc(self, payload: bytes) -> bytes:
        self.dtcs.clear()
        return bytes([0x54])
