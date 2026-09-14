"""Classic CAN 2.0B telemetry frames for the BMS digital twin.

The broadcaster uses standard 11-bit arbitration IDs and fixed 8-byte
payloads.  Signals are little-endian (Intel) and scaled exactly as a small
DBC would describe them; :meth:`BMSCanBus.parse` returns engineering units.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .sop import SOPLimit


@dataclass(frozen=True)
class CANFrame:
    """A validated classic-CAN data frame."""

    arbitration_id: int
    data: bytes
    is_extended_id: bool = False

    def __post_init__(self) -> None:
        max_id = 0x1FFFFFFF if self.is_extended_id else 0x7FF
        if not 0 <= self.arbitration_id <= max_id:
            raise ValueError("arbitration_id is outside the CAN identifier range")
        if len(self.data) != 8:
            raise ValueError("BMS telemetry frames must contain exactly 8 bytes")


# Valid CAN FD payload lengths (DLC): 0–8, then 12/16/20/24/32/48/64 bytes.
CANFD_DLCS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64)


@dataclass(frozen=True)
class CANFDFrame:
    """A CAN FD data frame — up to 64-byte payloads and an optional bit-rate
    switch, for the richer diagnostics/telemetry classic CAN's 8 bytes can't hold.
    """

    arbitration_id: int
    data: bytes
    is_extended_id: bool = False
    bitrate_switch: bool = True

    def __post_init__(self) -> None:
        max_id = 0x1FFFFFFF if self.is_extended_id else 0x7FF
        if not 0 <= self.arbitration_id <= max_id:
            raise ValueError("arbitration_id is outside the CAN identifier range")
        if len(self.data) not in CANFD_DLCS:
            raise ValueError(f"CAN FD payload length {len(self.data)} is not a valid "
                             f"DLC ({', '.join(map(str, CANFD_DLCS))})")

    @staticmethod
    def pad_to_dlc(data: bytes) -> bytes:
        """Zero-pad ``data`` up to the next valid CAN FD DLC (≤ 64 bytes)."""
        n = len(data)
        target = next((d for d in CANFD_DLCS if d >= n), None)
        if target is None:
            raise ValueError("payload exceeds the 64-byte CAN FD maximum")
        return bytes(data) + b"\x00" * (target - n)


def can_checksum(data: bytes) -> int:
    """8-bit additive checksum over a frame payload (0–255)."""
    return sum(bytes(data)) & 0xFF


class BMSCanBus:
    """Encode and decode a compact, DBC-compatible BMS broadcast set."""

    STATUS_ID = 0x180
    CELL_EXTREMA_ID = 0x181
    THERMAL_ID = 0x182
    SOP_2S_ID = 0x183
    SOP_10S_ID = 0x184
    SOP_30S_ID = 0x185
    HEALTH_ID = 0x186
    SOP_IDS = {2.0: SOP_2S_ID, 10.0: SOP_10S_ID, 30.0: SOP_30S_ID}

    STATE_CODES = {"idle": 0, "precharge": 1, "operating": 2,
                   "balancing": 3, "fault": 4, "shutdown": 5}
    FAULT_CODES = {"none": 0, "overcharge": 1, "short_circuit": 2,
                   "thermal_runaway": 3, "sensor_dropout": 4,
                   "sensor_bias": 5, "undervoltage": 6, "gas_venting": 7,
                   "internal_short": 8, "swelling": 9, "electrolyte_leak": 10}

    def __init__(self) -> None:
        self._alive = 0        # rolling 0–255 alive counter (transmitter health)
        self._tx_count = 0     # total frames transmitted (rolling u16)

    @staticmethod
    def _u16(value: float, scale: float) -> int:
        return int(np.clip(round(value / scale), 0, 0xFFFF))

    @staticmethod
    def _i16(value: float, scale: float) -> int:
        return int(np.clip(round(value / scale), -0x8000, 0x7FFF))

    def status_frame(self, result: dict) -> CANFrame:
        """Encode pack voltage/current, mean SOC, state, and fault."""
        state = self.STATE_CODES.get(str(result.get("state", "idle")), 0xFF)
        fault = self.FAULT_CODES.get(str(result.get("fault_label", "none")), 0xFF)
        payload = struct.pack(
            "<HhHBB",
            self._u16(float(result["v_pack"]), 0.01),
            self._i16(float(result.get("cmd_current", 0.0)), 0.1),
            self._u16(float(np.mean(result["soc"])) * 100.0, 0.1),
            state,
            fault,
        )
        return CANFrame(self.STATUS_ID, payload)

    def cell_extrema_frame(self, cell_voltages_V: np.ndarray,
                           soc: np.ndarray) -> CANFrame:
        """Encode cell/group voltage and SOC extrema plus group count."""
        v = np.asarray(cell_voltages_V, dtype=float)
        s = np.asarray(soc, dtype=float)
        payload = struct.pack(
            "<HHBBH",
            self._u16(float(v.min()), 0.001), self._u16(float(v.max()), 0.001),
            int(np.clip(round(float(s.min()) * 200.0), 0, 200)),
            int(np.clip(round(float(s.max()) * 200.0), 0, 200)),
            min(len(v), 0xFF),
        )
        return CANFrame(self.CELL_EXTREMA_ID, payload)

    def thermal_frame(self, temperatures_C: np.ndarray, cooling_duty: float) -> CANFrame:
        """Encode min/max temperature, spread, and cooling duty."""
        t = np.asarray(temperatures_C, dtype=float)
        payload = struct.pack(
            "<hhhBB",
            self._i16(float(t.min()), 0.1), self._i16(float(t.max()), 0.1),
            self._i16(float(t.max() - t.min()), 0.1),
            int(np.clip(round(cooling_duty * 100.0), 0, 100)), 0,
        )
        return CANFrame(self.THERMAL_ID, payload)

    def sop_frame(self, limit: SOPLimit) -> CANFrame:
        """Encode one SOP horizon; standard horizons use dedicated IDs."""
        identifier = self.SOP_IDS.get(float(limit.horizon_s))
        if identifier is None:
            raise ValueError("CAN SOP broadcast supports 2 s, 10 s, and 30 s horizons")
        payload = struct.pack(
            "<HHHH",
            self._u16(limit.traction_power_W, 1.0), self._u16(limit.regen_power_W, 1.0),
            self._u16(limit.traction_current_A, 0.1),
            self._u16(limit.regen_current_A, 0.1),
        )
        return CANFrame(identifier, payload)

    def health_frame(self, status_data: bytes, bus_off: bool = False) -> CANFrame:
        """Bus-health frame: alive counter, status checksum, tx count, bus-off flag."""
        payload = struct.pack("<BBHBBBB", self._alive & 0xFF,
                              can_checksum(status_data), self._tx_count & 0xFFFF,
                              int(bool(bus_off)), 0, 0, 0)
        return CANFrame(self.HEALTH_ID, payload)

    def broadcast(self, result: dict, sop: dict[float, SOPLimit]) -> list[CANFrame]:
        """Build the status/cell/thermal/SOP cycle plus a bus-health frame.

        Each call advances the rolling alive counter, so a receiver can detect a
        stalled transmitter; the health frame carries a checksum over the status
        frame for integrity checking.
        """
        status = self.status_frame(result)
        frames = [status,
                  self.cell_extrema_frame(result["v_cells"], result["soc"]),
                  self.thermal_frame(result["T_cells"], result["cooling_duty"])]
        frames.extend(self.sop_frame(sop[h]) for h in sorted(sop) if h in self.SOP_IDS)
        self._alive = (self._alive + 1) & 0xFF
        self._tx_count = (self._tx_count + len(frames) + 1) & 0xFFFF
        frames.append(self.health_frame(status.data))
        return frames

    def parse(self, frame: CANFrame) -> dict:
        """Decode a frame into DBC-style signal names and engineering units."""
        data = frame.data
        if frame.arbitration_id == self.STATUS_ID:
            v, current, soc, state, fault = struct.unpack("<HhHBB", data)
            return {"message": "BMS_Status", "pack_voltage_V": v * 0.01,
                    "pack_current_A": current * 0.1, "soc_pct": soc * 0.1,
                    "state_code": state, "fault_code": fault}
        if frame.arbitration_id == self.CELL_EXTREMA_ID:
            vmin, vmax, smin, smax, count = struct.unpack("<HHBBH", data)
            return {"message": "BMS_CellExtrema", "cell_voltage_min_V": vmin * 0.001,
                    "cell_voltage_max_V": vmax * 0.001, "soc_min_pct": smin * 0.5,
                    "soc_max_pct": smax * 0.5, "series_groups": count}
        if frame.arbitration_id == self.THERMAL_ID:
            tmin, tmax, spread, duty, _ = struct.unpack("<hhhBB", data)
            return {"message": "BMS_Thermal", "temperature_min_C": tmin * 0.1,
                    "temperature_max_C": tmax * 0.1, "temperature_spread_C": spread * 0.1,
                    "cooling_duty_pct": duty}
        if frame.arbitration_id in self.SOP_IDS.values():
            traction, regen, current, regen_i = struct.unpack("<HHHH", data)
            horizon = next(h for h, identifier in self.SOP_IDS.items()
                           if identifier == frame.arbitration_id)
            return {"message": f"BMS_SOP_{int(horizon)}s", "horizon_s": horizon,
                    "traction_power_W": float(traction), "regen_power_W": float(regen),
                    "traction_current_A": current * 0.1, "regen_current_A": regen_i * 0.1}
        if frame.arbitration_id == self.HEALTH_ID:
            alive, checksum, tx, bus_off, _r1, _r2, _r3 = struct.unpack("<BBHBBBB", data)
            return {"message": "BMS_Health", "alive_counter": alive,
                    "status_checksum": checksum, "tx_count": tx, "bus_off": bool(bus_off)}
        raise ValueError(f"unsupported BMS CAN identifier: 0x{frame.arbitration_id:X}")

    def parse_all(self, frames: Iterable[CANFrame]) -> list[dict]:
        return [self.parse(frame) for frame in frames]

    def to_dbc(self) -> str:
        """Emit a Vector ``.dbc`` describing the BMS broadcast message/signal set."""
        lines = ['VERSION ""', "", "BS_:", "", "BU_: BMS", ""]

        def message(mid, name, signals):
            return [f"BO_ {mid} {name}: 8 BMS"] + [f" SG_ {s}" for s in signals] + [""]

        lines += message(self.STATUS_ID, "BMS_Status", [
            'pack_voltage_V : 0|16@1+ (0.01,0) [0|655.35] "V" Vector__XXX',
            'pack_current_A : 16|16@1- (0.1,0) [-3276.8|3276.7] "A" Vector__XXX',
            'soc_pct : 32|16@1+ (0.1,0) [0|100] "%" Vector__XXX',
            'state_code : 48|8@1+ (1,0) [0|255] "" Vector__XXX',
            'fault_code : 56|8@1+ (1,0) [0|255] "" Vector__XXX',
        ])
        lines += message(self.CELL_EXTREMA_ID, "BMS_CellExtrema", [
            'cell_voltage_min_V : 0|16@1+ (0.001,0) [0|65.535] "V" Vector__XXX',
            'cell_voltage_max_V : 16|16@1+ (0.001,0) [0|65.535] "V" Vector__XXX',
            'soc_min_pct : 32|8@1+ (0.5,0) [0|100] "%" Vector__XXX',
            'soc_max_pct : 40|8@1+ (0.5,0) [0|100] "%" Vector__XXX',
            'series_groups : 48|16@1+ (1,0) [0|65535] "" Vector__XXX',
        ])
        lines += message(self.THERMAL_ID, "BMS_Thermal", [
            'temperature_min_C : 0|16@1- (0.1,0) [-3276.8|3276.7] "degC" Vector__XXX',
            'temperature_max_C : 16|16@1- (0.1,0) [-3276.8|3276.7] "degC" Vector__XXX',
            'temperature_spread_C : 32|16@1- (0.1,0) [0|3276.7] "degC" Vector__XXX',
            'cooling_duty_pct : 48|8@1+ (1,0) [0|100] "%" Vector__XXX',
        ])
        for horizon, mid in sorted(self.SOP_IDS.items()):
            lines += message(mid, f"BMS_SOP_{int(horizon)}s", [
                'traction_power_W : 0|16@1+ (1,0) [0|65535] "W" Vector__XXX',
                'regen_power_W : 16|16@1+ (1,0) [0|65535] "W" Vector__XXX',
                'traction_current_A : 32|16@1+ (0.1,0) [0|6553.5] "A" Vector__XXX',
                'regen_current_A : 48|16@1+ (0.1,0) [0|6553.5] "A" Vector__XXX',
            ])
        lines += message(self.HEALTH_ID, "BMS_Health", [
            'alive_counter : 0|8@1+ (1,0) [0|255] "" Vector__XXX',
            'status_checksum : 8|8@1+ (1,0) [0|255] "" Vector__XXX',
            'tx_count : 16|16@1+ (1,0) [0|65535] "" Vector__XXX',
            'bus_off : 32|8@1+ (1,0) [0|1] "" Vector__XXX',
        ])
        return "\n".join(lines) + "\n"


class CanBusMonitor:
    """Receiver-side CAN health: message freshness/timeout, alive-counter
    continuity, checksum validation, and bus-off (ISO 11898 error counting).
    """

    def __init__(self, timeout_s: float = 0.5, bus_off_threshold: int = 256):
        self.timeout_s = float(timeout_s)
        self.bus_off_threshold = int(bus_off_threshold)
        self._last_seen: dict[int, float] = {}
        self._last_alive: int | None = None
        self.error_count = 0                    # transmit/receive error counter

    def receive(self, frame: CANFrame, t: float) -> None:
        self._last_seen[frame.arbitration_id] = float(t)

    def is_stale(self, arbitration_id: int, now: float) -> bool:
        last = self._last_seen.get(arbitration_id)
        return last is None or (now - last) > self.timeout_s

    def alive_ok(self, alive: int) -> bool:
        ok = self._last_alive is None or (alive & 0xFF) == ((self._last_alive + 1) & 0xFF)
        self._last_alive = alive & 0xFF
        self._register(ok)
        return ok

    def checksum_ok(self, data: bytes, expected: int) -> bool:
        ok = can_checksum(data) == (expected & 0xFF)
        self._register(ok)
        return ok

    def _register(self, ok: bool) -> None:
        # CAN error counter: +8 on error, -1 on success (ISO 11898).
        self.error_count = (max(0, self.error_count - 1) if ok
                            else min(512, self.error_count + 8))

    @property
    def bus_off(self) -> bool:
        return self.error_count >= self.bus_off_threshold

    def recover(self) -> None:
        """Reset after a bus-off recovery sequence."""
        self.error_count = 0
        self._last_alive = None
