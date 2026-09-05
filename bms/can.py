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


class BMSCanBus:
    """Encode and decode a compact, DBC-compatible BMS broadcast set."""

    STATUS_ID = 0x180
    CELL_EXTREMA_ID = 0x181
    THERMAL_ID = 0x182
    SOP_2S_ID = 0x183
    SOP_10S_ID = 0x184
    SOP_30S_ID = 0x185
    SOP_IDS = {2.0: SOP_2S_ID, 10.0: SOP_10S_ID, 30.0: SOP_30S_ID}

    STATE_CODES = {"idle": 0, "precharge": 1, "operating": 2,
                   "balancing": 3, "fault": 4, "shutdown": 5}
    FAULT_CODES = {"none": 0, "overcharge": 1, "short_circuit": 2,
                   "thermal_runaway": 3, "sensor_dropout": 4,
                   "sensor_bias": 5, "undervoltage": 6}

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

    def broadcast(self, result: dict, sop: dict[float, SOPLimit]) -> list[CANFrame]:
        """Build the complete status/cell/thermal/SOP broadcast cycle."""
        frames = [self.status_frame(result),
                  self.cell_extrema_frame(result["v_cells"], result["soc"]),
                  self.thermal_frame(result["T_cells"], result["cooling_duty"])]
        frames.extend(self.sop_frame(sop[h]) for h in sorted(sop) if h in self.SOP_IDS)
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
        raise ValueError(f"unsupported BMS CAN identifier: 0x{frame.arbitration_id:X}")

    def parse_all(self, frames: Iterable[CANFrame]) -> list[dict]:
        return [self.parse(frame) for frame in frames]
