"""High-voltage safety monitors: insulation resistance and contactor welding.

Two real BMS functions that live outside the cell models:

* :class:`InsulationMonitor` — estimates the isolation resistance between the HV
  bus and chassis and alarms when it falls below the regulatory **Ω/V** limit (a
  ground-fault / insulation-breakdown detector).
* :class:`ContactorWeldDetector` — after the main contactor is commanded open,
  the DC-link should bleed down; if it stays near pack voltage the contactor is
  **welded closed**, an interlock hazard.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InsulationMonitor:
    """Isolation-resistance monitor (IMD).

    ``resistance`` is estimated from the pack voltage and the measured leakage
    current to chassis (``R_iso = V_pack / I_leak``).  The alarm follows the
    standard **Ω/V** criterion: a fixed absolute resistance is not enough on its
    own, because the hazard scales with pack voltage.
    """

    warn_ohm_per_volt: float = 500.0
    fault_ohm_per_volt: float = 100.0
    resistance_ohm: float = field(default=float("inf"))

    def update(self, pack_voltage_V: float, leakage_current_A: float) -> float:
        """Estimate R_iso [Ω] from pack voltage and leakage current to chassis."""
        vp = abs(float(pack_voltage_V))
        il = abs(float(leakage_current_A))
        self.resistance_ohm = vp / il if il > 1e-12 else float("inf")
        return self.resistance_ohm

    def ohm_per_volt(self, pack_voltage_V: float) -> float:
        vp = abs(float(pack_voltage_V))
        return self.resistance_ohm / vp if vp > 1e-9 else float("inf")

    def status(self, pack_voltage_V: float) -> str:
        """``"ok"`` / ``"warning"`` / ``"fault"`` per the Ω/V thresholds."""
        opv = self.ohm_per_volt(pack_voltage_V)
        if opv < self.fault_ohm_per_volt:
            return "fault"
        if opv < self.warn_ohm_per_volt:
            return "warning"
        return "ok"


@dataclass
class ContactorWeldDetector:
    """Detect a main contactor welded closed after an open command.

    Feed the measured DC-link voltage while the contactor is commanded open.
    If, after ``settle_s`` of open time, the link voltage is still above
    ``bleed_ratio`` of pack voltage, the contactor has not opened → welded.
    """

    settle_s: float = 2.0
    bleed_ratio: float = 0.5
    _open_elapsed_s: float = 0.0
    welded: bool = False

    def reset(self) -> None:
        self._open_elapsed_s = 0.0
        self.welded = False

    def update(self, commanded_open: bool, dc_link_voltage_V: float,
               pack_voltage_V: float, dt: float) -> bool:
        """Advance the check; returns ``True`` when a weld is detected."""
        if not commanded_open:
            self._open_elapsed_s = 0.0
            return self.welded
        self._open_elapsed_s += max(0.0, float(dt))
        if self._open_elapsed_s >= self.settle_s:
            vp = abs(float(pack_voltage_V))
            if vp > 1e-6 and abs(float(dc_link_voltage_V)) > self.bleed_ratio * vp:
                self.welded = True
        return self.welded
