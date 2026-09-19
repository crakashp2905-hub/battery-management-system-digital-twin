"""Degradation-mode *inference* — not just how much a cell aged, but **why**.

`degradation_modes.py` splits capacity fade into LLI vs LAM from an
incremental-capacity curve.  This layer turns that into a full diagnosis:

1. it fuses the **LLI / LAM** capacity-fade split with **resistance growth**
   (from `R0` now vs beginning-of-life, or EIS) into one normalised attribution —
   *"SoH 84 %: LLI 61 %, LAM 24 %, resistance 15 %"*; and
2. it reasons from the cell's **operating history** (mean SoC, temperature
   exposure, calendar time, cycle count, fast/cold-charge events) to the **likely
   physical mechanism** — calendar/SEI growth at high SoC, high-temperature
   cycling, lithium plating from cold/fast charging, or high-rate stress —
   scored deterministically and made consistent with the observed dominant mode.

The reasoning is rule-based and explainable by design (the repo keeps physics in
code, not in an LLM), so every number and verdict is traceable to its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .degradation_modes import diagnose_degradation_modes

# The candidate physical mechanisms and the modes each one tends to drive.
_MECHANISMS = {
    "calendar_sei_high_soc": ("LLI", "calendar/SEI growth at high state-of-charge"),
    "high_temperature_cycling": ("LAM", "high-temperature cycling"),
    "lithium_plating": ("LLI", "lithium plating from cold/fast charging"),
    "high_rate_stress": ("resistance", "high-rate cycling stress / particle cracking"),
}


def _sat(x: float, scale: float) -> float:
    """Saturating 0→1 ramp: ``x/scale`` clipped, smooth enough for scoring."""
    return float(np.clip(x / max(scale, 1e-9), 0.0, 1.0))


@dataclass
class OperatingHistory:
    """What the cell has been through — the evidence for a mechanism verdict."""

    mean_soc: float = 0.5
    mean_temperature_C: float = 25.0
    calendar_days: float = 0.0
    equivalent_full_cycles: float = 0.0
    mean_c_rate: float = 1.0
    fast_charge_events: int = 0
    cold_charge_events: int = 0


@dataclass(frozen=True)
class DegradationDiagnosis:
    """A full *why* for a cell's state of health."""

    soh_capacity: float
    capacity_loss_pct: float
    resistance_growth_pct: float
    mode_fractions: dict            # {'LLI','LAM','resistance'} — sum ≈ 1
    dominant_mode: str
    likely_mechanisms: list         # [(name, score, description)], ranked
    narrative: str

    def to_dict(self) -> dict:
        return {
            "soh_capacity": self.soh_capacity,
            "capacity_loss_pct": self.capacity_loss_pct,
            "resistance_growth_pct": self.resistance_growth_pct,
            "mode_fractions": self.mode_fractions,
            "dominant_mode": self.dominant_mode,
            "likely_mechanisms": self.likely_mechanisms,
            "narrative": self.narrative,
        }


def infer_mechanisms(history: OperatingHistory, dominant_mode: str
                     ) -> list[tuple[str, float, str]]:
    """Score each candidate mechanism from the operating history.

    Scores are deterministic saturating functions of the stressors, then boosted
    when the mechanism's characteristic mode matches the observed dominant mode.
    Returns ``(name, score, description)`` ranked most-likely first.
    """
    h = history
    hot = _sat(h.mean_temperature_C - 25.0, 25.0)      # 25→50 °C ramps 0→1
    cold_fast = _sat(h.cold_charge_events + h.fast_charge_events, 30.0)
    high_soc = _sat(h.mean_soc - 0.5, 0.4)             # 0.5→0.9 ramps 0→1
    calendar = _sat(h.calendar_days, 720.0)            # up to ~2 years
    cycling = _sat(h.equivalent_full_cycles, 1500.0)
    high_rate = _sat(h.mean_c_rate - 1.0, 2.0)         # 1C→3C ramps 0→1

    raw = {
        "calendar_sei_high_soc": 0.6 * calendar + 0.4 * high_soc,
        "high_temperature_cycling": 0.6 * hot + 0.4 * cycling,
        "lithium_plating": cold_fast,
        "high_rate_stress": 0.6 * high_rate + 0.4 * cycling,
    }
    scored = []
    for name, score in raw.items():
        mode, desc = _MECHANISMS[name]
        if mode == dominant_mode:                       # consistency boost
            score = min(1.0, score * 1.25 + 0.05)
        scored.append((name, float(round(score, 3)), desc))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def infer_degradation_modes(*, soh_capacity: float | None = None,
                            r0_bol: float, r0_now: float,
                            v_axis: np.ndarray | None = None,
                            ic_fresh: np.ndarray | None = None,
                            ic_aged: np.ndarray | None = None,
                            history: OperatingHistory | None = None
                            ) -> DegradationDiagnosis:
    """Diagnose a cell's degradation into modes and a likely mechanism.

    Provide an incremental-capacity pair (``v_axis``, ``ic_fresh``, ``ic_aged``)
    for the LLI/LAM capacity split; otherwise the capacity loss is reported as a
    single unresolved mode.  ``r0_bol``/``r0_now`` give the resistance-growth
    mode.  ``history`` (optional) drives the mechanism verdict.
    """
    r_growth = max(0.0, r0_now / max(r0_bol, 1e-12) - 1.0)

    if ic_fresh is not None and ic_aged is not None and v_axis is not None:
        d = diagnose_degradation_modes(v_axis, ic_fresh, ic_aged)
        lli, lam = d["lli"], d["lam"]
        cap_loss = d["total_capacity_loss"]
        soh = soh_capacity if soh_capacity is not None else 1.0 - cap_loss
        denom = lli + lam
        cap_lli = (lli / denom) * cap_loss if denom > 0 else 0.0
        cap_lam = (lam / denom) * cap_loss if denom > 0 else 0.0
        raw = {"LLI": cap_lli, "LAM": cap_lam, "resistance": r_growth}
    else:
        cap_loss = (1.0 - soh_capacity) if soh_capacity is not None else 0.0
        soh = soh_capacity if soh_capacity is not None else 1.0
        raw = {"capacity_loss": max(0.0, cap_loss), "resistance": r_growth}

    total = sum(raw.values())
    fractions = ({k: float(round(v / total, 4)) for k, v in raw.items()}
                 if total > 0 else {k: 0.0 for k in raw})
    dominant = max(fractions, key=fractions.get) if total > 0 else "none"
    # Map the reported dominant key onto a mechanism-mode label.
    dom_mode = {"LLI": "LLI", "LAM": "LAM", "resistance": "resistance",
                "capacity_loss": "LLI"}.get(dominant, "LLI")

    mechanisms = infer_mechanisms(history, dom_mode) if history is not None else []
    if mechanisms:
        top_name, top_score, top_desc = mechanisms[0]
        narrative = (f"SoH {100 * soh:.1f}% — dominant mode {dominant}; "
                     f"likely mechanism: {top_desc} (score {top_score:.2f}).")
    else:
        narrative = (f"SoH {100 * soh:.1f}% — dominant mode {dominant}; "
                     f"no operating history supplied for mechanism inference.")

    return DegradationDiagnosis(
        soh_capacity=float(soh),
        capacity_loss_pct=float(100.0 * cap_loss),
        resistance_growth_pct=float(100.0 * r_growth),
        mode_fractions=fractions, dominant_mode=dominant,
        likely_mechanisms=mechanisms, narrative=narrative,
    )


@dataclass
class DegradationInference:
    """Convenience wrapper binding the beginning-of-life reference resistance."""

    r0_bol: float
    history: OperatingHistory | None = field(default=None)

    def diagnose(self, *, r0_now: float, soh_capacity: float | None = None,
                 v_axis: np.ndarray | None = None,
                 ic_fresh: np.ndarray | None = None,
                 ic_aged: np.ndarray | None = None,
                 history: OperatingHistory | None = None) -> DegradationDiagnosis:
        return infer_degradation_modes(
            soh_capacity=soh_capacity, r0_bol=self.r0_bol, r0_now=r0_now,
            v_axis=v_axis, ic_fresh=ic_fresh, ic_aged=ic_aged,
            history=history if history is not None else self.history)
