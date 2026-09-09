"""Fault Tree Analysis (FTA) for the top hazard: thermal runaway.

Where the FMEA (:mod:`bms.fmea`) scores failure modes one at a time, a fault
tree shows how **combinations** of basic events lead to the top event through
AND/OR logic.  This module gives a tiny FTA engine — probability propagation and
**minimal cut sets** (the smallest event combinations that cause the top event)
— plus a reference tree for thermal runaway whose basic events line up with the
twin's detectors, so each leaf is something the BMS actually monitors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product


@dataclass
class Event:
    """A fault-tree node: a *basic* event (leaf with a probability) or a *gate*.

    A leaf sets ``prob``; a gate sets ``gate`` (``"AND"`` / ``"OR"``) and
    ``children``.  ``detector`` names the twin signal/module that watches this
    basic event (documentation of the mitigation).
    """

    name: str
    prob: float | None = None
    gate: str | None = None
    children: list["Event"] = field(default_factory=list)
    detector: str | None = None

    @property
    def is_basic(self) -> bool:
        return self.gate is None


def probability(event: Event) -> float:
    """Top-down probability of *event* assuming independent basic events."""
    if event.is_basic:
        return float(event.prob or 0.0)
    ps = [probability(c) for c in event.children]
    if event.gate == "AND":
        out = 1.0
        for p in ps:
            out *= p
        return out
    if event.gate == "OR":
        out = 1.0
        for p in ps:
            out *= (1.0 - p)
        return 1.0 - out
    raise ValueError(f"unknown gate {event.gate!r}")


def minimal_cut_sets(event: Event) -> list[frozenset[str]]:
    """The minimal sets of basic events whose joint occurrence causes *event*."""
    if event.is_basic:
        return [frozenset({event.name})]
    child_sets = [minimal_cut_sets(c) for c in event.children]
    if event.gate == "OR":
        cuts = [s for cs in child_sets for s in cs]
    elif event.gate == "AND":
        cuts = [frozenset().union(*combo) for combo in product(*child_sets)]
    else:
        raise ValueError(f"unknown gate {event.gate!r}")
    # Drop non-minimal cut sets (any superset of another).
    minimal: list[frozenset[str]] = []
    for c in sorted(cuts, key=len):
        if not any(m <= c for m in minimal):
            minimal.append(c)
    return minimal


def basic_events(event: Event) -> dict[str, Event]:
    """Map basic-event name → node (for listing leaves and their detectors)."""
    if event.is_basic:
        return {event.name: event}
    out: dict[str, Event] = {}
    for c in event.children:
        out.update(basic_events(c))
    return out


def to_mermaid(event: Event) -> str:
    """Render the tree as a Mermaid ``graph TD`` (gates shown as {AND}/{OR})."""
    lines = ["graph TD"]
    counter = {"n": 0}

    def node_id() -> str:
        counter["n"] += 1
        return f"E{counter['n']}"

    def walk(ev: Event) -> str:
        nid = node_id()
        if ev.is_basic:
            lines.append(f'    {nid}(["{ev.name}\\np={ev.prob:g}"])')
        else:
            lines.append(f'    {nid}{{"{ev.name}\\n{ev.gate}"}}')
            for child in ev.children:
                cid = walk(child)
                lines.append(f"    {nid} --> {cid}")
        return nid

    walk(event)
    return "\n".join(lines)


def thermal_runaway_tree() -> Event:
    """Reference fault tree for thermal runaway; leaves map to twin detectors."""
    return Event("Thermal runaway", gate="OR", children=[
        Event("Internal short", gate="OR", children=[
            Event("Manufacturing defect", prob=1e-5, detector="faults.internal_short"),
            Event("Dendrite growth", prob=2e-5, detector="mechanics.swelling"),
            Event("Mechanical crush", prob=5e-6, detector="mechanics.pressure"),
        ]),
        Event("Uncontrolled overcharge", gate="AND", children=[
            Event("Charger overvoltage", prob=1e-3, detector="faults.overcharge"),
            Event("Protection failure", gate="OR", children=[
                Event("Voltage-sensor fault", prob=1e-3, detector="sensor_fdi.voltage"),
                Event("Contactor welded", prob=5e-4, detector="hv_safety.weld"),
            ]),
        ]),
        Event("Sustained overtemperature", gate="AND", children=[
            Event("Coolant failure", prob=2e-3, detector="control.cooling_duty"),
            Event("Temp-sensor blind", prob=1e-3, detector="sensor_fdi.temperature"),
        ]),
        Event("Propagation from neighbour", prob=3e-5, detector="propagation"),
    ])
