"""Battery knowledge graph — structured history the agent can query.

A pile of telemetry is hard to reason over; a **graph** of the battery's parts
and events is not.  This module represents the system as typed nodes —
``Battery → Module → Cell → Sensor``, plus ``Fault`` and ``Degradation`` events —
linked by named relations (``contains``, ``has_sensor``, ``experienced``,
``contributed_to``).  A diagnostic layer can then ask structured questions:

* which cells contributed to fault 42?
* what sensors are on cell 17?
* which module does the limiting cell belong to?

It is deliberately dependency-light — a typed directed adjacency list, no
``networkx`` — and renders to Mermaid for inspection.  :func:`build_pack_graph`
seeds the Battery→Module→Cell hierarchy from a :class:`bms.BatteryPack`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BatteryKnowledgeGraph:
    """A typed directed graph of battery parts and events."""

    nodes: dict = field(default_factory=dict)          # id -> {"type", **attrs}
    edges: list = field(default_factory=list)          # (src, relation, dst)

    # ---- construction --------------------------------------------------
    def add_node(self, node_id: str, node_type: str, **attrs) -> str:
        self.nodes[node_id] = {"type": node_type, **attrs}
        return node_id

    def add_edge(self, src: str, relation: str, dst: str) -> None:
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError(f"edge references unknown node: {src!r} -> {dst!r}")
        self.edges.append((src, relation, dst))

    # ---- queries -------------------------------------------------------
    def nodes_of_type(self, node_type: str) -> list:
        return [nid for nid, a in self.nodes.items() if a["type"] == node_type]

    def neighbors(self, node_id: str, relation: str | None = None) -> list:
        """Outgoing neighbours of ``node_id`` (optionally filtered by relation)."""
        return [d for (s, r, d) in self.edges
                if s == node_id and (relation is None or r == relation)]

    def predecessors(self, node_id: str, relation: str | None = None) -> list:
        """Incoming neighbours — who points *at* ``node_id``."""
        return [s for (s, r, d) in self.edges
                if d == node_id and (relation is None or r == relation)]

    def related(self, node_id: str, relation: str) -> list:
        """Neighbours in either direction along ``relation``."""
        return self.neighbors(node_id, relation) + self.predecessors(node_id, relation)

    def contributors_to(self, event_id: str) -> list:
        """Nodes linked to an event by ``contributed_to`` (e.g. cells → a fault)."""
        return self.predecessors(event_id, "contributed_to")

    def ancestors(self, node_id: str, relation: str = "contains") -> list:
        """Walk incoming ``relation`` edges up to the root (e.g. cell → module → battery)."""
        chain, cur = [], node_id
        seen = set()
        while True:
            parents = self.predecessors(cur, relation)
            if not parents or parents[0] in seen:
                break
            cur = parents[0]
            seen.add(cur)
            chain.append(cur)
        return chain

    # ---- rendering -----------------------------------------------------
    def to_mermaid(self) -> str:
        lines = ["graph TD"]
        for nid, a in self.nodes.items():
            lines.append(f'    {nid}["{a["type"]}: {nid}"]')
        for s, r, d in self.edges:
            lines.append(f"    {s} -->|{r}| {d}")
        return "\n".join(lines)

    def summary(self) -> dict:
        by_type: dict = {}
        for a in self.nodes.values():
            by_type[a["type"]] = by_type.get(a["type"], 0) + 1
        return {"n_nodes": len(self.nodes), "n_edges": len(self.edges),
                "by_type": by_type}


def build_pack_graph(pack, *, battery_id: str = "battery",
                     cells_per_module: int = 4) -> BatteryKnowledgeGraph:
    """Seed a Battery → Module → Cell graph from a :class:`bms.BatteryPack`.

    Series groups are grouped into modules of ``cells_per_module`` for a
    realistic three-level hierarchy; each cell node carries its SoC and capacity.
    """
    g = BatteryKnowledgeGraph()
    g.add_node(battery_id, "Battery", n_cells=pack.total_cells)
    socs = pack.soc
    caps = pack.capacities_Ah
    for gi in range(pack.n_cells):
        module_id = f"module_{gi // cells_per_module}"
        if module_id not in g.nodes:
            g.add_node(module_id, "Module")
            g.add_edge(battery_id, "contains", module_id)
        cell_id = f"cell_{gi}"
        g.add_node(cell_id, "Cell", soc=float(socs[gi]), capacity_Ah=float(caps[gi]))
        g.add_edge(module_id, "contains", cell_id)
    return g
