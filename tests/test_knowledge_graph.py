"""Tests: the battery knowledge graph."""

from __future__ import annotations

import bms


def _graph():
    g = bms.BatteryKnowledgeGraph()
    g.add_node("battery", "Battery")
    g.add_node("mod0", "Module")
    g.add_node("cell17", "Cell", soc=0.62)
    g.add_node("temp17", "Sensor", kind="temperature")
    g.add_node("fault42", "Fault", kind="thermal")
    g.add_edge("battery", "contains", "mod0")
    g.add_edge("mod0", "contains", "cell17")
    g.add_edge("cell17", "has_sensor", "temp17")
    g.add_edge("cell17", "contributed_to", "fault42")
    return g


class TestKnowledgeGraph:
    def test_typed_queries(self):
        g = _graph()
        assert g.nodes_of_type("Cell") == ["cell17"]
        assert "temp17" in g.neighbors("cell17", "has_sensor")
        assert g.neighbors("battery", "contains") == ["mod0"]

    def test_contributors_and_ancestry(self):
        g = _graph()
        assert g.contributors_to("fault42") == ["cell17"]     # which cell caused it
        # cell17 → module → battery, walking the containment hierarchy upward
        assert g.ancestors("cell17", "contains") == ["mod0", "battery"]

    def test_bad_edge_is_rejected(self):
        g = bms.BatteryKnowledgeGraph()
        g.add_node("a", "Cell")
        try:
            g.add_edge("a", "contains", "ghost")
            assert False, "expected KeyError"
        except KeyError:
            pass

    def test_mermaid_and_summary(self):
        g = _graph()
        m = g.to_mermaid()
        assert m.startswith("graph TD") and "contributed_to" in m
        s = g.summary()
        assert s["n_nodes"] == 5 and s["by_type"]["Cell"] == 1

    def test_build_pack_graph_hierarchy(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=8, seed=0))
        g = bms.build_pack_graph(pack, cells_per_module=4)
        assert len(g.nodes_of_type("Cell")) == 8
        assert len(g.nodes_of_type("Module")) == 2            # 8 cells / 4 per module
        # every cell traces back to the battery root
        for cell in g.nodes_of_type("Cell"):
            assert "battery" in g.ancestors(cell, "contains")
