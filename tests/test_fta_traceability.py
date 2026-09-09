"""Tests: Fault Tree Analysis + FMEA→test traceability."""

from __future__ import annotations

import bms
from bms.fta import Event


class TestFaultTree:
    def test_or_and_gate_probabilities(self):
        a = Event("a", prob=0.1)
        b = Event("b", prob=0.2)
        or_gate = Event("or", gate="OR", children=[a, b])
        and_gate = Event("and", gate="AND", children=[a, b])
        assert abs(bms.probability(or_gate) - (1 - 0.9 * 0.8)) < 1e-12   # 0.28
        assert abs(bms.probability(and_gate) - 0.02) < 1e-12

    def test_minimal_cut_sets_or_is_singletons(self):
        a, b = Event("a", prob=0.1), Event("b", prob=0.2)
        mcs = bms.minimal_cut_sets(Event("or", gate="OR", children=[a, b]))
        assert {frozenset({"a"}), frozenset({"b"})} == set(mcs)

    def test_minimal_cut_sets_and_is_the_combination(self):
        a, b = Event("a", prob=0.1), Event("b", prob=0.2)
        mcs = bms.minimal_cut_sets(Event("and", gate="AND", children=[a, b]))
        assert mcs == [frozenset({"a", "b"})]

    def test_thermal_runaway_tree_structure(self):
        tree = bms.thermal_runaway_tree()
        p = bms.probability(tree)
        assert 0.0 < p < 1e-3                       # rare top event
        mcs = bms.minimal_cut_sets(tree)
        # An internal-short basic event alone causes runaway (single-event cut).
        assert any(len(c) == 1 for c in mcs)
        # Overcharge needs BOTH a charger fault and a protection failure.
        assert any(len(c) == 2 for c in mcs)
        # Every basic event names a detector in the twin.
        for name, node in bms.basic_events(tree).items():
            assert node.detector, name

    def test_mermaid_renders_gates_and_leaves(self):
        m = bms.to_mermaid(bms.thermal_runaway_tree())
        assert m.startswith("graph TD")
        assert "OR" in m and "AND" in m


class TestFmeaTraceability:
    def test_all_high_rpn_modes_are_covered(self):
        ids = _collect()
        df = bms.fmea_traceability(collected_test_ids=ids, rpn_threshold=100)
        # No high-RPN failure mode is left without a covering test.
        assert not df["gap"].any(), df[df["gap"]]["failure_mode"].tolist()

    def test_gap_is_flagged_for_an_unlinked_high_rpn_mode(self):
        # With an empty id list nothing matches → every high-RPN mode is a gap.
        df = bms.fmea_traceability(collected_test_ids=[], rpn_threshold=100)
        assert df["gap"].any()
        assert not df["covered"].any()

    def test_links_cover_every_failure_mode(self):
        df = bms.build_fmea_table()
        for mode in df["failure_mode"]:
            assert mode in bms.FMEA_TEST_LINKS, mode


def _collect() -> list[str]:
    from pathlib import Path
    ids = []
    tests_dir = Path(__file__).resolve().parent
    for f in sorted(tests_dir.glob("test_*.py")):
        for line in f.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("class Test") or s.startswith("def test_"):
                name = s.split("(")[0].removeprefix("class ").removeprefix("def ").strip(": ")
                ids.append(f"{f.name}::{name}")
    return ids
