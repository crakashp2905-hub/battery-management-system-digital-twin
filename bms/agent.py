"""
Optional LLM diagnostic-agent layer over the twin — provider- and
framework-agnostic.

The core ``bms`` library stays dependency-light and deterministic; this is the
thin, **optional** layer where LLM tooling lives (``pip install '.[agent]'``). It:

* exposes the twin's read-only functions as **tools** (:func:`twin_tools`);
* runs a **deterministic diagnostic workflow** (:meth:`DiagnosticAgent.diagnose`)
  that needs no LLM, and optionally asks one for the recommendation;
* provides lazy **integration points** for LangChain (:func:`to_langchain_tools`),
  LangGraph (:func:`build_langgraph_agent`), and Langfuse (:func:`traced`) — each
  imported only when used, so the core has no hard dependency on them.

``llm`` everywhere is any callable ``str -> str`` (a Claude / OpenAI SDK call, a
LangChain model's ``.invoke``, or a local model), keeping the layer LLM-agnostic.
Claude is the natural default provider for this stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

_SEVERITY_ORDER = {"ok": 0, "info": 1, "warning": 2, "critical": 3}


@dataclass
class Tool:
    """A named callable the agent (or an LLM) can invoke."""

    name: str
    description: str
    fn: Callable

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


@dataclass
class Finding:
    """One diagnostic observation."""

    signal: str
    detail: str
    severity: str = "info"          # ok | info | warning | critical


@dataclass
class DiagnosisReport:
    severity: str
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "summary": self.summary,
            "recommendation": self.recommendation,
            "findings": [f.__dict__ for f in self.findings],
        }


def _worst(findings: list[Finding]) -> str:
    if not findings:
        return "ok"
    return max((f.severity for f in findings),
               key=lambda s: _SEVERITY_ORDER.get(s, 0))


class DiagnosticAgent:
    """Reason over the twin's signals and recommend an action.

    Deterministic by default; pass ``llm`` (any ``str -> str`` callable) to have
    the recommendation written by an LLM instead of the built-in rule table.
    """

    _RULES = {
        "fault": "Open the contactor, hold in FAULT, and require an operator reset.",
        "gas": "Gas/pressure rising — pre-emptively open the contactor and go to max "
               "cooling BEFORE the temperature rule trips.",
        "temperature": "Raise cooling duty and derate current now.",
        "soh": "Schedule maintenance and cap fast-charge C-rate (SoH-aware control).",
    }

    def __init__(self, llm: Callable[[str], str] | None = None,
                 tools: list[Tool] | None = None):
        self.llm = llm
        self.tools = tools or []

    def diagnose(self, result: dict, mechanical_state=None,
                 soh: float | None = None) -> DiagnosisReport:
        findings: list[Finding] = []
        state = str(result.get("state", "?"))

        fault = result.get("fault_label", "none")
        if fault != "none":
            findings.append(Finding(
                "fault", f"{fault} (via {result.get('fault_source', '?')} layer)", "critical"))

        if "T_cells" in result:
            t_max = float(np.max(result["T_cells"]))
            if t_max >= 55.0:
                findings.append(Finding(
                    "temperature", f"hottest cell {t_max:.0f} °C",
                    "critical" if t_max >= 70.0 else "warning"))

        if mechanical_state is not None:
            p = float(getattr(mechanical_state, "pressure_kPa", 0.0))
            if getattr(mechanical_state, "vent_event", False) or p >= 300.0:
                findings.append(Finding("gas", f"pressure {p:.0f} kPa / venting", "critical"))
            elif p >= 220.0:
                findings.append(Finding(
                    "gas", f"pressure {p:.0f} kPa rising (runaway precursor)", "warning"))

        if soh is not None and soh < 0.80:
            findings.append(Finding("soh", f"SoH {soh * 100:.0f}% below 80% EoL", "warning"))

        severity = _worst(findings)
        summary = (f"State {state.upper()}; "
                   + ("; ".join(f"{f.signal}: {f.detail}" for f in findings)
                      if findings else "all signals nominal") + ".")
        return DiagnosisReport(severity=severity, findings=findings, summary=summary,
                               recommendation=self._recommend(findings, severity, summary))

    def _recommend(self, findings: list[Finding], severity: str, summary: str) -> str:
        if self.llm is not None:
            prompt = ("You are a battery-safety engineer. Given these findings give ONE "
                      f"concise recommended action.\nSeverity: {severity}\nFindings: {summary}")
            return str(self.llm(prompt)).strip()
        if not findings:
            return "Continue normal operation."
        worst = max(findings, key=lambda f: _SEVERITY_ORDER.get(f.severity, 0))
        return self._RULES.get(worst.signal, "Investigate further.")


# ======================================================================
# Tools + framework integration points (all lazily imported)
# ======================================================================
def twin_tools(supervisor) -> list[Tool]:
    """Expose a supervisor's read-only functions as agent tools."""
    return [
        Tool("state_of_power", "Multi-horizon traction/regen power limits [W].",
             lambda: {h: lim.traction_power_W for h, lim in supervisor.state_of_power().items()}),
        Tool("soh", "Pack capacity state-of-health (0–1).",
             lambda: supervisor.soh_capacity),
        Tool("fault_log", "List of logged fault events.",
             lambda: supervisor.fault_log),
        Tool("passport", "Lifetime accounting snapshot (EFC, RTE, throughput).",
             lambda: supervisor.passport.summary()),
    ]


def to_langchain_tools(tools: list[Tool]):
    """Convert :class:`Tool`s to LangChain ``StructuredTool``s (requires langchain-core)."""
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:                       # pragma: no cover
        raise ImportError("LangChain not installed — `pip install '.[agent]'`") from exc
    return [StructuredTool.from_function(func=t.fn, name=t.name, description=t.description)
            for t in tools]


def build_langgraph_agent(llm: Callable[[str], str] | None = None,
                          tools: list[Tool] | None = None):
    """Build a LangGraph diagnostic workflow (requires langgraph).

    A compiled graph with an ``assess`` node running :meth:`DiagnosticAgent.diagnose`;
    invoke it with ``{"result": ..., "mechanical_state": ..., "soh": ...}`` and read
    ``["report"]``.  Extend with tool/act nodes as needed.
    """
    try:
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:                       # pragma: no cover
        raise ImportError("LangGraph not installed — `pip install '.[agent]'`") from exc

    agent = DiagnosticAgent(llm=llm, tools=tools)

    class _State(TypedDict, total=False):
        result: dict
        mechanical_state: object
        soh: float
        report: object

    def assess(state: "_State") -> dict:
        return {"report": agent.diagnose(state.get("result", {}),
                                         state.get("mechanical_state"), state.get("soh"))}

    graph = StateGraph(_State)
    graph.add_node("assess", assess)
    graph.add_edge(START, "assess")
    graph.add_edge("assess", END)
    return graph.compile()


def traced(name: str | None = None):
    """Decorator: trace the wrapped call with Langfuse if installed, else no-op.

    Lets you instrument LLM/agent calls for observability & evals without making
    Langfuse a hard dependency.
    """
    def decorator(fn: Callable) -> Callable:
        try:                                         # pragma: no cover
            from langfuse.decorators import observe
            return observe(name=name or fn.__name__)(fn)
        except ImportError:
            return fn
    return decorator


def langfuse_available() -> bool:
    """True if Langfuse is importable in this environment."""
    try:                                             # pragma: no cover
        import langfuse  # noqa: F401
        return True
    except ImportError:
        return False
