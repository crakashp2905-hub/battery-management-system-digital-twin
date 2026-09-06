"""Tests: agent."""

from __future__ import annotations

import numpy as np
import pytest

import bms


class TestInterpretability:
    def _fitted_detector(self):
        det = bms.HybridFaultDetector()
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, det._buffer.full_feature_size))
        y = rng.choice(["none", "overcharge", "thermal_runaway"], size=60)
        det.fit(X, y)
        return det

    def test_feature_importances_named_and_normalised(self):
        imp = bms.feature_importances(self._fitted_detector())
        assert set(imp) <= set(bms.FEATURE_NAMES)
        assert sum(imp.values()) == pytest.approx(1.0, abs=1e-6)
        values = list(imp.values())
        assert values == sorted(values, reverse=True)      # most-important first

    def test_feature_importances_requires_fit(self):
        with pytest.raises(RuntimeError):
            bms.feature_importances(bms.HybridFaultDetector())

    def test_estimator_agreement_inverse_variance_fusion(self):
        ests = {"loose": bms.Estimate(0.50, 0.10),
                "tight": bms.Estimate(0.60, 0.01)}
        ag = bms.estimator_agreement(ests)
        assert ag["spread"] == pytest.approx(0.10)
        assert ag["disagreement"] is True                  # 0.10 > 0.05
        assert ag["fused"] > 0.58                           # pulled toward the tight one
        assert ag["mean"] == pytest.approx(0.55)

    def test_soc_report_with_and_without_uncertainty(self):
        p, ocv = bms.ECMParameters(), bms.OCVSOC()
        ekf = bms.make_soc_estimator("ekf", params=p, ocv_curve=ocv)
        assert "±" in bms.soc_report(ekf)
        assert "unknown" in bms.soc_report(bms.make_soc_estimator("coulomb"))

    def test_explain_state_and_charge(self):
        sup = bms.BMSSupervisor(bms.BatteryPack(bms.PackConfig(n_cells=4, seed=1)),
                                bms.ThermalModel(n_cells=4),
                                bms.HybridFaultDetector())
        text = bms.explain_state(sup.step(2.0, 1.0), soh=0.9)
        assert "SoC" in text and "SoH 90.0%" in text
        cr = bms.ChargingModel().simulate(
            bms.ChargeProtocol(bms.ChargeMethod.DC_ULTRA, soc_start=0.2, soc_end=0.9),
            pack_energy_kWh=60.0, q_nom_Ah=2.3, r0_ohm=0.025, v_nom=3.7)
        assert "plating" in bms.explain_charge(cr).lower()



class TestLLMNarrationAndAgent:
    def _result(self, fault=False):
        sup = bms.BMSSupervisor(bms.BatteryPack(bms.PackConfig(n_cells=4, seed=1)),
                                bms.ThermalModel(n_cells=4), bms.HybridFaultDetector())
        r = sup.step(2.0, 1.0)
        if fault:
            r = dict(r, fault_label="thermal_runaway", fault_source="rule")
        return r, sup

    def test_explain_state_llm_hook(self):
        r, _ = self._result()
        assert bms.explain_state(r, llm=lambda p: "NARRATED") == "NARRATED"
        assert "State:" in bms.explain_state(r)          # deterministic default

    def test_explain_charge_llm_hook(self):
        cr = bms.ChargingModel().simulate(
            bms.ChargeProtocol(bms.ChargeMethod.DC_FAST, soc_start=0.2, soc_end=0.9),
            pack_energy_kWh=60.0, q_nom_Ah=2.3, r0_ohm=0.025, v_nom=3.7)
        assert bms.explain_charge(cr, llm=lambda p: "CHARGE NARRATION") == "CHARGE NARRATION"

    def test_agent_nominal_and_fault(self):
        r, _ = self._result()
        rep = bms.DiagnosticAgent().diagnose(r)
        assert rep.severity == "ok" and "normal" in rep.recommendation.lower()
        rf, _ = self._result(fault=True)
        repf = bms.DiagnosticAgent().diagnose(rf)
        assert repf.severity == "critical" and "contactor" in repf.recommendation.lower()

    def test_agent_gas_precursor_and_soh(self):
        r, _ = self._result()
        gas = bms.DiagnosticAgent().diagnose(r, mechanical_state=bms.CellMechanicalState(pressure_kPa=250.0))
        assert gas.severity == "warning"
        assert any(f.signal == "gas" for f in gas.findings)
        aged = bms.DiagnosticAgent().diagnose(r, soh=0.7)
        assert any(f.signal == "soh" for f in aged.findings)

    def test_agent_llm_recommendation_and_report_dict(self):
        rf, _ = self._result(fault=True)
        rep = bms.DiagnosticAgent(llm=lambda p: "MOCK ACTION").diagnose(rf)
        assert rep.recommendation == "MOCK ACTION"
        assert set(rep.to_dict()) == {"severity", "summary", "recommendation", "findings"}

    def test_twin_tools_read_only(self):
        _, sup = self._result()
        tools = bms.twin_tools(sup)
        names = {t.name for t in tools}
        assert {"state_of_power", "soh", "fault_log", "passport"} <= names
        assert next(t for t in tools if t.name == "soh")() == pytest.approx(1.0)

    def test_traced_is_noop_without_langfuse(self):
        assert bms.traced("x")(lambda a, b: a + b)(2, 3) == 5

    def test_langgraph_agent_builds(self):
        pytest.importorskip("langgraph")
        graph = bms.build_langgraph_agent(llm=None)
        r, _ = self._result(fault=True)
        out = graph.invoke({"result": r})
        assert out["report"].severity == "critical"

    def test_langchain_tools_convert(self):
        pytest.importorskip("langchain_core")
        _, sup = self._result()
        lc = bms.to_langchain_tools(bms.twin_tools(sup))
        assert len(lc) == 4

