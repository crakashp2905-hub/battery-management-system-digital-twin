"""Tests: local fault explanations + second-life economics."""

from __future__ import annotations

import numpy as np

import bms
from bms._train_detector import generate_fault_training_data


class TestLocalFaultExplanation:
    @staticmethod
    def _fitted():
        det = bms.HybridFaultDetector(chemistry="nmc")
        X, y = generate_fault_training_data(chemistry="nmc")
        det.fit(X, y)
        return det, X, y

    def test_explains_a_flagged_sample(self):
        det, X, y = self._fitted()
        classes = list(det.clf.classes_)
        nf = classes.index("none") if "none" in classes else 0
        faulty = np.where(np.argmax(det.clf.predict_proba(X), axis=1) != nf)[0]
        ex = bms.explain_fault_prediction(det, X[faulty[0]])
        assert ex["predicted"] != "none"
        assert 0.0 <= ex["probability"] <= 1.0
        assert len(ex["contributions"]) == X.shape[1]
        # Contributions are ordered most-positive first.
        vals = list(ex["contributions"].values())
        assert vals == sorted(vals, reverse=True)

    def test_named_features_and_requires_fit(self):
        det, X, _ = self._fitted()
        ex = bms.explain_fault_prediction(det, X[0])
        assert set(ex["contributions"]).issuperset(set(bms.FEATURE_NAMES[:3]))
        raw = bms.HybridFaultDetector(chemistry="nmc")
        try:
            bms.explain_fault_prediction(raw, X[0])
            assert False, "should require a fitted detector"
        except RuntimeError:
            pass


class TestSecondLife:
    def test_verdict_thresholds(self):
        assert bms.assess_second_life(0.90).verdict == "first_life"
        assert bms.assess_second_life(0.72).verdict == "second_life"
        assert bms.assess_second_life(0.55).verdict == "recycle"

    def test_residual_value_decreases_with_soh(self):
        hi = bms.assess_second_life(0.78, capacity_kWh=60).residual_value
        lo = bms.assess_second_life(0.65, capacity_kWh=60).residual_value
        assert hi > lo > 0.0

    def test_second_life_years_positive_for_candidate_zero_for_recycle(self):
        assert bms.assess_second_life(0.72).second_life_years > 0.0
        assert bms.assess_second_life(0.55).second_life_years == 0.0

    def test_levelized_cost_drops_with_second_life(self):
        first_only = bms.levelized_cost_per_kWh(60, 130, first_life_cycles=1500)
        with_second = bms.levelized_cost_per_kWh(60, 130, first_life_cycles=1500,
                                                 second_life_cycles=1200)
        assert with_second < first_only
        assert first_only > 0.0
