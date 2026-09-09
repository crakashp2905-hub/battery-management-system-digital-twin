"""Tests: model-card + out-of-distribution detection for plugged-in models."""

from __future__ import annotations

import numpy as np
import pytest

import bms


class TestModelCard:
    def test_defaults_and_feature_contract(self):
        card = bms.ModelCard(feature_names=["v", "i", "T", "prev_soc"],
                             chemistry="nmc", val_rmse=0.012)
        assert card.feature_names == ["v", "i", "T", "prev_soc"]
        assert card.model_version == "1.0"
        assert card.created is not None            # auto-stamped date


class TestOODDetector:
    @staticmethod
    def _train():
        rng = np.random.default_rng(0)
        # [voltage, current, temperature, prev_soc] in a realistic band.
        return rng.normal([3.7, 0.0, 25.0, 0.5], [0.1, 2.0, 5.0, 0.2], size=(800, 4))

    def test_flags_out_of_distribution_inputs(self):
        ood = bms.OODDetector.fit(self._train(), quantile=0.99)
        assert not ood.is_ood([3.7, 0.0, 25.0, 0.5])      # nominal
        assert ood.is_ood([3.7, 60.0, 90.0, 0.5])          # wild current + temp
        assert ood.mahalanobis_sq([3.7, 60.0, 90.0, 0.5]) > ood.threshold

    def test_false_positive_rate_is_near_the_quantile(self):
        X = self._train()
        ood = bms.OODDetector.fit(X, quantile=0.99, margin=1.0)
        flags = np.array([ood.is_ood(x) for x in X])
        assert flags.mean() < 0.05                          # ~1% by construction


class TestEstimatorOODWiring:
    @staticmethod
    def _fit_model_and_ood():
        from sklearn.linear_model import LinearRegression
        rng = np.random.default_rng(1)
        X = rng.normal([3.7, 0.0, 25.0, 0.5], [0.1, 2.0, 5.0, 0.2], size=(600, 4))
        y = X[:, 3]                                          # SoC ≈ prev_soc
        model = LinearRegression().fit(X, y)
        return model, bms.OODDetector.fit(X, quantile=0.99)

    def test_estimator_counts_and_flags_ood(self):
        pytest.importorskip("sklearn")
        model, ood = self._fit_model_and_ood()
        card = bms.ModelCard(feature_names=["v", "i", "T", "prev_soc"])
        est = bms.SklearnSocEstimator(model, ood_detector=ood, card=card)
        est.reset(0.5)
        est.update(current=0.0, voltage=3.7, dt=1.0, temperature_C=25.0)   # nominal
        assert est.last_ood is False
        est.update(current=0.0, voltage=3.7, dt=1.0, temperature_C=200.0)  # absurd T
        assert est.last_ood is True
        assert est.n_ood == 1
        assert est.card is card                              # provenance attached

    def test_no_detector_means_no_flagging(self):
        pytest.importorskip("sklearn")
        model, _ = self._fit_model_and_ood()
        est = bms.SklearnSocEstimator(model)
        est.reset(0.5)
        est.update(current=0.0, voltage=3.7, dt=1.0, temperature_C=200.0)
        assert est.last_ood is False and est.n_ood == 0
