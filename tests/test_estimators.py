"""Tests: estimators."""

from __future__ import annotations

import numpy as np
import pytest

import bms


@pytest.fixture(scope="module")
def truth_trace():
    oc = bms.OCVSOC()
    p = bms.ECMParameters(R0=0.025, R1=0.012, C1=2500, R2=0.025,
                          C2=10_000, Q_nom_Ah=2.3)
    m = bms.SecondOrderECM(params=p, ocv_curve=oc)
    n = 1800
    i = np.zeros(n)
    i[100:600] = 1.5
    i[800:1400] = 2.5
    out = m.simulate(i, 1.0, soc0=0.95)
    return p, oc, i, out["v_terminal"], out["soc"]


class TestEstimators:
    def test_coulomb_counting_no_bias(self, truth_trace):
        p, oc, i, v, soc = truth_trace
        cc = bms.CoulombCounter(p.Q_nom_Ah, soc0=0.95)
        soc_hat = cc.run(i, v, 1.0)
        assert np.max(np.abs(soc_hat - soc)) < 5e-3

    def test_ekf_corrects_bias(self, truth_trace):
        p, oc, i, v, soc = truth_trace
        i_biased = i + 0.05
        ekf = bms.EKFEstimator(params=p, ocv_curve=oc); ekf.reset(0.92)
        soc_hat = ekf.run(i_biased, v, 1.0)
        rmse = np.sqrt(np.mean((soc_hat - soc) ** 2))
        # CC with the same bias would drift by several percent — EKF must
        # do an order of magnitude better.
        assert rmse < 0.01

    def test_ukf_corrects_bias(self, truth_trace):
        p, oc, i, v, soc = truth_trace
        ukf = bms.UKFEstimator(params=p, ocv_curve=oc, dt=1.0); ukf.reset(0.92)
        soc_hat = ukf.run(i + 0.05, v, 1.0)
        assert np.sqrt(np.mean((soc_hat - soc) ** 2)) < 0.01



class TestEstimatorRegistry:
    @staticmethod
    def _trace():
        p = bms.ECMParameters()
        ocv = bms.OCVSOC()
        ecm = bms.SecondOrderECM(params=p, ocv_curve=ocv)
        ecm.reset(0.8)
        current = np.full(200, 1.0)
        sim = ecm.simulate(current, dt=1.0)
        return p, ocv, current, sim["v_terminal"], sim["soc"]

    def test_registry_lists_builtins(self):
        assert {"coulomb", "ekf", "ukf", "lstm"} <= set(bms.available_soc_estimators())

    def test_all_registered_satisfy_protocol(self):
        p, ocv, *_ = self._trace()
        for name in bms.available_soc_estimators():
            est = bms.make_soc_estimator(name, params=p, ocv_curve=ocv, capacity_Ah=2.3)
            assert isinstance(est, bms.SocEstimator), name

    def test_recursive_vs_batch_distinction(self):
        p, ocv, *_ = self._trace()
        for name in ("coulomb", "ekf", "ukf"):
            est = bms.make_soc_estimator(name, params=p, ocv_curve=ocv)
            assert isinstance(est, bms.RecursiveSocEstimator), name
        # The sequence LSTM is a SocEstimator but not a recursive one.
        lstm = bms.make_soc_estimator("lstm")
        assert isinstance(lstm, bms.SocEstimator)
        assert not isinstance(lstm, bms.RecursiveSocEstimator)

    def test_recursive_estimators_track_and_stay_bounded(self):
        p, ocv, current, voltage, truth = self._trace()
        for name in ("coulomb", "ekf", "ukf"):
            est = bms.make_soc_estimator(name, params=p, ocv_curve=ocv, capacity_Ah=2.3)
            est.reset(0.8)
            out = est.run(current, voltage, 1.0)
            assert np.all((out >= 0.0) & (out <= 1.0)), name
            assert abs(out[-1] - truth[-1]) < 0.05, name

    def test_uncertainty_reported_when_available(self):
        p, ocv, current, voltage, _ = self._trace()
        ekf = bms.make_soc_estimator("ekf", params=p, ocv_curve=ocv)
        ekf.reset(0.8)
        ekf.run(current, voltage, 1.0)
        assert bms.soc_estimate(ekf).has_uncertainty
        assert not bms.soc_estimate(bms.make_soc_estimator("coulomb")).has_uncertainty

    def test_function_estimator_is_model_agnostic(self):
        # Wrap an arbitrary callable (stand-in for a trained sklearn/torch/onnx
        # model) as a first-class estimator.
        est = bms.FunctionSocEstimator(
            lambda v, i, dt, T, s: s - i * dt / (2.3 * 3600.0), name="byo")
        assert isinstance(est, bms.SocEstimator)
        out = est.run(np.full(100, 1.0), np.full(100, 3.7), 1.0, soc0=0.9)
        assert np.all((out >= 0.0) & (out <= 1.0)) and out[-1] < 0.9

    def test_unknown_estimator_raises(self):
        with pytest.raises(KeyError):
            bms.make_soc_estimator("does_not_exist")

    def test_custom_registration_round_trips(self):
        @bms.register_soc_estimator("const_half")
        def _factory(**_):
            return bms.FunctionSocEstimator(lambda v, i, dt, T, s: 0.5,
                                            name="const_half")
        assert "const_half" in bms.available_soc_estimators()
        assert bms.make_soc_estimator("const_half").update(1.0, 3.7, 1.0) == 0.5



class TestJointEKFSoH:
    @staticmethod
    def _aged_trace(q_true=2.0, n=2200):
        ocv = bms.OCVSOC()
        ecm = bms.SecondOrderECM(params=bms.ECMParameters(Q_nom_Ah=q_true),
                                 ocv_curve=ocv)
        ecm.reset(0.95)
        current = np.full(n, 1.0)
        sim = ecm.simulate(current, dt=1.0)
        rng = np.random.default_rng(0)
        return current, sim["v_terminal"] + rng.normal(0, 0.003, n), sim["soc"]

    def test_in_registry_and_recursive(self):
        assert "joint_ekf" in bms.available_soc_estimators()
        est = bms.make_soc_estimator("joint_ekf", capacity_Ah=2.3)
        assert isinstance(est, bms.RecursiveSocEstimator)
        assert hasattr(est, "soh") and hasattr(est, "capacity_Ah")

    def test_learns_aged_capacity(self):
        current, voltage, _ = self._aged_trace(q_true=2.0)
        est = bms.make_soc_estimator("joint_ekf", capacity_Ah=2.3)  # BOL guess 2.3
        est.reset(0.95)
        est.run(current, voltage, 1.0)
        # Converges from 2.3 toward the true aged 2.0 Ah.
        assert est.capacity_Ah < 2.2
        assert abs(est.capacity_Ah - 2.0) < abs(2.3 - 2.0)     # improved
        assert 0.80 < est.soh < 0.96
        assert est.soh_uncertainty_1sigma > 0.0

    def test_soc_bounded_and_tracks(self):
        current, voltage, truth = self._aged_trace(q_true=2.0)
        est = bms.make_soc_estimator("joint_ekf", capacity_Ah=2.3)
        est.reset(0.95)
        out = est.run(current, voltage, 1.0)
        assert np.all((out >= 0.0) & (out <= 1.0))
        assert abs(out[-1] - truth[-1]) < 0.03



class TestModelAdapters:
    @staticmethod
    def _fitted_linear():
        from sklearn.linear_model import LinearRegression
        d = bms.synthetic_drivecycle("nmc", duration_s=500, seed=1)
        X = np.column_stack([d.voltage_V, d.current_A, d.temperature_C])
        return LinearRegression().fit(X, d.soc_true), d

    def test_sklearn_adapter_in_protocol_and_tracks(self):
        model, d = self._fitted_linear()
        est = bms.SklearnSocEstimator(model, feature_fn=lambda v, i, dt, T, s: [v, i, T])
        assert isinstance(est, bms.RecursiveSocEstimator)
        est.reset(float(d.soc_true[0]))
        out = est.run(d.current_A, d.voltage_V, d.dt)
        assert np.all((out >= 0.0) & (out <= 1.0))
        assert np.sqrt(np.mean((out - d.soc_true) ** 2)) < 0.05

    def test_model_from_file_sklearn(self, tmp_path):
        import joblib
        model, _ = self._fitted_linear()
        p = tmp_path / "m.joblib"
        joblib.dump(model, p)
        est = bms.model_from_file(p, trust_pickle=True,
                                  feature_fn=lambda v, i, dt, T, s: [v, i, T])
        assert isinstance(est, bms.SklearnSocEstimator)
        assert 0.0 <= est.update(1.0, 3.7, 1.0) <= 1.0

    def test_pickle_load_requires_explicit_trust(self, tmp_path):
        import joblib
        model, _ = self._fitted_linear()
        p = tmp_path / "m.joblib"
        joblib.dump(model, p)
        with pytest.raises(ValueError, match="arbitrary code"):
            bms.model_from_file(p)                       # refused by default

    def test_model_from_file_checksum(self, tmp_path):
        import hashlib

        import joblib
        model, _ = self._fitted_linear()
        p = tmp_path / "m.joblib"
        joblib.dump(model, p)
        good = hashlib.sha256(p.read_bytes()).hexdigest()
        bms.model_from_file(p, trust_pickle=True, sha256=good)          # ok
        with pytest.raises(ValueError, match="checksum mismatch"):
            bms.model_from_file(p, trust_pickle=True, sha256="00" * 32)

    def test_model_from_file_unsupported(self, tmp_path):
        with pytest.raises(ValueError):
            bms.model_from_file(tmp_path / "model.bin")

    def test_onnx_adapter_roundtrip(self, tmp_path):
        pytest.importorskip("onnxruntime")
        skl2onnx = pytest.importorskip("skl2onnx")
        from skl2onnx.common.data_types import FloatTensorType
        model, d = self._fitted_linear()
        onx = skl2onnx.to_onnx(model, initial_types=[("x", FloatTensorType([None, 3]))])
        p = tmp_path / "m.onnx"
        p.write_bytes(onx.SerializeToString())
        est = bms.model_from_file(p, feature_fn=lambda v, i, dt, T, s: [v, i, T])
        assert isinstance(est, bms.OnnxSocEstimator)
        out = est.run(d.current_A[:50], d.voltage_V[:50], d.dt)
        assert np.all((out >= 0.0) & (out <= 1.0))

