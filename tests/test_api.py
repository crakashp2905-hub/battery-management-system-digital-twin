"""Tests: FastAPI live-twin service."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from bms.api import create_app  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(create_app("nmc", soc0=0.9))


class TestAPI:
    def test_health_and_info(self, client):
        import bms
        h = client.get("/health").json()
        assert h["status"] == "ok" and h["version"] == bms.__version__
        info = client.get("/info").json()
        assert info["chemistry"] == "nmc"
        assert "ekf" in info["estimators"]

    def test_step_assimilates_and_returns_state(self, client):
        r = client.post("/step", json={"voltage": 3.7, "current": 2.0,
                                       "temperature": 25.0, "dt": 1.0})
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["soc"] <= 1.0
        assert set(body) == {"soc", "predicted_voltage_V", "residual_V",
                             "residual_rms_V", "drift"}
        assert isinstance(body["drift"], bool)

    def test_step_validates_input(self, client):
        # dt must be > 0; a bad body is a 422.
        assert client.post("/step", json={"voltage": 3.7, "current": 1.0,
                                          "dt": 0.0}).status_code == 422
        assert client.post("/step", json={"current": 1.0}).status_code == 422

    def test_reset_sets_soc(self, client):
        client.post("/step", json={"voltage": 3.6, "current": 5.0, "dt": 1.0})
        out = client.post("/reset", params={"soc0": 0.42}).json()
        assert out["soc"] == pytest.approx(0.42)
        assert client.get("/state").json()["soc"] == pytest.approx(0.42)

    def test_safety_endpoint(self, client):
        s = client.post("/safety", json={
            "temperatures_C": [25, 25, 72, 25],
            "cell_voltages_V": [3.7, 3.7, 3.7, 3.7], "soh": 0.9}).json()
        assert s["sos"] == 0.0                          # 72 °C cell → critical
        assert s["worst_signal"] == "temperature"

    def test_openapi_schema_is_served(self, client):
        assert client.get("/openapi.json").status_code == 200
