"""FastAPI service exposing the twin as a live endpoint.

Turns the digital twin into a testable **service**: stream a device's measured
(voltage, current, temperature) at ``POST /step`` and get back the assimilated
SoC and a model-drift flag; query safety, reset, and metadata.  This is the
software-in-the-loop surface a real BMS (or a data replay) can talk to.

Kept out of ``bms/__init__`` so the core library never requires FastAPI; install
the service deps with ``pip install '.[api]'`` and create the app with
:func:`create_app`.  Run it with::

    uvicorn --factory bms.api:create_app --host 0.0.0.0 --port 8000
"""

# NOTE: no ``from __future__ import annotations`` here — FastAPI must see the
# real Pydantic model classes (not stringized annotations) to treat them as
# request bodies rather than query parameters.

from typing import Optional


def create_app(chemistry: str = "nmc", soc0: float = 1.0):
    """Build the FastAPI application backed by a live :class:`bms.TwinSync`."""
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    import bms

    props = bms.get_chemistry_props(chemistry)
    d = props["default_ecm"]
    params = bms.ECMParameters(R0=d["R0"], R1=d["R1"], C1=d["C1"], R2=d["R2"],
                               C2=d["C2"], Q_nom_Ah=props["default_capacity_Ah"],
                               chemistry=chemistry)
    ocv = bms.OCVSOC.from_chemistry(chemistry)
    twin = bms.TwinSync(params=params, ocv_curve=ocv)
    twin.reset(soc0)

    app = FastAPI(title="BMS Digital Twin API", version=bms.__version__)
    app.state.twin = twin
    app.state.chemistry = chemistry

    # ---- schemas -----------------------------------------------------
    class StepIn(BaseModel):
        voltage: float = Field(..., description="measured terminal voltage [V]")
        current: float = Field(..., description="measured current [A], >0 = discharge")
        temperature: float = Field(25.0, description="cell temperature [°C]")
        dt: float = Field(1.0, gt=0, description="time step [s]")

    class StepOut(BaseModel):
        soc: float
        predicted_voltage_V: float
        residual_V: float
        residual_rms_V: float
        drift: bool

    class SafetyIn(BaseModel):
        temperatures_C: list[float]
        cell_voltages_V: list[float]
        imbalance: float = 0.0
        soh: Optional[float] = None

    # ---- routes ------------------------------------------------------
    @app.get("/health")
    def health():
        return {"status": "ok", "version": bms.__version__,
                "chemistry": app.state.chemistry}

    @app.get("/info")
    def info():
        return {"chemistry": app.state.chemistry,
                "version": bms.__version__,
                "capacity_Ah": props["default_capacity_Ah"],
                "v_window": [props["v_min"], props["v_max"]],
                "estimators": bms.available_soc_estimators()}

    @app.post("/step", response_model=StepOut)
    def step(inp: StepIn):
        out = app.state.twin.assimilate(inp.current, inp.voltage, inp.dt,
                                        temperature_C=inp.temperature)
        return StepOut(soc=out["soc"], predicted_voltage_V=out["predicted_voltage_V"],
                       residual_V=out["residual_V"], residual_rms_V=out["residual_rms_V"],
                       drift=out["drift"])

    @app.get("/state")
    def state():
        t = app.state.twin
        return {"soc": t.soc, "residual_rms_V": (t._resid_sq_ewma ** 0.5)}

    @app.post("/reset")
    def reset(soc0: float = 1.0):
        app.state.twin.reset(soc0)
        return {"status": "reset", "soc": app.state.twin.soc}

    @app.post("/safety")
    def safety(inp: SafetyIn):
        import numpy as np
        result = {"T_cells": np.asarray(inp.temperatures_C, float),
                  "v_cells": np.asarray(inp.cell_voltages_V, float),
                  "imbalance": inp.imbalance}
        return bms.state_of_safety(result, chemistry=app.state.chemistry, soh=inp.soh)

    return app
