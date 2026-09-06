"""Tests: chemistry."""

from __future__ import annotations

import numpy as np
import pytest

import bms


class TestLFPChemistry:
    def test_lfp_ocv_endpoints(self):
        oc = bms.OCVSOC.from_chemistry("lfp")
        assert oc.ocv(0.0) == pytest.approx(2.500, abs=1e-3)
        assert oc.ocv(1.0) == pytest.approx(3.650, abs=1e-3)

    def test_lfp_ocv_monotonic(self):
        oc = bms.OCVSOC.from_chemistry("lfp")
        v = oc.ocv(np.linspace(0, 1, 200))
        assert np.all(np.diff(v) >= -1e-6)

    def test_lfp_ecm_params(self):
        p = bms.ECMParameters.for_lfp()
        assert p.chemistry == "lfp"
        assert p.R0 == pytest.approx(0.020, rel=1e-6)
        assert p.Q_nom_Ah == pytest.approx(3.2, rel=1e-6)

    def test_lfp_arrhenius_different_from_nmc(self):
        p_nmc = bms.ECMParameters.for_nmc()
        p_lfp = bms.ECMParameters.for_lfp()
        # At 0 °C, LFP resistance increases less than NMC (lower Ea/R)
        r_nmc_cold = p_nmc.at_temperature(0.0).R0
        r_lfp_cold = p_lfp.at_temperature(0.0).R0
        # LFP R0 at 0°C / LFP R0 at 25°C < NMC R0 at 0°C / NMC R0 at 25°C
        ratio_nmc = r_nmc_cold / p_nmc.R0
        ratio_lfp = r_lfp_cold / p_lfp.R0
        assert ratio_lfp < ratio_nmc, "LFP should have lower Arrhenius scaling ratio"

    def test_lfp_pack_voltage_lower_than_nmc(self):
        p_nmc = bms.BatteryPack(bms.PackConfig(n_cells=4, chemistry="nmc", seed=0))
        p_lfp = bms.BatteryPack(bms.PackConfig(n_cells=4, chemistry="lfp", seed=0))
        assert p_lfp.pack_voltage() < p_nmc.pack_voltage()

    def test_lfp_fault_detector_higher_runaway_threshold(self):
        det_nmc = bms.HybridFaultDetector(chemistry="nmc")
        det_lfp = bms.HybridFaultDetector(chemistry="lfp")
        v = np.array([3.2, 3.2, 3.2, 3.2])
        T = np.array([75., 75., 75., 75.])
        feats = bms.extract_features(v, np.zeros(4), T, np.zeros(4), np.zeros(4))
        label_nmc, _ = det_nmc.predict_step(feats, v, T)
        label_lfp, _ = det_lfp.predict_step(feats, v, T)
        # 75°C > NMC threshold (70°C) but < LFP threshold (90°C)
        assert label_nmc == "thermal_runaway"
        assert label_lfp == "none"

    def test_lfp_overcharge_lower_threshold(self):
        det = bms.HybridFaultDetector(chemistry="lfp")
        v = np.array([3.2, 3.72, 3.2, 3.2])   # 3.72 > LFP threshold 3.70
        T = np.array([25., 25., 25., 25.])
        feats = bms.extract_features(v, np.zeros(4), T, np.zeros(4), np.zeros(4))
        label, src = det.predict_step(feats, v, T)
        assert label == "overcharge"
        assert src == "rule"

    def test_cell_chemistry_enum(self):
        assert bms.CellChemistry.NMC.value == "nmc"
        assert bms.CellChemistry.LFP.value == "lfp"
        props = bms.get_chemistry_props(bms.CellChemistry.LFP)
        assert props["nominal_voltage_V"] == pytest.approx(3.2)



class TestNewChemistries:
    # ── LMFP ─────────────────────────────────────────────────────────
    def test_lmfp_ocv_endpoints(self):
        oc = bms.OCVSOC.from_chemistry("lmfp")
        assert oc.ocv(0.0) == pytest.approx(2.800, abs=1e-3)
        assert oc.ocv(1.0) == pytest.approx(4.150, abs=1e-3)

    def test_lmfp_ocv_monotonic(self):
        oc = bms.OCVSOC.from_chemistry("lmfp")
        v = oc.ocv(np.linspace(0, 1, 200))
        assert np.all(np.diff(v) >= -1e-6)

    def test_lmfp_ecm_params(self):
        p = bms.ECMParameters.for_lmfp()
        assert p.chemistry == "lmfp"
        assert p.R0 == pytest.approx(0.018, rel=1e-6)
        assert p.Q_nom_Ah == pytest.approx(3.0, rel=1e-6)

    def test_lmfp_higher_runaway_than_nmc(self):
        props_nmc = bms.get_chemistry_props("nmc")
        props_lmfp = bms.get_chemistry_props("lmfp")
        assert props_lmfp["T_runaway_C"] > props_nmc["T_runaway_C"]

    # ── LTO ─────────────────────────────────────────────────────────
    def test_lto_ocv_endpoints(self):
        oc = bms.OCVSOC.from_chemistry("lto")
        assert oc.ocv(0.0) == pytest.approx(1.500, abs=1e-3)
        assert oc.ocv(1.0) == pytest.approx(2.800, abs=1e-3)

    def test_lto_ocv_monotonic(self):
        oc = bms.OCVSOC.from_chemistry("lto")
        v = oc.ocv(np.linspace(0, 1, 200))
        assert np.all(np.diff(v) >= -1e-6)

    def test_lto_ecm_very_low_r0(self):
        p = bms.ECMParameters.for_lto()
        assert p.chemistry == "lto"
        # LTO R0 (0.008) is much lower than NMC (0.025)
        p_nmc = bms.ECMParameters.for_nmc()
        assert p.R0 < p_nmc.R0

    def test_lto_pack_lower_voltage_than_nmc(self):
        p_lto = bms.BatteryPack(bms.PackConfig(n_cells=4, chemistry="lto", seed=0))
        p_nmc = bms.BatteryPack(bms.PackConfig(n_cells=4, chemistry="nmc", seed=0))
        assert p_lto.pack_voltage() < p_nmc.pack_voltage()

    def test_lto_highest_runaway_threshold(self):
        props = bms.get_chemistry_props("lto")
        for chem in ["nmc", "lfp", "lmfp", "nca", "lmo"]:
            assert props["T_runaway_C"] >= bms.get_chemistry_props(chem)["T_runaway_C"]

    # ── NCA ──────────────────────────────────────────────────────────
    def test_nca_ocv_endpoints(self):
        oc = bms.OCVSOC.from_chemistry("nca")
        assert oc.ocv(0.0) == pytest.approx(3.000, abs=1e-3)
        assert oc.ocv(1.0) == pytest.approx(4.200, abs=1e-3)

    def test_nca_ocv_monotonic(self):
        oc = bms.OCVSOC.from_chemistry("nca")
        v = oc.ocv(np.linspace(0, 1, 200))
        assert np.all(np.diff(v) >= -1e-6)

    def test_nca_lower_runaway_than_nmc(self):
        props_nmc = bms.get_chemistry_props("nmc")
        props_nca = bms.get_chemistry_props("nca")
        # NCA is LESS thermally stable — lower onset temperature
        assert props_nca["T_runaway_C"] < props_nmc["T_runaway_C"]

    def test_nca_higher_arrhenius_than_nmc(self):
        p_nmc = bms.ECMParameters.for_nmc()
        p_nca = bms.ECMParameters.for_nca()
        # NCA is more temperature-sensitive
        assert p_nca._arrhenius_K > p_nmc._arrhenius_K

    # ── LMO ──────────────────────────────────────────────────────────
    def test_lmo_ocv_endpoints(self):
        oc = bms.OCVSOC.from_chemistry("lmo")
        assert oc.ocv(0.0) == pytest.approx(3.000, abs=1e-3)
        assert oc.ocv(1.0) == pytest.approx(4.200, abs=1e-3)

    def test_lmo_ocv_monotonic(self):
        oc = bms.OCVSOC.from_chemistry("lmo")
        v = oc.ocv(np.linspace(0, 1, 200))
        assert np.all(np.diff(v) >= -1e-6)

    def test_lmo_highest_self_discharge(self):
        props_lmo = bms.get_chemistry_props("lmo")
        for chem in ["nmc", "lfp", "lmfp", "lto", "nca"]:
            assert props_lmo["self_discharge_pct_per_month"] > \
                   bms.get_chemistry_props(chem)["self_discharge_pct_per_month"]

    def test_lmo_lowest_runaway_threshold(self):
        # LMO has lowest thermal-runaway onset due to Mn dissolution
        props_lmo = bms.get_chemistry_props("lmo")
        for chem in ["nmc", "lfp", "lmfp", "lto", "nca"]:
            assert props_lmo["T_runaway_C"] <= bms.get_chemistry_props(chem)["T_runaway_C"]

    def test_all_chemistry_enum_values(self):
        for chem in bms.CellChemistry:
            props = bms.get_chemistry_props(chem)
            assert "ocv_table" in props
            assert "v_overcharge" in props
            assert "T_runaway_C" in props

    def test_unknown_chemistry_raises(self):
        with pytest.raises(ValueError):
            bms.get_chemistry_props("unknownium")



class TestSSBChemistry:
    def test_ssb_in_enum(self):
        assert bms.CellChemistry.SSB == "ssb"

    def test_ssb_props_accessible(self):
        p = bms.get_chemistry_props("ssb")
        assert "ocv_table" in p
        assert p["default_capacity_Ah"] == pytest.approx(4.0)

    def test_ssb_ocv_monotonic(self):
        p = bms.get_chemistry_props("ssb")
        ocv_vals = p["ocv_table"][:, 1]
        assert np.all(np.diff(ocv_vals) > 0), "SSB OCV table must be strictly increasing"

    def test_ssb_highest_voltage(self):
        ssb_p = bms.get_chemistry_props("ssb")
        nmc_p = bms.get_chemistry_props("nmc")
        assert ssb_p["v_max"] > nmc_p["v_max"]

    def test_ssb_highest_t_runaway(self):
        runaway_temps = {
            ch: bms.get_chemistry_props(ch)["T_runaway_C"]
            for ch in ["nmc", "lfp", "lmfp", "lto", "nca", "lmo", "ssb"]
        }
        assert runaway_temps["ssb"] == max(runaway_temps.values()), \
            "SSB must have the highest thermal-runaway temperature"

    def test_ssb_highest_arrhenius(self):
        arrhenius = {
            ch: bms.get_chemistry_props(ch)["arrhenius_K"]
            for ch in ["nmc", "lfp", "lmfp", "lto", "nca", "lmo", "ssb"]
        }
        assert arrhenius["ssb"] == max(arrhenius.values()), \
            "SSB solid electrolyte must have the highest Arrhenius constant"

    def test_ssb_lowest_self_discharge(self):
        sd = {
            ch: bms.get_chemistry_props(ch)["self_discharge_pct_per_month"]
            for ch in ["nmc", "lfp", "lmfp", "lto", "nca", "lmo", "ssb"]
        }
        assert sd["ssb"] == min(sd.values()), \
            "SSB must have the lowest self-discharge rate"

    def test_for_ssb_classmethod(self):
        params = bms.ECMParameters.for_ssb()
        assert params.chemistry == "ssb"
        assert params.Q_nom_Ah == pytest.approx(4.0)
        assert params.self_discharge_pct_per_month == pytest.approx(0.3)

    def test_ssb_ocvsoc_roundtrip(self):
        oc = bms.OCVSOC.from_chemistry("ssb")
        for soc in [0.1, 0.3, 0.5, 0.8, 0.95]:
            v = float(oc.ocv(soc))
            soc_back = float(oc.soc(v))
            assert abs(soc_back - soc) < 0.02

    def test_ssb_pack_builds(self):
        pack = bms.BatteryPack(bms.PackConfig(n_cells=4, chemistry="ssb"))
        assert pack.n_cells == 4
        v = pack.pack_voltage()
        assert 12.0 < v < 18.0   # 4 × SSB cells ~3.0–4.35 V each

