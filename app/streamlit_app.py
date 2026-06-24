"""
Streamlit dashboard for the BMS Digital Twin — v5.

Run with::

    streamlit run app/streamlit_app.py

New in v5
---------
* **Solid-State Battery (SSB)** — 7th chemistry with Li-metal anode, highest
  voltage (4.35 V), highest T_runaway (150 °C), and ultra-low self-discharge.
* **Range Predictor tab** — physics-based EV range estimation with:
  - Vehicle presets (compact / sedan / SUV / truck).
  - 5 drive-cycle route profiles (WLTP, city, highway, mixed, mountain).
  - Custom segment builder (distance, speed, grade, traffic).
  - Full weather coupling: temperature (HVAC + battery derating + Arrhenius
    cold penalty), headwind (aero drag), precipitation (rolling resistance),
    altitude (air density).
  - Energy breakdown chart (Traction / Stop-Go / HVAC / Accessories − Regen).
  - SOC-along-route curve.
  - Temperature sweep chart — range vs. −30 to 50 °C for any chemistry.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bms import (
    BatteryPack, PackConfig,
    ThermalModel, ThermalParameters,
    HybridFaultDetector,
    BMSSupervisor, SupervisorConfig,
    FaultInjector, FaultSpec, FaultMode,
    EKFEstimator,
    get_chemistry_props,
    generate_load_profile, generate_power_profile, generate_cccv_profile,
    load_nasa_like_dataset, generate_aging_profile,
    estimate_rul, estimate_rul_with_resistance,
    compute_dva, compute_ica, synthetic_discharge_for_dva,
    simulate_eis, compute_crate_map,
    ECMParameters,
    RangePredictor, VehicleParams, WeatherConditions, RouteSegment,
    ROUTE_PROFILES, VEHICLE_PRESETS, INDIA_CITY_ROUTES, INDIA_WEATHER,
    RangePrediction,
)
from bms._train_detector import generate_fault_training_data

# ── Design tokens ────────────────────────────────────────────────────────────
_P = {          # indigo / amber / emerald / rose / violet / pink / cyan
    "indigo":  "#6366F1",
    "amber":   "#F59E0B",
    "emerald": "#10B981",
    "rose":    "#EF4444",
    "violet":  "#8B5CF6",
    "pink":    "#EC4899",
    "cyan":    "#06B6D4",
    "slate":   "#64748B",
}
_PALETTE     = list(_P.values())
_CELL_COLORS = _PALETTE
_FONT        = "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif"
_GRID_COLOR  = "#E2E8F0"
_PLOT_BG     = "#F8FAFD"
_HOVER_BG    = "#1E293B"

_FAULT_COLORS = {
    "overcharge":      _P["rose"],
    "short_circuit":   _P["amber"],
    "thermal_runaway": _P["violet"],
    "sensor_dropout":  _P["indigo"],
    "sensor_bias":     _P["pink"],
    "none":            _P["emerald"],
}

# ── Global Plotly theme ───────────────────────────────────────────────────────
pio.templates["bms"] = go.layout.Template(dict(layout=dict(
    font=dict(family=_FONT, size=12, color="#374151"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=_PLOT_BG,
    colorway=_PALETTE,
    xaxis=dict(gridcolor=_GRID_COLOR, linecolor="#CBD5E1", zeroline=False,
               tickfont=dict(size=11), title_font=dict(size=12, color="#6B7280")),
    yaxis=dict(gridcolor=_GRID_COLOR, linecolor="#CBD5E1", zeroline=False,
               tickfont=dict(size=11), title_font=dict(size=12, color="#6B7280")),
    hoverlabel=dict(bgcolor=_HOVER_BG, font=dict(color="#F1F5F9", size=12),
                    bordercolor="rgba(0,0,0,0)"),
    legend=dict(bgcolor="rgba(255,255,255,0.88)", bordercolor=_GRID_COLOR,
                borderwidth=1, font=dict(size=11)),
    margin=dict(l=52, r=20, t=46, b=46),
    title=dict(font=dict(size=13, color="#1E293B"), x=0.0, xanchor="left"),
)))
pio.templates.default = "plotly_white+bms"
_CHEM_LABELS = {
    "nmc":  "NMC — LiNiMnCoO₂ (18650)",
    "lfp":  "LFP — LiFePO₄ (prismatic)",
    "lmfp": "LMFP — LiMnFePO₄ (dual-plateau)",
    "lto":  "LTO — Li₄Ti₅O₁₂ (ultra-safe)",
    "nca":  "NCA — LiNiCoAlO₂ (high energy)",
    "lmo":  "LMO — LiMn₂O₄ (spinel)",
    "ssb":  "SSB — Solid-State (Li-metal, 4.35 V)",
}


# ── Cached resources ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training fault detector…")
def trained_detector(chemistry: str = "nmc") -> HybridFaultDetector:
    X, y = generate_fault_training_data(samples_per_class=200, seed=0,
                                         chemistry=chemistry)
    det = HybridFaultDetector(n_trees=60, chemistry=chemistry)
    det.fit(X, y)
    return det


def build_supervisor(n_cells: int, n_parallel: int, seed: int,
                     chemistry: str) -> BMSSupervisor:
    props = get_chemistry_props(chemistry)
    cap_Ah = props["default_capacity_Ah"]
    pack = BatteryPack(PackConfig(
        n_cells=n_cells, n_parallel=n_parallel,
        chemistry=chemistry, seed=seed,
        initial_soc_sigma=0.06,
        nominal_capacity_Ah=cap_Ah,
    ))
    thermal = ThermalModel(n_cells=n_cells, params=ThermalParameters())
    detector = trained_detector(chemistry)
    return BMSSupervisor(pack, thermal, detector,
                         config=SupervisorConfig(T_setpoint_C=35.0))


# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="BMS Digital Twin v5",
                   page_icon="🔋", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── BMS Digital Twin v5 — Premium UI ─────────────────────────────────────── */
:root{
  --c-indigo:#6366F1;--c-indigo-d:#4338CA;--c-indigo-l:#818CF8;
  --c-emerald:#10B981;--c-amber:#F59E0B;--c-rose:#EF4444;
  --c-violet:#8B5CF6;--c-cyan:#06B6D4;--c-slate:#64748B;
  --sh-xs:0 1px 3px rgba(15,23,42,.05);
  --sh-sm:0 2px 10px rgba(15,23,42,.07),0 4px 20px rgba(15,23,42,.04);
  --sh-md:0 6px 24px rgba(15,23,42,.09),0 12px 40px rgba(15,23,42,.05);
  --sh-glow:0 4px 22px rgba(99,102,241,.20),0 8px 44px rgba(99,102,241,.10);
  --r-sm:8px;--r-md:12px;--r-lg:16px;
  --ease:cubic-bezier(.4,0,.2,1);
}
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.65;transform:scale(1.22)}}
@keyframes shimmer{0%{background-position:-400% 0}100%{background-position:400% 0}}
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
@keyframes glowPulse{0%,100%{box-shadow:0 0 0 0 rgba(99,102,241,.42)}50%{box-shadow:0 0 0 9px rgba(99,102,241,0)}}

/* ── Background ──────────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"]>.main{
  background:
    radial-gradient(ellipse 70% 50% at 10% 5%,rgba(99,102,241,.08) 0%,transparent 55%),
    radial-gradient(ellipse 60% 45% at 88% 92%,rgba(16,185,129,.07) 0%,transparent 55%),
    linear-gradient(155deg,#EEF2FF 0%,#F8FAFF 48%,#F0FDF8 100%);
  min-height:100vh;
}
[data-testid="stHeader"]{background:transparent!important;backdrop-filter:none!important}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"]{
  background:
    radial-gradient(ellipse 90% 35% at 50% 0%,rgba(99,102,241,.24) 0%,transparent 55%),
    linear-gradient(180deg,#0B1437 0%,#111C4A 55%,#0D2040 100%)!important;
  border-right:1px solid rgba(99,102,241,.22)!important;
  box-shadow:4px 0 32px rgba(0,0,0,.26)!important;
}
section[data-testid="stSidebar"]>div:first-child{padding-top:1.25rem!important}
section[data-testid="stSidebar"] *{color:#B8C8E4!important}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{
  color:#A5B4FC!important;font-size:.72rem!important;font-weight:700!important;
  text-transform:uppercase!important;letter-spacing:.10em!important;
  padding-bottom:.45rem!important;margin-top:1.4rem!important;margin-bottom:.8rem!important;
  border-bottom:1px solid rgba(165,180,252,.14)!important;
}
section[data-testid="stSidebar"] hr{border:none!important;border-top:1px solid rgba(165,180,252,.10)!important;margin:.6rem 0!important}
section[data-testid="stSidebar"] label{color:#6E84A4!important;font-size:.72rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:.07em!important}
section[data-testid="stSidebar"] .stButton>button{
  background:linear-gradient(135deg,#3730A3 0%,#4F46E5 45%,#6366F1 100%)!important;
  color:#fff!important;border:none!important;border-radius:13px!important;
  font-weight:700!important;font-size:.94rem!important;letter-spacing:.03em!important;
  padding:.75rem 1rem!important;width:100%!important;margin-top:.55rem!important;
  box-shadow:0 4px 18px rgba(99,102,241,.45),inset 0 1px 0 rgba(255,255,255,.18)!important;
  transition:all .22s var(--ease)!important;
}
section[data-testid="stSidebar"] .stButton>button:hover{
  transform:translateY(-2px) scale(1.01)!important;
  box-shadow:0 8px 30px rgba(99,102,241,.55),inset 0 1px 0 rgba(255,255,255,.22)!important;
  animation:glowPulse 1.5s infinite!important;
}
section[data-testid="stSidebar"] .stButton>button:active{transform:translateY(0) scale(.99)!important}

/* ── Metric cards ────────────────────────────────────────────────────────── */
[data-testid="stMetric"]{
  background:linear-gradient(145deg,rgba(255,255,255,.97) 0%,rgba(248,250,255,.93) 100%)!important;
  backdrop-filter:blur(20px) saturate(180%)!important;
  -webkit-backdrop-filter:blur(20px) saturate(180%)!important;
  border:1px solid rgba(255,255,255,.90)!important;
  border-top:3px solid var(--c-indigo)!important;
  border-radius:var(--r-lg)!important;
  padding:16px 20px 14px!important;
  box-shadow:var(--sh-sm)!important;
  transition:all .25s var(--ease)!important;
  animation:fadeUp .4s ease both!important;
}
[data-testid="stMetric"]:nth-child(1){border-top-color:#6366F1!important;animation-delay:.05s!important}
[data-testid="stMetric"]:nth-child(2){border-top-color:#EF4444!important;animation-delay:.10s!important}
[data-testid="stMetric"]:nth-child(3){border-top-color:#10B981!important;animation-delay:.15s!important}
[data-testid="stMetric"]:nth-child(4){border-top-color:#F59E0B!important;animation-delay:.20s!important}
[data-testid="stMetric"]:nth-child(5){border-top-color:#8B5CF6!important;animation-delay:.25s!important}
[data-testid="stMetric"]:nth-child(6){border-top-color:#06B6D4!important;animation-delay:.30s!important}
[data-testid="stMetric"]:hover{transform:translateY(-3px)!important;box-shadow:var(--sh-glow)!important}
[data-testid="stMetricLabel"]>div{font-size:.67rem!important;font-weight:700!important;text-transform:uppercase!important;letter-spacing:.09em!important;color:#64748B!important}
[data-testid="stMetricValue"]>div{font-size:1.60rem!important;font-weight:800!important;letter-spacing:-.025em!important;color:#0F172A!important;line-height:1.1!important}
[data-testid="stMetricDelta"]>div{font-size:.77rem!important;font-weight:600!important}

/* ── Tab bar ─────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{
  background:rgba(255,255,255,.84)!important;
  backdrop-filter:blur(14px)!important;-webkit-backdrop-filter:blur(14px)!important;
  border-radius:15px!important;padding:5px 6px!important;gap:3px!important;
  border:1px solid rgba(99,102,241,.09)!important;
  box-shadow:0 2px 16px rgba(15,23,42,.05)!important;
}
.stTabs [data-baseweb="tab"]{
  border-radius:10px!important;font-weight:500!important;color:#475569!important;
  background:transparent!important;padding:8px 18px!important;font-size:.875rem!important;
  transition:all .18s var(--ease)!important;white-space:nowrap!important;border:none!important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]){background:rgba(99,102,241,.07)!important;color:#4338CA!important}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,#4338CA 0%,#6366F1 60%,#818CF8 100%)!important;
  color:#fff!important;font-weight:700!important;
  box-shadow:0 4px 16px rgba(99,102,241,.40)!important;
}

/* ── Chart wrappers ──────────────────────────────────────────────────────── */
[data-testid="stPlotlyChart"]>div{
  border-radius:var(--r-lg)!important;overflow:hidden!important;
  box-shadow:var(--sh-xs),0 0 0 1px rgba(226,232,240,.65)!important;
  background:#fff!important;transition:box-shadow .25s!important;
}
[data-testid="stPlotlyChart"]>div:hover{box-shadow:var(--sh-md),0 0 0 1px rgba(99,102,241,.12)!important}

/* ── Progress bar (shimmer) ──────────────────────────────────────────────── */
[data-testid="stProgressBar"]>div>div>div>div{
  background:linear-gradient(90deg,#4338CA 0%,#6366F1 40%,#818CF8 70%,#06B6D4 100%)!important;
  background-size:200% 100%!important;animation:shimmer 1.8s ease infinite!important;
  border-radius:6px!important;
}

/* ── Expanders ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"]{border:1px solid rgba(99,102,241,.10)!important;border-radius:var(--r-md)!important;overflow:hidden!important;box-shadow:var(--sh-xs)!important}
[data-testid="stExpander"]>details>summary{background:rgba(99,102,241,.035)!important;border-radius:var(--r-md)!important;font-weight:600!important;font-size:.875rem!important;color:#374151!important;padding:.68rem 1rem!important;transition:background .15s!important}
[data-testid="stExpander"]>details>summary:hover{background:rgba(99,102,241,.07)!important}
[data-testid="stExpander"]>details[open]>summary{color:#4338CA!important;background:rgba(99,102,241,.07)!important}

/* ── Alerts ──────────────────────────────────────────────────────────────── */
.stAlert{border-radius:12px!important}

/* ── Download button ─────────────────────────────────────────────────────── */
.stDownloadButton>button{border-radius:10px!important;border:1.5px solid rgba(99,102,241,.30)!important;color:#4338CA!important;font-weight:600!important;font-size:.84rem!important;background:rgba(99,102,241,.04)!important;transition:all .18s!important}
.stDownloadButton>button:hover{background:#EEF2FF!important;border-color:#6366F1!important;transform:translateY(-1px)!important;box-shadow:0 4px 14px rgba(99,102,241,.18)!important}

/* ── HR dividers ─────────────────────────────────────────────────────────── */
hr{border:none!important;height:1px!important;background:linear-gradient(90deg,transparent,rgba(99,102,241,.22) 20%,rgba(16,185,129,.18) 50%,rgba(99,102,241,.22) 80%,transparent)!important;margin:1.25rem 0!important}

/* ── Typography ──────────────────────────────────────────────────────────── */
h1{letter-spacing:-.025em!important}
h2{font-size:1.12rem!important;font-weight:700!important;color:#0F172A!important;letter-spacing:-.01em!important}
h3{font-size:.96rem!important;font-weight:700!important;color:#1E293B!important}
.stCaption p{color:#64748B!important;font-size:.80rem!important;line-height:1.5!important}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
[data-testid="stSelectbox"]>div>div,[data-testid="stNumberInput"]>div{border-radius:var(--r-sm)!important;transition:border-color .15s,box-shadow .15s!important}
[data-testid="stSelectbox"]>div>div:focus-within,[data-testid="stNumberInput"]>div:focus-within{border-color:var(--c-indigo)!important;box-shadow:0 0 0 3px rgba(99,102,241,.12)!important}

/* ── DataFrame ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"]{border-radius:var(--r-md)!important;overflow:hidden!important;box-shadow:var(--sh-xs)!important}

/* ── Live dot ────────────────────────────────────────────────────────────── */
.live-dot{display:inline-block;width:9px;height:9px;border-radius:50%;
  background:#10B981;animation:pulse-dot 1.8s ease-in-out infinite;
  margin-right:7px;vertical-align:middle;box-shadow:0 0 8px rgba(16,185,129,.55)}

/* ── Feature card (landing) ──────────────────────────────────────────────── */
.feat-card{background:#fff;border:1px solid rgba(0,0,0,.055);border-radius:16px;
  padding:20px 18px;box-shadow:0 2px 8px rgba(15,23,42,.05);
  transition:transform .22s var(--ease),box-shadow .22s var(--ease)}
.feat-card:hover{transform:translateY(-3px);box-shadow:0 8px 28px rgba(15,23,42,.09)}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;align-items:center;gap:14px;padding:4px 0 2px">
  <div>
    <h1 style="
      background:linear-gradient(95deg,#4338CA 0%,#6366F1 35%,#8B5CF6 65%,#06B6D4 100%);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
      font-size:2.0rem;font-weight:900;margin:0;line-height:1.1;letter-spacing:-.03em">
      BMS Digital Twin
    </h1>
    <div style="font-size:.78rem;color:#64748B;margin-top:3px;display:flex;align-items:center;gap:8px;flex-wrap:wrap">
      <span><span class="live-dot"></span><strong style="color:#059669">READY</strong></span>
      <span style="color:#CBD5E1">|</span>
      <span style="background:rgba(99,102,241,.09);color:#4338CA;padding:2px 9px;border-radius:999px;font-weight:700;font-size:.70rem;border:1px solid rgba(99,102,241,.18)">v5</span>
      <span style="color:#CBD5E1">|</span>
      <span>🔬 7 Chemistries</span>
      <span>📐 nS×mP Topology</span>
      <span>📡 EKF ±σ</span>
      <span>🚗 EV Range Predictor</span>
      <span>🇮🇳 India Mode</span>
      <span>🛵 Two-Wheeler</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Pack")
    chemistry = st.selectbox(
        "Cell chemistry",
        list(_CHEM_LABELS.keys()), index=0,
        format_func=lambda x: _CHEM_LABELS[x],
    )
    n_cells = st.slider("Series groups (nS)", 2, 20, 4, 1,
                         help="Number of series-connected cell groups")
    n_parallel = st.slider("Parallel cells per group (mP)", 1, 20, 1, 1,
                            help="Total cells = nS × mP")
    seed = st.number_input("Random seed", value=42, step=1)

    _props = get_chemistry_props(chemistry)
    _total_cells = n_cells * n_parallel
    _pack_V  = n_cells * _props["nominal_voltage_V"]
    _pack_Ah = n_parallel * _props["default_capacity_Ah"]
    _pack_kWh = _pack_V * _pack_Ah / 1000.0

    # Pack topology summary card
    st.markdown(f"""
    <div style="background:rgba(99,102,241,.07);border:1px solid rgba(99,102,241,.18);
                border-radius:10px;padding:10px 13px;margin:4px 0">
      <div style="font-size:.65rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:.09em;color:#A5B4FC;margin-bottom:6px">Pack Summary</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px">
        <div style="font-size:.78rem;color:#CBD5E1"><span style="color:#fff;font-weight:700">{n_cells}S × {n_parallel}P</span>&nbsp;topology</div>
        <div style="font-size:.78rem;color:#CBD5E1"><span style="color:#fff;font-weight:700">{_total_cells}</span>&nbsp;cells total</div>
        <div style="font-size:.78rem;color:#CBD5E1"><span style="color:#A5B4FC;font-weight:700">{_pack_V:.1f} V</span>&nbsp;nominal</div>
        <div style="font-size:.78rem;color:#CBD5E1"><span style="color:#A5B4FC;font-weight:700">{_pack_Ah:.1f} Ah</span>&nbsp;capacity</div>
        <div style="font-size:.78rem;color:#CBD5E1;grid-column:1/-1"><span style="color:#86EFAC;font-weight:700">{_pack_kWh:.2f} kWh</span>&nbsp;pack energy</div>
        <div style="font-size:.72rem;color:#64748B">V {_props['v_min']:.2f}–{_props['v_max']:.2f} V cell</div>
        <div style="font-size:.72rem;color:#64748B">T&#8331; {_props['T_runaway_C']:.0f} °C</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    if _total_cells > 100:
        st.warning(
            f"⚡ **{_total_cells}-cell pack** — simulation may take ~"
            f"{max(5, _total_cells // 40)}-{max(10, _total_cells // 20)} s. "
            "Large packs are fully supported."
        )

    st.header("Load profile")
    load_type = st.radio("Load type",
                         ["Current (C-rate)", "Power (W)", "CC-CV Charge"],
                         horizontal=False)
    mode = st.selectbox("Mode", ["drive", "pulse", "constant"], index=0,
                         disabled=(load_type == "CC-CV Charge"))
    duration_s = st.slider("Duration (s)", 60, 1800, 600, step=60,
                            disabled=(load_type == "CC-CV Charge"))
    if load_type == "Current (C-rate)":
        c_rate = st.slider("C-rate", 0.2, 3.0, 1.0, step=0.1)
        p_rate = None
        cccv_mode = False
    elif load_type == "Power (W)":
        p_rate = st.slider("Power rate", 0.2, 3.0, 1.0, step=0.1,
                           help="Multiplier × (capacity × nominal voltage)")
        c_rate = None
        cccv_mode = False
    else:   # CC-CV Charge
        c_rate = None
        p_rate = None
        cccv_mode = True
        st.caption("Simulates full CC-CV charge session (negative current = charge).")

    st.header("Fault injection")
    fault_mode = st.selectbox(
        "Mode", ["none", "overcharge", "short_circuit", "thermal_runaway",
                 "sensor_dropout", "sensor_bias"], index=0)
    fault_cell = st.number_input("Cell group index", min_value=0,
                                  max_value=int(n_cells) - 1, value=0)
    if not cccv_mode:
        fault_start = st.slider("Start step", 0, duration_s - 1,
                                 int(duration_s * 0.4))
        fault_end = st.slider("End step", fault_start, duration_s,
                               int(duration_s * 0.7))
    else:
        fault_start, fault_end = 0, 0
    fault_severity = st.slider("Severity", 0.1, 2.0, 1.0, step=0.1)

    st.header("Options")
    show_ekf = st.checkbox("Overlay EKF SOC estimate ± 1σ", value=True)
    aging_cycles = st.slider("Aging cycles (SoH tab)", 20, 200, 80, step=10)

    run_clicked = st.button("▶ Run simulation", type="primary",
                             use_container_width=True)


# ── Helper: chart builders ───────────────────────────────────────────────────
def plotly_lines(df: pd.DataFrame, title: str, yaxis_title: str,
                 colors: list[str] | None = None,
                 hline: float | None = None) -> go.Figure:
    fig = go.Figure()
    n_traces = len(df.columns)
    # For large packs: thin lines + reduced opacity, hide legend to keep chart clean
    lw = 1.2 if n_traces > 8 else 2.0
    opacity = max(0.35, 1.0 - n_traces * 0.025)
    show_legend = n_traces <= 12
    for i, col in enumerate(df.columns):
        c = (colors[i % len(colors)] if colors else _CELL_COLORS[i % len(_CELL_COLORS)])
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=col,
            line=dict(color=c, width=lw),
            opacity=opacity,
            showlegend=show_legend,
            hovertemplate=f"<b>{col}</b><br>t=%{{x:.0f}} s<br>"
                          f"{yaxis_title}=%{{y:.4f}}<extra></extra>",
        ))
    if n_traces > 8:
        # Add min/max envelope traces for readability at large cell counts
        fig.add_trace(go.Scatter(
            x=df.index, y=df.max(axis=1), name="max",
            line=dict(color=_P["rose"], width=1.8, dash="dot"),
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df.min(axis=1), name="min",
            line=dict(color=_P["emerald"], width=1.8, dash="dot"),
            fill="tonexty", fillcolor="rgba(99,102,241,0.04)",
            showlegend=True,
        ))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dot", line_color=_P["rose"],
                      line_width=1.5,
                      annotation_text=f"  {hline}",
                      annotation_font_color=_P["rose"],
                      annotation_font_size=11)
    legend_cfg = (dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                  if show_legend else dict(orientation="h", y=1.02, x=1, xanchor="right",
                                           font=dict(size=9)))
    fig.update_layout(title=title, yaxis_title=yaxis_title,
                      xaxis_title="time [s]", height=290,
                      legend=legend_cfg)
    return fig


def _hex_fill(hex_color: str, alpha: float = 0.09) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plotly_single(series: pd.Series, title: str, yaxis_title: str,
                  color: str = _P["emerald"]) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=series.index, y=series.values,
        line=dict(color=color, width=2.0),
        fill="tozeroy", fillcolor=_hex_fill(color),
        hovertemplate=f"{yaxis_title}=%{{y:.4f}}<extra></extra>",
    ))
    fig.update_layout(title=title, yaxis_title=yaxis_title,
                      xaxis_title="time [s]", height=290)
    return fig


# ── Simulation runner ─────────────────────────────────────────────────────────
def run_sim(n_cells: int, n_parallel: int, chemistry: str, seed: int,
            mode: str, duration_s: int, c_rate: float | None,
            p_rate: float | None, cccv_mode: bool,
            fault_mode: str, fault_cell: int,
            fault_start: int, fault_end: int, fault_severity: float,
            show_ekf: bool) -> dict:
    sup = build_supervisor(n_cells=int(n_cells), n_parallel=int(n_parallel),
                           seed=int(seed), chemistry=chemistry)
    inj = FaultInjector(chemistry=chemistry)
    if fault_mode != "none":
        inj.add(FaultSpec(mode=FaultMode(fault_mode),
                          start_step=fault_start, end_step=fault_end,
                          cell_index=int(fault_cell),
                          severity=float(fault_severity)))

    props = get_chemistry_props(chemistry)
    cap_Ah = props["default_capacity_Ah"]
    v_nom = props["nominal_voltage_V"]

    if cccv_mode:
        # Generate CC-CV profile (negative = charge)
        i_load = generate_cccv_profile(
            Q_nom_Ah=cap_Ah, soc_start=float(sup.pack.soc.mean()),
            chemistry=chemistry, i_charge_C=0.5,
        )
        p_load = None
    elif p_rate is not None:
        p_load = generate_power_profile(
            duration_s, dt=1.0, mode=mode, p_rate=float(p_rate),
            nominal_capacity_Ah=cap_Ah, nominal_voltage_V=v_nom,
            seed=int(seed),
        )
        i_load = None
    else:
        i_load = generate_load_profile(duration_s, dt=1.0, mode=mode,
                                        c_rate=float(c_rate), capacity_Ah=cap_Ah,
                                        seed=int(seed))
        p_load = None

    n = len(i_load) if i_load is not None else len(p_load)

    # EKF
    ekf = None
    if show_ekf:
        first_cell_params = sup.pack.groups[0].cells[0].params
        ekf = EKFEstimator(params=first_cell_params, ocv_curve=sup.pack.ocv_curve)
        ekf.reset(soc0=sup.pack.soc[0])

    soc_log = np.empty((n, sup.pack.n_cells))
    v_log = np.empty((n, sup.pack.n_cells))
    T_log = np.empty((n, sup.pack.n_cells))
    duty_log = np.empty(n)
    i_log = np.empty(n)
    p_log = np.empty(n)
    peak_p_log = np.empty(n)
    soe_log = np.empty(n)
    ekf_soc_log = np.full(n, np.nan)
    ekf_sigma_log = np.full(n, np.nan)
    state_log: list[str] = []
    fault_alerts: list[dict] = []
    bal_log: list[str] = []
    derate_log: list[bool] = []

    status_text = st.empty()
    progress = st.progress(0)

    for k in range(n):
        if fault_mode == "thermal_runaway" and k == fault_start:
            sup.thermal.T[int(fault_cell)] = props["T_runaway_C"] + 5.0
        if fault_mode == "thermal_runaway":
            sup.thermal.T[:] = inj.apply_to_temperatures(sup.thermal.T, k, 1.0)

        if p_load is not None:
            req_power = float(p_load[k])
            cell_extra = inj.apply_to_currents(np.zeros(sup.pack.n_cells), k)
            if cell_extra.any():
                sup.pack.step(0.0, 1.0, balancing_currents=cell_extra,
                               cell_temperatures_C=sup.thermal.T)
            out = sup.step(requested_power_W=req_power, dt=1.0, k=k)
            i_log[k] = out["cmd_current"]
        else:
            req_current = float(i_load[k])
            cell_currents = np.full(sup.pack.n_cells, req_current)
            cell_currents = inj.apply_to_currents(cell_currents, k)
            extra = cell_currents - req_current
            sup.pack.step(req_current, 1.0,
                          balancing_currents=extra,
                          cell_temperatures_C=sup.thermal.T)
            out = sup.step(req_current, 1.0, k=k)
            i_log[k] = req_current

        soc_log[k] = out["soc"]
        v_log[k] = out["v_cells"]
        T_log[k] = out["T_cells"]
        duty_log[k] = out["cooling_duty"]
        p_log[k] = out["power_W"]
        peak_p_log[k] = out["peak_power_W"]
        soe_log[k] = out.get("soe_Wh", 0.0)
        state_log.append(out["state"])
        bal_log.append(out["balancer"])
        derate_log.append(out.get("derated", False))

        if ekf is not None:
            v_meas = inj.apply_to_voltage_meas(out["v_cells"], k)
            ekf_soc_log[k] = ekf.update(float(i_log[k]), float(v_meas[0]),
                                          dt=1.0,
                                          temperature_C=float(out["T_cells"][0]))
            ekf_sigma_log[k] = ekf.soc_uncertainty_1sigma

        if out["fault_label"] != "none":
            fault_alerts.append({
                "step": k,
                "mode": out["fault_label"],
                "source": out["fault_source"],
                "T_max": float(out["T_cells"].max()),
                "V_min": float(out["v_cells"].min()),
            })

        if k % max(1, n // 40) == 0:
            progress.progress((k + 1) / n)
            status_text.text(
                f"Step {k+1}/{n} — SOC min={out['soc'].min():.3f} "
                f"max={out['soc'].max():.3f}  T_max={out['T_cells'].max():.1f}°C  "
                f"P={out['power_W']:.1f}W  SoE={out.get('soe_Wh',0):.1f}Wh  "
                f"state={out['state']}"
            )

    progress.progress(1.0)
    status_text.empty()

    return dict(
        soc=soc_log, v=v_log, T=T_log, duty=duty_log,
        i_load=i_log, power=p_log, peak_power=peak_p_log,
        soe=soe_log,
        ekf_soc=ekf_soc_log, ekf_sigma=ekf_sigma_log,
        state=state_log, bal=bal_log, derate=derate_log,
        fault_alerts=fault_alerts,
        n_cells=sup.pack.n_cells,
        chemistry=chemistry,
        n_parallel=n_parallel,
        passport=sup.passport.summary(),
        sup=sup,   # kept to access ECMParameters for diagnostics
    )


# ── Results renderer ──────────────────────────────────────────────────────────
def render_results(res: dict, show_ekf: bool, aging_cycles: int):
    chem = res["chemistry"]
    n_par = res["n_parallel"]
    cells = [f"group {i}" for i in range(res["n_cells"])]
    t = np.arange(len(res["duty"]))
    props = get_chemistry_props(chem)

    # ── KPI cards ──────────────────────────────────────────────────────────
    energy_Wh = float(np.trapezoid(np.abs(res["power"]), dx=1.0) / 3600)
    _total_cells = res["n_cells"] * n_par
    _pack_v_nom  = res["n_cells"] * props["nominal_voltage_V"]
    _pack_Ah_nom = n_par * props["default_capacity_Ah"]
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Min SOC", f"{res['soc'].min():.3f}",
              delta=f"{res['soc'].min() - res['soc'][0].mean():.3f}")
    k2.metric("Peak Temperature", f"{res['T'].max():.1f} °C")
    k3.metric("Energy Consumed",
              f"{energy_Wh/1000:.3f} kWh" if energy_Wh >= 1000 else f"{energy_Wh:.1f} Wh")
    k4.metric("Peak Power",
              f"{res['peak_power'].max()/1000:.2f} kW" if res['peak_power'].max() >= 1000
              else f"{res['peak_power'].max():.0f} W")
    k5.metric("Pack Config",
              f"{res['n_cells']}S×{n_par}P",
              delta=f"{_total_cells} cells · {_pack_v_nom:.0f} V · {_pack_Ah_nom:.1f} Ah")
    k6.metric("Fault Alarms", str(len(res["fault_alerts"])),
              delta=None if not res["fault_alerts"] else "⚠",
              delta_color="inverse")

    _chem_color = {
        "nmc": "#6366F1", "lfp": "#10B981", "lmfp": "#06B6D4",
        "lto": "#8B5CF6", "nca": "#F59E0B", "lmo": "#EF4444", "ssb": "#EC4899",
    }.get(chem, "#6366F1")
    st.markdown(f"""
    <div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;
                background:rgba(255,255,255,.75);backdrop-filter:blur(12px);
                border:1px solid rgba(226,232,240,.8);border-left:4px solid {_chem_color};
                border-radius:12px;padding:10px 16px;margin:6px 0">
      <span style="background:{_chem_color}18;color:{_chem_color};
                   padding:3px 10px;border-radius:999px;font-weight:800;
                   font-size:.72rem;border:1px solid {_chem_color}33;letter-spacing:.05em">
        {chem.upper()}
      </span>
      <span style="font-size:.82rem;color:#374151;font-weight:600">{_CHEM_LABELS[chem]}</span>
      <span style="color:#CBD5E1;font-size:.8rem">|</span>
      <span style="font-size:.80rem;color:#64748B">Topology <strong style="color:#0F172A">{res['n_cells']}S×{n_par}P</strong></span>
      <span style="font-size:.80rem;color:#64748B">V_oc <strong style="color:#0F172A">{props['v_overcharge']:.2f} V</strong></span>
      <span style="font-size:.80rem;color:#64748B">T_runaway <strong style="color:#EF4444">{props['T_runaway_C']:.0f} °C</strong></span>
      <span style="font-size:.80rem;color:#64748B">Self-disch <strong style="color:#0F172A">{props['self_discharge_pct_per_month']:.1f} %/mo</strong></span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Live Signals",
        "🔬 SoH & Aging",
        "📋 Battery Passport",
        "🔭 Diagnostics",
        "⚠ Fault Analysis",
    ])

    # ── Tab 1: Live Signals ────────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            soc_df = pd.DataFrame(res["soc"], columns=cells, index=t)
            fig_soc = plotly_lines(soc_df, "Per-group SOC", "SOC [-]")
            if show_ekf and not np.all(np.isnan(res["ekf_soc"])):
                fig_soc.add_trace(go.Scatter(
                    x=t, y=res["ekf_soc"], name="EKF (group 0)",
                    line=dict(color=_P["slate"], dash="dash", width=1.8),
                ))
                valid = ~np.isnan(res["ekf_sigma"])
                if valid.any():
                    upper = res["ekf_soc"] + res["ekf_sigma"]
                    lower = res["ekf_soc"] - res["ekf_sigma"]
                    fig_soc.add_trace(go.Scatter(
                        x=np.concatenate([t[valid], t[valid][::-1]]),
                        y=np.concatenate([upper[valid], lower[valid][::-1]]),
                        fill="toself", fillcolor="rgba(99,102,241,0.12)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="EKF ±1σ", showlegend=True,
                    ))
            st.plotly_chart(fig_soc, use_container_width=True)

        with c2:
            v_df = pd.DataFrame(res["v"], columns=cells, index=t)
            st.plotly_chart(
                plotly_lines(v_df, "Per-group terminal voltage", "V [V]",
                             hline=props["v_overcharge"]),
                use_container_width=True,
            )

        c3, c4 = st.columns(2)
        with c3:
            T_df = pd.DataFrame(res["T"], columns=cells, index=t)
            st.plotly_chart(
                plotly_lines(T_df, "Per-group temperature", "T [°C]",
                             hline=props["T_runaway_C"]),
                use_container_width=True,
            )
        with c4:
            fig_pwr = go.Figure()
            fig_pwr.add_trace(go.Scatter(x=t, y=res["power"],
                                          name="instantaneous [W]",
                                          line=dict(color=_P["indigo"])))
            fig_pwr.add_trace(go.Scatter(x=t, y=res["peak_power"],
                                          name="peak capability [W]",
                                          line=dict(color=_P["rose"], dash="dot")))
            fig_pwr.update_layout(title="Pack power", yaxis_title="Power [W]",
                                   xaxis_title="time [s]", height=280,
                                   margin=dict(l=40, r=20, t=40, b=40),
                                   legend=dict(orientation="h", yanchor="bottom",
                                               y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_pwr, use_container_width=True)

        c5, c6 = st.columns(2)
        with c5:
            fig_soe = go.Figure()
            fig_soe.add_trace(go.Scatter(x=t, y=res["soe"],
                                          name="SoE [Wh]",
                                          fill="tozeroy",
                                          line=dict(color=_P["emerald"])))
            fig_soe.update_layout(title="State of Energy", yaxis_title="Energy [Wh]",
                                   xaxis_title="time [s]", height=280,
                                   margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_soe, use_container_width=True)

        with c6:
            fig_duty = make_subplots(specs=[[{"secondary_y": True}]])
            fig_duty.add_trace(go.Scatter(x=t, y=res["duty"],
                                           name="cooling duty",
                                           line=dict(color=_P["emerald"])))
            fig_duty.add_trace(go.Scatter(x=t, y=res["i_load"],
                                           name="load current [A]",
                                           line=dict(color=_P["indigo"], dash="dot")),
                               secondary_y=True)
            fig_duty.update_layout(title="Cooling duty & load current", height=280,
                                    margin=dict(l=40, r=40, t=40, b=40))
            fig_duty.update_yaxes(title_text="duty [0-1]", secondary_y=False)
            fig_duty.update_yaxes(title_text="current [A]", secondary_y=True)
            st.plotly_chart(fig_duty, use_container_width=True)

        # ── Supervisor state table ─────────────────────────────────────────
        sb = pd.DataFrame(
            {"state": res["state"], "balancer": res["bal"], "derated": res["derate"]},
            index=t,
        )
        def _color_state(val):
            cmap = {"fault": "background-color:#ffcccc",
                    "balancing": "background-color:#fff3cd",
                    "operating": "background-color:#d4edda"}
            return cmap.get(val, "")
        st.subheader("Supervisor state (last 20 steps)")
        st.dataframe(sb.tail(20).style.map(_color_state, subset=["state"]),
                     use_container_width=True)

        # ── CSV download ───────────────────────────────────────────────────
        full_df = pd.DataFrame(res["soc"], columns=[f"soc_{c}" for c in cells])
        for i, c in enumerate(cells):
            full_df[f"v_{c}"] = res["v"][:, i]
            full_df[f"T_{c}"] = res["T"][:, i]
        full_df["cooling_duty"] = res["duty"]
        full_df["i_load_A"] = res["i_load"]
        full_df["power_W"] = res["power"]
        full_df["peak_power_W"] = res["peak_power"]
        full_df["soe_Wh"] = res["soe"]
        full_df["state"] = res["state"]
        full_df["balancer"] = res["bal"]
        if show_ekf:
            full_df["ekf_soc_group0"] = res["ekf_soc"]
            full_df["ekf_sigma_group0"] = res["ekf_sigma"]
        csv_buf = io.StringIO()
        full_df.to_csv(csv_buf, index_label="step")
        st.download_button("⬇ Download simulation log (CSV)",
                           data=csv_buf.getvalue(),
                           file_name="bms_simulation_v4.csv",
                           mime="text/csv")

    # ── Tab 2: SoH & Aging ────────────────────────────────────────────────
    with tab2:
        cap_nom = props["default_capacity_Ah"]
        r0_nom = props["default_ecm"]["R0"] * 1000
        st.subheader(f"Capacity fade + resistance growth — {chem.upper()}")

        aging_df = load_nasa_like_dataset(cycles=aging_cycles,
                                          capacity_Ah=cap_nom, seed=int(seed))
        multi_df = generate_aging_profile(cycles=aging_cycles,
                                          capacity_Ah=cap_nom, r0_mOhm=r0_nom,
                                          seed=int(seed))

        rul_dual = estimate_rul_with_resistance(
            aging_df["cycle"].values, aging_df["capacity_Ah"].values,
            aging_df["resistance_mOhm"].values,
            nominal_capacity_Ah=cap_nom, nominal_resistance_mOhm=r0_nom,
        )
        rul_single = estimate_rul(aging_df["cycle"].values,
                                  aging_df["capacity_Ah"].values,
                                  nominal_capacity_Ah=cap_nom)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("SoH (capacity)", f"{rul_single['soh']*100:.1f}%")
        s2.metric("SoH (resistance)", f"{rul_dual['soh_resistance']*100:.1f}%")
        s3.metric("RUL (cycles)", f"{rul_dual['rul_cycles']:.0f}",
                  delta=f"limited by {rul_dual['limiting_mode']}")
        s4.metric("Cycles to cap EoL", f"{rul_dual['cycles_to_cap_eol']:.0f}")

        a1, a2 = st.columns(2)
        with a1:
            N_fit = np.linspace(1, rul_dual["cycles_to_cap_eol"] * 1.1, 300)
            cap_fit = cap_nom * (1 - rul_dual["alpha"] * np.sqrt(N_fit))
            fig_cap = go.Figure()
            fig_cap.add_trace(go.Scatter(x=aging_df["cycle"],
                                          y=aging_df["capacity_Ah"],
                                          mode="markers", name="observed",
                                          marker=dict(size=5, color=_P["indigo"])))
            fig_cap.add_trace(go.Scatter(x=N_fit, y=cap_fit,
                                          name=f"fit α={rul_dual['alpha']:.4f}",
                                          line=dict(color=_P["rose"])))
            fig_cap.add_hline(y=0.8 * cap_nom, line_dash="dash",
                               annotation_text="80% EoL")
            fig_cap.update_layout(title="Capacity fade", yaxis_title="Capacity [Ah]",
                                   xaxis_title="Cycle", height=300,
                                   margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_cap, use_container_width=True)

        with a2:
            N_fit_r = np.linspace(1, rul_dual["cycles_to_res_eol"] * 1.1, 300)
            res_fit = r0_nom * (1 + rul_dual["beta"] * np.sqrt(N_fit_r))
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=aging_df["cycle"],
                                          y=aging_df["resistance_mOhm"],
                                          mode="markers", name="observed",
                                          marker=dict(size=5, color=_P["amber"])))
            fig_res.add_trace(go.Scatter(x=N_fit_r, y=res_fit,
                                          name=f"fit β={rul_dual['beta']:.4f}",
                                          line=dict(color=_P["violet"])))
            fig_res.add_hline(y=r0_nom * 1.5, line_dash="dash",
                               annotation_text="150% EoL")
            fig_res.update_layout(title="Resistance growth", yaxis_title="Resistance [mΩ]",
                                   xaxis_title="Cycle", height=300,
                                   margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_res, use_container_width=True)

        st.subheader("DVA / ICA fingerprint")
        q_dva, v_dva = synthetic_discharge_for_dva(chem, n_points=500)
        q_ax, dva_curve = compute_dva(q_dva, v_dva)
        v_ax, ica_curve = compute_ica(q_dva, v_dva)

        d1, d2 = st.columns(2)
        with d1:
            fig_dva = go.Figure()
            fig_dva.add_trace(go.Scatter(x=q_ax, y=dva_curve,
                                          line=dict(color=_P["indigo"]),
                                          name="dV/dQ"))
            fig_dva.update_layout(title=f"DVA — dV/dQ ({chem.upper()})",
                                   xaxis_title="Discharge capacity [Ah]",
                                   yaxis_title="dV/dQ [V/Ah]", height=280,
                                   margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_dva, use_container_width=True)
        with d2:
            fig_ica = go.Figure()
            fig_ica.add_trace(go.Scatter(x=v_ax, y=ica_curve,
                                          fill="tozeroy",
                                          line=dict(color=_P["amber"]),
                                          name="dQ/dV"))
            fig_ica.update_layout(title=f"ICA — dQ/dV ({chem.upper()})",
                                   xaxis_title="Voltage [V]",
                                   yaxis_title="dQ/dV [Ah/V]", height=280,
                                   margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_ica, use_container_width=True)

        st.subheader("Multi-temperature aging profile")
        mt1, mt2 = st.columns(2)
        with mt1:
            fig_mt = go.Figure()
            for T_val in sorted(multi_df["temperature_C"].unique()):
                sub = multi_df[multi_df["temperature_C"] == T_val]
                fig_mt.add_trace(go.Scatter(x=sub["cycle"], y=sub["capacity_Ah"],
                                             mode="lines+markers",
                                             name=f"{T_val:.0f}°C",
                                             marker=dict(size=4)))
            fig_mt.update_layout(title="Capacity vs cycle (multi-temp)",
                                  yaxis_title="Capacity [Ah]", height=280,
                                  margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_mt, use_container_width=True)
        with mt2:
            fig_soh = go.Figure()
            fig_soh.add_trace(go.Scatter(x=multi_df["cycle"],
                                          y=multi_df["soh_capacity"] * 100,
                                          name="SoH capacity", fill="tozeroy",
                                          line=dict(color=_P["emerald"])))
            fig_soh.add_trace(go.Scatter(x=multi_df["cycle"],
                                          y=multi_df["soh_resistance"] * 100,
                                          name="SoH resistance",
                                          line=dict(color=_P["amber"], dash="dash")))
            fig_soh.add_hline(y=80, line_dash="dot", line_color=_P["rose"],
                               annotation_text="80% EoL")
            fig_soh.update_layout(title="State of Health trajectories",
                                   yaxis_title="SoH [%]", height=280,
                                   margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_soh, use_container_width=True)

    # ── Tab 3: Battery Passport ────────────────────────────────────────────
    with tab3:
        ps = res["passport"]
        st.subheader("Battery Passport — Lifetime Record")

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Equiv. Full Cycles (EFC)", f"{ps['equivalent_full_cycles']:.3f}")
        p2.metric("Depth-Weighted Cycles", f"{ps['depth_weighted_cycles']:.3f}")
        p3.metric("Round-Trip Efficiency", f"{ps['round_trip_efficiency']*100:.1f}%")
        p4.metric("Total Operating Time", f"{ps['total_time_h']:.3f} h")

        e1, e2 = st.columns(2)
        e1.metric("Energy Discharged", f"{ps['total_energy_out_Wh']:.3f} Wh")
        e2.metric("Energy Charged", f"{ps['total_energy_in_Wh']:.3f} Wh")

        st.caption(
            f"Chemistry: **{ps['chemistry'].upper()}**  •  "
            f"Nominal pack capacity: **{ps['nominal_capacity_Ah']:.2f} Ah**  •  "
            f"Self-discharge rate: **{props['self_discharge_pct_per_month']:.1f} %/month**"
        )
        st.info(
            "**EFC** (Equivalent Full Cycles) = total discharge Ah / nominal capacity.  \n"
            "**DWC** (Depth-Weighted Cycles) = Σ|ΔSOC| / 2 — rain-flow half-cycle count.  \n"
            "**RTE** = energy out / energy in — round-trip efficiency over this session."
        )

        # Passport as a table for download
        ps_df = pd.DataFrame([ps])
        st.dataframe(ps_df, use_container_width=True)
        csv_ps = io.StringIO()
        ps_df.to_csv(csv_ps, index=False)
        st.download_button("⬇ Download passport (CSV)",
                           data=csv_ps.getvalue(),
                           file_name="battery_passport.csv",
                           mime="text/csv")

    # ── Tab 4: Diagnostics ─────────────────────────────────────────────────
    with tab4:
        st.subheader(f"EIS Nyquist — {chem.upper()} ECM simulation")

        # Build params from chemistry defaults for display
        diag_params = ECMParameters(**{
            "R0": props["default_ecm"]["R0"],
            "R1": props["default_ecm"]["R1"],
            "C1": props["default_ecm"]["C1"],
            "R2": props["default_ecm"]["R2"],
            "C2": props["default_ecm"]["C2"],
            "Q_nom_Ah": props["default_capacity_Ah"],
            "chemistry": chem,
        })

        diag_T = st.select_slider("EIS temperature [°C]",
                                   options=[-10, 0, 10, 25, 40, 60], value=25)
        freq_Hz, Z_re, Z_neg_im = simulate_eis(diag_params, temperature_C=float(diag_T))

        eis_col1, eis_col2 = st.columns(2)
        with eis_col1:
            fig_eis = go.Figure()
            fig_eis.add_trace(go.Scatter(x=Z_re, y=Z_neg_im,
                                          mode="lines+markers",
                                          marker=dict(size=3, color=_P["indigo"]),
                                          line=dict(color=_P["indigo"]),
                                          name="Nyquist"))
            fig_eis.add_annotation(x=diag_params.R0, y=0,
                                    text=f"R0={diag_params.R0*1000:.1f}mΩ",
                                    showarrow=True, arrowhead=2, ax=40, ay=-30)
            fig_eis.update_layout(
                title=f"EIS Nyquist — {chem.upper()} @ {diag_T}°C",
                xaxis_title="Z' (Re Z) [Ω]",
                yaxis_title="−Z'' (−Im Z) [Ω]",
                height=360, margin=dict(l=60, r=20, t=50, b=50),
                yaxis=dict(scaleanchor="x", scaleratio=1),
            )
            st.plotly_chart(fig_eis, use_container_width=True)

        with eis_col2:
            Z_mag = np.sqrt(Z_re ** 2 + Z_neg_im ** 2)
            Z_phase_deg = np.degrees(np.arctan2(-Z_neg_im, Z_re))
            fig_bode = go.Figure()
            fig_bode.add_trace(go.Scatter(
                x=freq_Hz, y=Z_mag, name="|Z| [Ω]",
                line=dict(color=_P["indigo"], width=2), yaxis="y"))
            fig_bode.add_trace(go.Scatter(
                x=freq_Hz, y=Z_phase_deg, name="Phase [°]",
                line=dict(color=_P["amber"], dash="dash", width=2), yaxis="y2"))
            fig_bode.update_layout(
                title=f"Bode plot — {chem.upper()} @ {diag_T}°C",
                xaxis=dict(title="Frequency [Hz]", type="log"),
                yaxis=dict(title=dict(text="|Z| [Ω]",
                                      font=dict(color=_P["indigo"]))),
                yaxis2=dict(title=dict(text="Phase [°]",
                                       font=dict(color=_P["amber"])),
                            overlaying="y", side="right"),
                height=360, margin=dict(l=60, r=60, t=50, b=50),
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig_bode, use_container_width=True)

        st.subheader(f"C-rate Capability Map — {chem.upper()}")
        st.caption("Max discharge C-rate limited by V_terminal ≥ V_min; "
                   "white = impossible at that condition.")

        soc_ax, T_ax, cmap = compute_crate_map(diag_params, chemistry=chem,
                                                soc_points=20, temp_points=15,
                                                T_min_C=-20.0, T_max_C=60.0)
        fig_cmap = go.Figure(go.Heatmap(
            x=T_ax, y=soc_ax, z=cmap,
            colorscale="RdYlGn", zsmooth="best",
            colorbar=dict(title="C-rate"),
        ))
        fig_cmap.update_layout(
            title=f"Max discharge C-rate — {chem.upper()}",
            xaxis_title="Temperature [°C]",
            yaxis_title="SOC [-]",
            height=380, margin=dict(l=60, r=20, t=50, b=50),
        )
        st.plotly_chart(fig_cmap, use_container_width=True)

    # ── Tab 5: Fault Analysis ──────────────────────────────────────────────
    with tab5:
        if res["fault_alerts"]:
            st.error(f"**{len(res['fault_alerts'])} fault alarm(s) raised**")
            fa_df = pd.DataFrame(res["fault_alerts"])

            def _color_fault(val):
                c = _FAULT_COLORS.get(val, "#cccccc")
                return f"background-color:{c}22; color:{c}"

            st.dataframe(fa_df.style.map(_color_fault, subset=["mode"]),
                         use_container_width=True)

            fig_fault = go.Figure()
            for mode_val in fa_df["mode"].unique():
                sub = fa_df[fa_df["mode"] == mode_val]
                fig_fault.add_trace(go.Scatter(
                    x=sub["step"], y=[mode_val] * len(sub),
                    mode="markers", name=mode_val,
                    marker=dict(size=8, color=_FAULT_COLORS.get(mode_val, "grey"),
                                symbol="x"),
                ))
            fig_fault.update_layout(title="Fault event timeline",
                                     xaxis_title="step", yaxis_title="fault mode",
                                     height=280,
                                     margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_fault, use_container_width=True)
        else:
            st.success("No fault alarms raised during this simulation.")

        from bms import build_fmea_table
        st.subheader("FMEA — Risk Priority Numbers")
        fmea_df = build_fmea_table()
        fig_fmea = go.Figure(go.Bar(
            x=fmea_df["RPN"], y=fmea_df["failure_mode"],
            orientation="h",
            marker_color=[
                _P["rose"] if rpn >= 200 else
                _P["amber"] if rpn >= 100 else _P["emerald"]
                for rpn in fmea_df["RPN"]
            ],
            marker_line_width=0,
        ))
        fig_fmea.update_layout(title="FMEA — failure modes ranked by RPN",
                                xaxis_title="RPN", height=360,
                                margin=dict(l=220, r=20, t=40, b=40),
                                yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fmea, use_container_width=True)
        st.dataframe(fmea_df[["failure_mode", "effect", "S", "O", "D", "RPN"]],
                     use_container_width=True)

    # (Range Predictor is rendered via render_range_predictor() at top-level)
    _rp_placeholder = None  # noqa: F841 — keeps diff clean


def render_range_predictor():
    st.subheader("EV Range Predictor")
    st.caption(
        "Physics-based range estimation: aero drag · grade force · rolling resistance · "
        "stop-and-go cycling · HVAC load · battery temperature derating (Arrhenius)."
    )

    # ── Mode toggles ────────────────────────────────────────────────────
    mode_col1, mode_col2, mode_col3 = st.columns([1, 1, 2])
    india_mode = mode_col1.toggle("🇮🇳 India Mode", value=False, key="rp_india")
    two_wheeler = mode_col2.toggle("🛵 Two-Wheeler", value=False, key="rp_2w")

    rp_left, rp_right = st.columns([1, 2])

    with rp_left:
        # ── Vehicle ────────────────────────────────────────────────────
        st.markdown("**Vehicle**")
        if two_wheeler:
            _2w_presets = {
                "e_scooter": "E-Scooter (Ola S1 / Ather 450X class)",
                "e_motorcycle": "E-Motorcycle (Revolt RV400 class)",
                "e_moped": "E-Moped (Hero Electric / Ampere class)",
            }
            vehicle_preset = st.selectbox(
                "Two-Wheeler Preset",
                list(_2w_presets.keys()) + ["custom"],
                index=0,
                format_func=lambda x: _2w_presets.get(x, "Custom"),
                key="rp_vehicle_2w",
            )
            st.info("No HVAC — cabin temperature control not applicable to 2-wheelers.")
        else:
            vehicle_preset = st.selectbox(
                "Preset", ["compact", "sedan", "suv", "truck", "custom"],
                index=1, key="rp_vehicle")

        if vehicle_preset == "custom":
            if two_wheeler:
                rp_mass = st.slider("Mass [kg]", 50, 300, 130, 5, key="rp_mass")
                rp_cd = st.slider("Drag Cd", 0.40, 1.0, 0.65, 0.01, key="rp_cd")
                rp_area = st.slider("Frontal area [m²]", 0.3, 0.9, 0.55, 0.05, key="rp_area")
                vehicle = VehicleParams(mass_kg=float(rp_mass),
                                        drag_coefficient=float(rp_cd),
                                        frontal_area_m2=float(rp_area),
                                        hvac_max_W=0.0,
                                        accessory_load_W=45.0)
            else:
                rp_mass = st.slider("Mass [kg]", 1000, 4000, 2000, 50, key="rp_mass")
                rp_cd = st.slider("Drag Cd", 0.15, 0.55, 0.28, 0.01, key="rp_cd")
                rp_area = st.slider("Frontal area [m²]", 1.5, 4.0, 2.4, 0.1, key="rp_area")
                rp_hvac = st.slider("Max HVAC [W]", 1000, 8000, 4000, 500, key="rp_hvac")
                vehicle = VehicleParams(mass_kg=float(rp_mass),
                                        drag_coefficient=float(rp_cd),
                                        frontal_area_m2=float(rp_area),
                                        hvac_max_W=float(rp_hvac))
        else:
            vehicle = VEHICLE_PRESETS[vehicle_preset]
            st.caption(
                f"Mass **{vehicle.mass_kg:.0f} kg** · "
                f"Cd **{vehicle.drag_coefficient:.2f}** · "
                f"A **{vehicle.frontal_area_m2:.1f} m²**"
                + (f" · HVAC **{vehicle.hvac_max_W/1000:.1f} kW**"
                   if vehicle.hvac_max_W > 0 else " · No HVAC")
            )

        # ── Battery ────────────────────────────────────────────────────
        st.markdown("**Battery**")
        rp_chem = st.selectbox(
            "Chemistry", list(_CHEM_LABELS.keys()), index=0,
            format_func=lambda x: _CHEM_LABELS[x], key="rp_chem")
        if two_wheeler:
            rp_kwh = st.slider("Pack energy [kWh]", 0.5, 15.0, 3.0, 0.5, key="rp_kwh")
        else:
            rp_kwh = st.slider("Pack energy [kWh]", 20.0, 160.0, 80.0, 5.0, key="rp_kwh")
        rp_soc = st.slider("Initial SOC", 0.20, 1.00, 1.00, 0.05, key="rp_soc")

        # ── Route ────────────────────────────────────────────────────
        st.markdown("**Route**")
        if india_mode:
            india_route_type = st.radio(
                "Route type", ["City preset", "Drive cycle", "Custom"],
                horizontal=True, key="rp_india_route_type")

            if india_route_type == "City preset":
                _city_labels = {
                    "delhi_ncr": "Delhi / NCR",
                    "mumbai": "Mumbai",
                    "bangalore": "Bangalore",
                    "chennai": "Chennai",
                    "pune": "Pune",
                    "hyderabad": "Hyderabad",
                    "kolkata": "Kolkata",
                }
                rp_city = st.selectbox(
                    "City", list(_city_labels.keys()),
                    format_func=lambda x: _city_labels[x], key="rp_city")
                route_segs = INDIA_CITY_ROUTES[rp_city]
                rp_profile = "city"
                total_km = sum(s.distance_km for s in route_segs)
                st.caption(
                    f"{len(route_segs)} segments · **{total_km:.0f} km** total · "
                    f"road quality: {', '.join(s.road_quality for s in route_segs)}"
                )

            elif india_route_type == "Drive cycle":
                rp_profile = st.selectbox(
                    "Drive cycle",
                    ["midc", "india_nh", "city", "mixed"],
                    format_func=lambda x: {
                        "midc": "MIDC — Modified Indian Drive Cycle",
                        "india_nh": "India National Highway",
                        "city": "City (generic)",
                        "mixed": "Mixed (urban + highway)",
                    }.get(x, x),
                    key="rp_profile_india")
                route_segs = ROUTE_PROFILES[rp_profile]

            else:  # Custom
                rp_profile = "city"
                st.caption("Add up to 5 custom segments:")
                custom_segs = []
                _rq_opts = ["excellent", "good", "average", "poor"]
                for i in range(5):
                    with st.expander(f"Segment {i+1}", expanded=(i == 0)):
                        seg_d = st.number_input(f"Distance [km]", 0.0, 200.0, 5.0,
                                                 key=f"rp_d{i}")
                        seg_v = st.number_input(f"Avg speed [km/h]", 5.0, 120.0, 30.0,
                                                 key=f"rp_v{i}")
                        seg_g = st.number_input(f"Grade [%]", -15.0, 15.0, 0.0,
                                                 key=f"rp_g{i}")
                        seg_t = st.slider(f"Traffic factor", 0.0, 1.0, 0.60,
                                           key=f"rp_t{i}")
                        seg_rq = st.selectbox(f"Road quality", _rq_opts, index=2,
                                               key=f"rp_rq{i}")
                        if seg_d > 0:
                            custom_segs.append(RouteSegment(
                                seg_d, seg_v, seg_g, seg_t,
                                f"Seg {i+1}", road_quality=seg_rq))
                route_segs = custom_segs if custom_segs else INDIA_CITY_ROUTES["bangalore"]

        else:
            rp_profile = st.selectbox(
                "Drive cycle profile",
                ["wltp", "city", "highway", "mixed", "mountain", "custom"],
                index=0, key="rp_profile")

            if rp_profile == "custom":
                st.caption("Add up to 5 custom segments:")
                custom_segs = []
                for i in range(5):
                    with st.expander(f"Segment {i+1}", expanded=(i == 0)):
                        seg_d = st.number_input(f"Distance [km]", 0.0, 500.0, 10.0,
                                                 key=f"rp_d{i}")
                        seg_v = st.number_input(f"Avg speed [km/h]", 5.0, 200.0, 60.0,
                                                 key=f"rp_v{i}")
                        seg_g = st.number_input(f"Grade [%]", -15.0, 15.0, 0.0,
                                                 key=f"rp_g{i}")
                        seg_t = st.slider(f"Traffic factor", 0.0, 1.0, 1.0,
                                           key=f"rp_t{i}")
                        if seg_d > 0:
                            custom_segs.append(RouteSegment(seg_d, seg_v, seg_g,
                                                             seg_t, f"Seg {i+1}"))
                route_segs = custom_segs if custom_segs else ROUTE_PROFILES["wltp"]
            else:
                route_segs = ROUTE_PROFILES[rp_profile]

        # ── Weather ────────────────────────────────────────────────────
        st.markdown("**Weather & Environment**")
        if india_mode:
            _season_opts = ["summer", "monsoon", "winter", "spring"]
            _region_opts = {
                "north": "North (Delhi, UP, Bihar, Punjab)",
                "south_deccan": "South Deccan (Bangalore, Pune, Hyderabad)",
                "coastal": "Coastal (Mumbai, Chennai, Kochi)",
                "hilly": "Hilly (Shimla, Ooty, Darjeeling)",
            }
            wc1, wc2 = st.columns(2)
            rp_season = wc1.selectbox("Season", _season_opts,
                                       format_func=lambda x: x.capitalize(),
                                       key="rp_season")
            rp_region = wc2.selectbox("Region", list(_region_opts.keys()),
                                       format_func=lambda x: _region_opts[x],
                                       key="rp_region")
            _india_key = f"{rp_season}_{rp_region}"
            if _india_key in INDIA_WEATHER:
                weather_cond = INDIA_WEATHER[_india_key]
                st.caption(
                    f"T = **{weather_cond.temperature_C:.0f} °C** · "
                    f"Wind **{weather_cond.wind_speed_ms:.0f} m/s** · "
                    f"Precip: **{weather_cond.precipitation}** · "
                    f"Altitude: **{weather_cond.altitude_m:.0f} m**"
                )
            else:
                weather_cond = WeatherConditions.mild()
                st.caption("Weather preset not found — using mild defaults.")
            rp_temp = weather_cond.temperature_C
            rp_wind = weather_cond.wind_speed_ms
            rp_precip = weather_cond.precipitation
        else:
            rp_temp = st.slider("Temperature [°C]", -30, 50, 20, key="rp_temp")
            rp_wind = st.slider("Wind speed [m/s]", 0, 30, 0, key="rp_wind")
            rp_wind_dir = st.select_slider(
                "Wind direction",
                options=[0, 45, 90, 135, 180],
                value=0,
                format_func=lambda x: {
                    0: "0° Headwind", 45: "45° Quartering",
                    90: "90° Cross", 135: "135° Quartering", 180: "180° Tailwind"
                }[x],
                key="rp_wdir",
            )
            rp_precip = st.selectbox("Precipitation",
                                      ["none", "rain", "snow", "ice"],
                                      key="rp_precip")
            rp_alt = st.slider("Altitude [m]", 0, 4000, 0, 100, key="rp_alt")
            weather_cond = WeatherConditions(
                temperature_C=float(rp_temp),
                wind_speed_ms=float(rp_wind),
                wind_heading_deg=float(rp_wind_dir),
                precipitation=rp_precip,
                altitude_m=float(rp_alt),
            )

    # ── Compute prediction ──────────────────────────────────────────────
    predictor = RangePredictor(vehicle=vehicle)
    pack_Wh = float(rp_kwh) * 1000.0
    result_rp = predictor.predict(
        pack_Wh, str(rp_chem), route_segs, weather_cond, float(rp_soc))

    with rp_right:
        # ── Context summary cards ─────────────────────────────────────────
        _vh_icon = "🛵" if two_wheeler else ("🚗" if vehicle.mass_kg < 2200 else "🚙")
        _wx_icon = {"none": "☀️", "rain": "🌧️", "snow": "❄️", "ice": "🧊"}.get(rp_precip, "🌤️")
        _bat_icon = "🔋"
        st.markdown(f"""
        <div style="display:flex;gap:10px;margin-bottom:10px">
          <div style="flex:1;background:linear-gradient(135deg,rgba(99,102,241,.06),rgba(139,92,246,.04));
                      border:1px solid rgba(99,102,241,.14);border-radius:14px;padding:13px 15px">
            <div style="font-size:.64rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:#64748B;margin-bottom:4px">VEHICLE</div>
            <div style="font-size:1.35rem;line-height:1">{_vh_icon}</div>
            <div style="font-size:.86rem;font-weight:700;color:#1E293B;margin-top:4px">{vehicle.mass_kg:.0f} kg · Cd {vehicle.drag_coefficient:.2f}</div>
            <div style="font-size:.73rem;color:#64748B">{vehicle.frontal_area_m2:.1f} m² · {vehicle.motor_efficiency*100:.0f}% η</div>
          </div>
          <div style="flex:1;background:linear-gradient(135deg,rgba(16,185,129,.06),rgba(6,182,212,.04));
                      border:1px solid rgba(16,185,129,.14);border-radius:14px;padding:13px 15px">
            <div style="font-size:.64rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:#64748B;margin-bottom:4px">BATTERY</div>
            <div style="font-size:1.35rem;line-height:1">{_bat_icon}</div>
            <div style="font-size:.86rem;font-weight:700;color:#1E293B;margin-top:4px">{pack_Wh/1000:.1f} kWh · {rp_chem.upper()}</div>
            <div style="font-size:.73rem;color:#64748B">SOC {rp_soc*100:.0f}% start · {result_rp.usable_energy_Wh/1000:.2f} kWh usable</div>
          </div>
          <div style="flex:1;background:linear-gradient(135deg,rgba(245,158,11,.06),rgba(239,68,68,.04));
                      border:1px solid rgba(245,158,11,.14);border-radius:14px;padding:13px 15px">
            <div style="font-size:.64rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:#64748B;margin-bottom:4px">ENVIRONMENT</div>
            <div style="font-size:1.35rem;line-height:1">{_wx_icon}</div>
            <div style="font-size:.86rem;font-weight:700;color:#1E293B;margin-top:4px">{rp_temp:.0f}°C · {rp_wind:.0f} m/s wind</div>
            <div style="font-size:.73rem;color:#64748B">{rp_precip.capitalize()} · {weather_cond.altitude_m:.0f} m alt</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI row (4 metrics) ───────────────────────────────────────────
        rk1, rk2, rk3, rk4 = st.columns(4)
        rk1.metric("Estimated Range", f"{result_rp.estimated_range_km:.0f} km")
        rk2.metric("Avg Efficiency", f"{result_rp.avg_efficiency_Whkm:.1f} Wh/km")
        rk3.metric("Usable Energy", f"{result_rp.usable_energy_Wh/1000:.2f} kWh")
        if result_rp.route_completable:
            rk4.metric("Route Status", "✓ Complete",
                        delta=f"SOC at dest: {result_rp.soc_at_destination*100:.1f}%")
        else:
            rk4.metric("Route Status", "✗ Range limit",
                        delta=f"Reached {result_rp.estimated_range_km:.0f} km",
                        delta_color="inverse")

        # ── Weather penalty banner ─────────────────────────────────────────
        if abs(result_rp.weather_penalty_pct) > 1.5:
            sign = "+" if result_rp.weather_penalty_pct > 0 else ""
            _pen_c = "#EF4444" if result_rp.weather_penalty_pct > 15 else \
                      "#F59E0B" if result_rp.weather_penalty_pct > 5 else "#10B981"
            _pen_icon = "🌡️" if result_rp.weather_penalty_pct > 0 else "💨"
            st.markdown(f"""
            <div style="background:rgba(248,250,255,.8);border:1px solid rgba(226,232,240,.8);
                        border-left:4px solid {_pen_c};border-radius:10px;
                        padding:9px 14px;margin:4px 0;display:flex;align-items:center;gap:10px">
              <span style="font-size:1.1rem">{_pen_icon}</span>
              <span style="font-weight:700;color:{_pen_c};font-size:.87rem">{sign}{result_rp.weather_penalty_pct:.1f}% energy overhead</span>
              <span style="color:#94A3B8;font-size:.80rem">vs. mild reference (T={rp_temp:.0f}°C · {rp_wind:.0f} m/s · {rp_precip})</span>
            </div>""", unsafe_allow_html=True)

        # ── Route segment strip ───────────────────────────────────────────
        seg_names = [s.name or f"Seg {i+1}" for i, s in enumerate(result_rp.segment_results)]
        soc_vals  = [s.soc_end for s in result_rp.segment_results]
        eff_vals  = [s.efficiency_Whkm for s in result_rp.segment_results]
        cum_km    = np.cumsum([s.distance_km for s in result_rp.segment_results])
        _eff_lo, _eff_hi = (20, 60) if two_wheeler else (150, 250)
        _seg_colors = [
            _P["emerald"] if e < _eff_lo else _P["amber"] if e < _eff_hi else _P["rose"]
            for e in eff_vals
        ]

        fig_strip = go.Figure()
        for i, (seg_r, col, name) in enumerate(
                zip(result_rp.segment_results, _seg_colors, seg_names)):
            fig_strip.add_trace(go.Bar(
                x=[seg_r.distance_km], y=[""],
                orientation="h",
                marker_color=col,
                marker_line=dict(width=2, color="white"),
                name=name, showlegend=False,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"Distance: <b>{seg_r.distance_km:.1f} km</b><br>"
                    f"Efficiency: <b>{seg_r.efficiency_Whkm:.0f} Wh/km</b><br>"
                    f"SOC end: <b>{seg_r.soc_end*100:.1f}%</b><extra></extra>"
                ),
            ))
        _total_km = float(cum_km[-1]) if len(cum_km) else 1.0
        fig_strip.update_layout(
            barmode="stack", height=52,
            margin=dict(l=0, r=0, t=6, b=24),
            xaxis=dict(range=[0, _total_km], showgrid=False, zeroline=False,
                        title=dict(text="Distance [km]", font=dict(size=9)),
                        tickfont=dict(size=9)),
            yaxis=dict(showticklabels=False, showgrid=False, showline=False),
            plot_bgcolor="rgba(248,250,255,1)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.markdown(
            "**Route Profile** &nbsp;"
            "<span style='font-size:.74rem;color:#64748B'>"
            "🟢 efficient &nbsp;🟡 moderate &nbsp;🔴 high draw</span>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_strip, use_container_width=True)

        # ── Energy waterfall + SOC gauge ──────────────────────────────────
        wf_col, gauge_col = st.columns([3, 2])
        with wf_col:
            bd = result_rp.energy_breakdown
            wf_keys = list(bd.keys())
            wf_vals = [v / 1000.0 for v in bd.values()]
            net_kWh = result_rp.total_consumed_Wh / 1000.0
            fig_wf = go.Figure(go.Waterfall(
                x=wf_keys + ["Net Total"],
                y=wf_vals + [net_kWh],
                measure=["relative"] * len(wf_keys) + ["total"],
                orientation="v",
                connector=dict(line=dict(color="#CBD5E1", width=1, dash="dot")),
                increasing=dict(marker=dict(color=_P["rose"],    line=dict(width=0))),
                decreasing=dict(marker=dict(color=_P["emerald"], line=dict(width=0))),
                totals=dict(marker=dict(color=_P["indigo"],      line=dict(width=0))),
                text=[f"{abs(v):.2f}" for v in wf_vals] + [f"{net_kWh:.2f}"],
                textposition="outside", textfont=dict(size=10, color="#374151"),
            ))
            fig_wf.update_layout(
                title=dict(text="Energy Flow [kWh]", font=dict(size=12)),
                yaxis_title="kWh",
                height=310,
                margin=dict(l=44, r=20, t=52, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_wf, use_container_width=True)

        with gauge_col:
            soc_dest_pct = (result_rp.soc_at_destination * 100
                            if result_rp.route_completable else 0.0)
            gc = (_P["emerald"] if soc_dest_pct > 30
                  else _P["amber"] if soc_dest_pct > 15 else _P["rose"])
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=soc_dest_pct,
                title={"text": "SOC at Destination",
                       "font": {"size": 12, "color": "#374151"}},
                number={"suffix": "%", "font": {"size": 34, "color": gc},
                         "valueformat": ".0f"},
                gauge={
                    "axis": {"range": [0, 100], "nticks": 6,
                              "tickcolor": "#94A3B8", "tickwidth": 1},
                    "bar": {"color": gc, "thickness": 0.30},
                    "bgcolor": "#F8FAFF",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 15],   "color": "rgba(239,68,68,0.08)"},
                        {"range": [15, 30],  "color": "rgba(245,158,11,0.08)"},
                        {"range": [30, 100], "color": "rgba(16,185,129,0.06)"},
                    ],
                    "threshold": {
                        "line": {"color": _P["rose"], "width": 3},
                        "thickness": 0.75, "value": 10,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=310, margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            if not result_rp.route_completable:
                fig_gauge.add_annotation(
                    text="<b>RANGE LIMIT</b>", x=0.5, y=0.12,
                    xref="paper", yref="paper", showarrow=False,
                    font=dict(size=13, color=_P["rose"]),
                )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── SOC along route ───────────────────────────────────────────────
        fig_soc_route = go.Figure()
        fig_soc_route.add_hrect(
            y0=0, y1=predictor.SOC_RESERVE * 100,
            fillcolor="rgba(239,68,68,0.05)", line_width=0,
            annotation_text="Reserve",
            annotation_font=dict(size=9, color=_P["rose"]),
            annotation_position="bottom left",
        )
        fig_soc_route.add_trace(go.Scatter(
            x=list(cum_km), y=[v * 100 for v in soc_vals],
            mode="lines+markers",
            line=dict(color=_P["indigo"], width=2.5),
            marker=dict(size=9, color=_seg_colors,
                         line=dict(color="white", width=2)),
            text=seg_names,
            hovertemplate="<b>%{text}</b><br>%{x:.1f} km · SOC <b>%{y:.1f}%</b><extra></extra>",
            fill="tozeroy", fillcolor=_hex_fill(_P["indigo"], 0.06),
        ))
        fig_soc_route.add_hline(
            y=predictor.SOC_RESERVE * 100,
            line_dash="dot", line_color=_P["rose"], line_width=1.5,
            annotation_text=f"  {predictor.SOC_RESERVE*100:.0f}% reserve",
            annotation_font=dict(color=_P["rose"], size=10),
        )
        fig_soc_route.update_layout(
            title=dict(text="Battery SOC Along Route", font=dict(size=12)),
            xaxis_title="Distance [km]", yaxis_title="SOC [%]",
            height=275, margin=dict(l=52, r=20, t=46, b=46),
            yaxis=dict(range=[0, 105]), showlegend=False,
        )
        st.plotly_chart(fig_soc_route, use_container_width=True)

        # ── Per-segment efficiency ─────────────────────────────────────────
        fig_seg_eff = go.Figure(go.Bar(
            y=seg_names, x=eff_vals, orientation="h",
            marker_color=_seg_colors, marker_line_width=0,
            text=[f"{e:.0f}" for e in eff_vals],
            textposition="outside",
            textfont=dict(size=10, color="#374151"),
            hovertemplate="<b>%{y}</b><br>%{x:.1f} Wh/km<extra></extra>",
        ))
        fig_seg_eff.add_vline(x=_eff_lo, line_dash="dot",
                               line_color=_P["emerald"], line_width=1.5)
        fig_seg_eff.add_vline(x=_eff_hi, line_dash="dot",
                               line_color=_P["rose"], line_width=1.5)
        fig_seg_eff.update_layout(
            title=dict(text="Net Efficiency per Segment [Wh/km]", font=dict(size=12)),
            xaxis_title="Wh/km",
            height=max(230, 52 * len(seg_names) + 70),
            margin=dict(l=185, r=60, t=46, b=40),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="#FAFBFF",
        )
        st.plotly_chart(fig_seg_eff, use_container_width=True)

        # ── Temperature sweep ─────────────────────────────────────────────
        _sweep_profile = ("midc" if india_mode
                          else (rp_profile if rp_profile in ROUTE_PROFILES else "wltp"))
        _sweep_temps = list(range(-5, 46, 5)) if india_mode else list(range(-30, 51, 5))

        st.markdown(
            f"**Range vs Temperature** &nbsp; "
            f"<span style='font-size:.80rem;color:#64748B'>"
            f"{rp_chem.upper()} · {pack_Wh/1000:.1f} kWh · {_sweep_profile.upper()}"
            f"</span>",
            unsafe_allow_html=True,
        )
        sweep = predictor.temperature_range_sweep(
            pack_Wh, str(rp_chem), _sweep_profile, temperatures_C=_sweep_temps)
        _sw_T = list(sweep.keys())
        _sw_R = list(sweep.values())

        fig_sweep = go.Figure()
        if not india_mode:
            fig_sweep.add_vrect(x0=-30, x1=0,
                                 fillcolor="rgba(6,182,212,.05)", line_width=0,
                                 annotation_text="❄️ Cold",
                                 annotation_font=dict(size=9, color=_P["cyan"]))
            fig_sweep.add_vrect(x0=32, x1=51,
                                 fillcolor="rgba(239,68,68,.05)", line_width=0,
                                 annotation_text="🔥 Hot",
                                 annotation_font=dict(size=9, color=_P["rose"]))
        else:
            fig_sweep.add_vrect(x0=32, x1=46,
                                 fillcolor="rgba(245,158,11,.07)", line_width=0,
                                 annotation_text="🌅 India Summer",
                                 annotation_font=dict(size=9, color=_P["amber"]))
        fig_sweep.add_trace(go.Scatter(
            x=_sw_T, y=_sw_R,
            mode="lines+markers",
            line=dict(color=_P["indigo"], width=2.5),
            marker=dict(
                size=8,
                color=_sw_R,
                colorscale=[[0, _P["rose"]], [0.45, _P["amber"]], [1, _P["emerald"]]],
                cmin=min(_sw_R), cmax=max(_sw_R),
                line=dict(color="white", width=2), showscale=False,
            ),
            fill="tozeroy", fillcolor=_hex_fill(_P["indigo"], 0.08),
            hovertemplate="<b>T = %{x}°C</b> → Range <b>%{y:.0f} km</b><extra></extra>",
        ))
        fig_sweep.add_vline(
            x=float(rp_temp), line_dash="dash",
            line_color=_P["rose"], line_width=2,
            annotation_text=f" ◀ Now ({rp_temp:.0f}°C)",
            annotation_font=dict(color=_P["rose"], size=11),
            annotation_position="top right",
        )
        fig_sweep.update_layout(
            xaxis_title="Ambient Temperature [°C]",
            yaxis_title="Estimated Range [km]",
            height=300, margin=dict(l=55, r=20, t=28, b=46),
            showlegend=False,
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

        # ── Multi-chemistry comparison ─────────────────────────────────────
        with st.expander("📊 Compare all chemistries at these conditions"):
            comp_rows = []
            for ch in _CHEM_LABELS:
                r_km = predictor.predict_max_range_km(
                    pack_Wh, ch, _sweep_profile, weather_cond)
                r_mild_km = predictor.predict_max_range_km(
                    pack_Wh, ch, _sweep_profile, WeatherConditions.mild())
                comp_rows.append({
                    "Chemistry": _CHEM_LABELS[ch],
                    f"Range @{rp_temp:.0f}°C [km]": round(r_km, 0),
                    "Range @20°C [km]": round(r_mild_km, 0),
                    "Temp penalty [%]": round(
                        (r_mild_km - r_km) / max(r_mild_km, 1) * 100, 1),
                })
            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(
                comp_df.style.background_gradient(
                    subset=[f"Range @{rp_temp:.0f}°C [km]"], cmap="RdYlGn"),
                use_container_width=True,
            )





# ── Main ────────────────────────────────────────────────────────────────────────────
if run_clicked:
    res = run_sim(
        n_cells=int(n_cells), n_parallel=int(n_parallel),
        chemistry=str(chemistry), seed=int(seed),
        mode=mode, duration_s=int(duration_s),
        c_rate=float(c_rate) if c_rate is not None else None,
        p_rate=float(p_rate) if p_rate is not None else None,
        cccv_mode=bool(cccv_mode),
        fault_mode=fault_mode, fault_cell=int(fault_cell),
        fault_start=int(fault_start), fault_end=int(fault_end),
        fault_severity=float(fault_severity),
        show_ekf=bool(show_ekf),
    )
    st.session_state["bms_res"] = res
    st.session_state["bms_ekf"] = show_ekf
    st.session_state["bms_aging"] = aging_cycles

# ── Top-level tab bar ────────────────────────────────────────────────────────
_sim_tab, _rp_tab = st.tabs(["🔬 Simulation", "🚗 Range Predictor"])

with _sim_tab:
    if "bms_res" in st.session_state:
        render_results(
            st.session_state["bms_res"],
            show_ekf=bool(st.session_state["bms_ekf"]),
            aging_cycles=int(st.session_state["bms_aging"]),
        )
    else:
        st.markdown(
            """
    <div style="text-align:center;padding:32px 0 20px">
      <div style="font-size:2.4rem;font-weight:900;
                  background:linear-gradient(100deg,#4338CA,#6366F1,#8B5CF6,#06B6D4);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  display:inline-block;margin-bottom:6px">BMS Digital Twin v5</div>
      <div style="color:#64748B;font-size:.95rem;max-width:540px;margin:0 auto;line-height:1.55">
        Configure the simulation on the left and click
        <strong>&#9654; Run simulation</strong> to begin.<br>
        The <strong>Range Predictor</strong> tab above is always available.
      </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:10px 0">
      <div class="feat-card">
        <div style="font-size:1.5rem;margin-bottom:5px">&#9889;</div>
        <div style="font-weight:700;font-size:.92rem;color:#1E293B;margin-bottom:3px">Tab 1 · Cell Physics</div>
        <div style="font-size:.79rem;color:#64748B;line-height:1.5">2nd-order RC ECM · 7 chemistries · CC-CV · EKF/UKF/LSTM ±1σ</div>
      </div>
      <div class="feat-card">
        <div style="font-size:1.5rem;margin-bottom:5px">🏗️</div>
        <div style="font-weight:700;font-size:.92rem;color:#1E293B;margin-bottom:3px">Tab 2 · Pack Topology</div>
        <div style="font-size:.79rem;color:#64748B;line-height:1.5">nS×mP · 3 balancer types · 2-D thermal model</div>
      </div>
      <div class="feat-card">
        <div style="font-size:1.5rem;margin-bottom:5px">🛡️</div>
        <div style="font-weight:700;font-size:.92rem;color:#1E293B;margin-bottom:3px">Tab 3 · Fault Detection</div>
        <div style="font-size:.79rem;color:#64748B;line-height:1.5">8 fault modes · hybrid LSTM+rules · FMEA RPN</div>
      </div>
      <div class="feat-card">
        <div style="font-size:1.5rem;margin-bottom:5px">📋</div>
        <div style="font-weight:700;font-size:.92rem;color:#1E293B;margin-bottom:3px">Tab 4 · Battery Passport</div>
        <div style="font-size:.79rem;color:#64748B;line-height:1.5">EFC · DWC · round-trip η · DVA/ICA · EIS</div>
      </div>
      <div class="feat-card">
        <div style="font-size:1.5rem;margin-bottom:5px">🔬</div>
        <div style="font-weight:700;font-size:.92rem;color:#1E293B;margin-bottom:3px">Tab 5 · Diagnostics</div>
        <div style="font-size:.79rem;color:#64748B;line-height:1.5">C-rate heatmap (SOC×T) · RUL · SoH trajectories</div>
      </div>
      <div class="feat-card" style="border-color:rgba(99,102,241,.22);background:linear-gradient(135deg,rgba(99,102,241,.05),rgba(6,182,212,.03))">
        <div style="font-size:1.5rem;margin-bottom:5px">🚗</div>
        <div style="font-weight:700;font-size:.92rem;color:#4338CA;margin-bottom:3px">Range Predictor → always live</div>
        <div style="font-size:.79rem;color:#64748B;line-height:1.5">WLTP · MIDC · 🇮🇳 India · 🛵 2-Wheeler</div>
      </div>
    </div>
            """,
            unsafe_allow_html=True,
        )

with _rp_tab:
    render_range_predictor()
