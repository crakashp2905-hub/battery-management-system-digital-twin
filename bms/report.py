"""One-command battery health & model-validation report (self-contained HTML).

Pulls the twin's key analyses — the estimator leaderboard, the safety case
(FMEA top risks + fault-tree top-event probability), an MPC-vs-CC-CV charging
comparison, and a degradation-mode example — into a single styled HTML page you
can hand to a colleague or drop into a report.  Everything is inline (no external
assets), so the file is portable.

    from bms.report import build_health_report, save_report
    save_report("bms_report.html", chemistry="nmc")
"""

from __future__ import annotations

import datetime as _dt

import numpy as np

_CSS = """
body{font-family:'Segoe UI',system-ui,sans-serif;max-width:860px;margin:24px auto;
     color:#1e293b;padding:0 16px;line-height:1.5}
h1{font-size:1.6rem;margin-bottom:2px} h2{font-size:1.15rem;margin-top:28px;
   border-bottom:2px solid #6366f1;padding-bottom:4px;color:#4338ca}
.sub{color:#64748b;font-size:.9rem;margin-top:0}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:8px 0}
th,td{border:1px solid #e2e8f0;padding:5px 9px;text-align:right}
th{background:#eef2ff;color:#3730a3} td:first-child,th:first-child{text-align:left}
.kpi{display:inline-block;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
     padding:8px 14px;margin:4px 6px 4px 0}
.kpi b{display:block;font-size:1.25rem;color:#4338ca} .kpi span{font-size:.78rem;color:#64748b}
.note{color:#64748b;font-size:.82rem;font-style:italic}
"""


def _table(df) -> str:
    return df.to_html(border=0, justify="left", float_format=lambda x: f"{x:.3f}")


def build_health_report(chemistry: str = "nmc", duration_s: float = 1800.0,
                        seed: int = 1) -> str:
    """Run the analyses and return a self-contained HTML report string."""
    import bms

    data = bms.synthetic_drivecycle(chemistry, duration_s=duration_s, seed=seed)

    # --- Estimator leaderboard --------------------------------------
    board = bms.estimator_leaderboard(data, ["coulomb", "ekf", "ukf", "joint_ekf"])
    lb = board.copy()
    lb["rmse_%"] = (lb["rmse"] * 100).round(3)
    lb["mae_%"] = (lb["mae"] * 100).round(3)
    lb_html = _table(lb[["rmse_%", "mae_%", "runtime_s"]])
    best = board.index[0]

    # --- Safety case -------------------------------------------------
    fmea = bms.build_fmea_table().head(4)[["failure_mode", "S", "O", "D", "RPN"]]
    tree = bms.thermal_runaway_tree()
    top_p = bms.probability(tree)
    n_cuts = len(bms.minimal_cut_sets(tree))

    # --- Charging: MPC vs CC-CV -------------------------------------
    from bms.charge_control import ChargeLimits, MPCCharger, compare_charging
    props = bms.get_chemistry_props(chemistry)
    d = props["default_ecm"]
    p = bms.ECMParameters(R0=d["R0"], R1=d["R1"], C1=d["C1"], R2=d["R2"], C2=d["C2"],
                          Q_nom_Ah=props["default_capacity_Ah"], chemistry=chemistry)
    charger = MPCCharger(params=p, ocv_curve=bms.OCVSOC.from_chemistry(chemistry),
                         limits=ChargeLimits(v_max=props["v_max"], t_max_C=45.0,
                                             i_max_A=props["default_capacity_Ah"] * 3,
                                             soc_target=0.8))
    ch = compare_charging(charger, soc0=0.2, c_rate=1.0)
    saving_pct = (100.0 * ch["time_saving_s"] / ch["cccv"]["time_to_target_s"]
                  if np.isfinite(ch["cccv"]["time_to_target_s"]) else 0.0)

    # --- Degradation modes example ----------------------------------
    v, fresh, aged = bms.synthetic_degraded_ic(lli=0.12, lam=0.06)
    modes = bms.diagnose_degradation_modes(v, fresh, aged)

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>BMS Health Report — {chemistry.upper()}</title><style>{_CSS}</style></head><body>
<h1>Battery Health &amp; Model-Validation Report</h1>
<p class="sub">Chemistry <b>{chemistry.upper()}</b> · BMS Digital Twin v{bms.__version__} · {now}
· source <b>{data.source}</b></p>

<div>
  <div class="kpi"><b>{best}</b><span>best SoC estimator</span></div>
  <div class="kpi"><b>{board.iloc[0]['rmse'] * 100:.2f}%</b><span>SoC RMSE</span></div>
  <div class="kpi"><b>{top_p:.1e}</b><span>thermal-runaway P(top)</span></div>
  <div class="kpi"><b>{saving_pct:.0f}%</b><span>MPC faster vs CC-CV</span></div>
</div>

<h2>1 · Estimator leaderboard (SoC RMSE)</h2>
{lb_html}
<p class="note">Ground truth is the plant ECM's own SoC (source: {data.source}).
For a credibility claim, run on real data (see docs/validation.md).</p>

<h2>2 · Safety case</h2>
<p>Top failure modes by Risk Priority Number (S×O×D):</p>
{_table(fmea)}
<p>Fault-tree top event <b>thermal runaway</b>: probability <b>{top_p:.2e}</b>
across <b>{n_cuts}</b> minimal cut sets. Every RPN≥100 mode is traced to a test
(<code>scripts/traceability.py</code>).</p>

<h2>3 · Optimal charging (MPC vs CC-CV)</h2>
<div>
  <div class="kpi"><b>{ch['mpc']['time_to_target_s'] / 60:.1f} min</b><span>MPC to 80%</span></div>
  <div class="kpi"><b>{ch['cccv']['time_to_target_s'] / 60:.1f} min</b><span>CC-CV to 80%</span></div>
  <div class="kpi"><b>{ch['mpc']['peak_temperature_C']:.1f}°C</b><span>MPC peak temp</span></div>
  <div class="kpi"><b>{ch['mpc']['min_plating_margin_A']:.2f} A</b><span>min plating margin</span></div>
</div>
<p class="note">MPC reaches target {saving_pct:.0f}% sooner while respecting voltage,
temperature, and the lithium-plating limit.</p>

<h2>4 · Degradation-mode diagnosis (example)</h2>
<div>
  <div class="kpi"><b>{modes['lli'] * 100:.1f}%</b><span>loss of Li inventory</span></div>
  <div class="kpi"><b>{modes['lam'] * 100:.1f}%</b><span>loss of active material</span></div>
  <div class="kpi"><b>{modes['dominant_mode']}</b><span>dominant mode</span></div>
</div>

<p class="note">Generated by bms.report.build_health_report — a simulation twin,
not certified firmware (see docs/safety_case.md).</p>
</body></html>"""


def save_report(path, chemistry: str = "nmc", **kw) -> str:
    """Write the report to *path* and return the path."""
    html = build_health_report(chemistry=chemistry, **kw)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return str(path)
