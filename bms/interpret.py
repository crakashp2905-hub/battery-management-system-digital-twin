"""
Interpretability: turn the twin's numeric state into things a human can read.

Nothing here changes the physics — it *explains* it:

* :func:`explain_state` / :func:`explain_charge` — one-line plain-language
  summaries of a supervisor step or a charge session.
* :func:`feature_importances` — the fault detector's RandomForest importances,
  mapped onto named features, so an ML alarm is explainable ("driven by
  temperature spread"), not a black box.
* :func:`estimator_agreement` — compare several estimators, report their spread,
  flag disagreement, and return an inverse-variance **fused** estimate that
  trusts the more-certain models more.  Divergence between EKF/UKF/LSTM is
  itself a trust signal.
* :func:`soc_report` — a SoC value rendered with its ±1σ / ±2σ band.

These are the building blocks a dashboard (or an LLM narration layer) turns into
"what is happening and why".
"""

from __future__ import annotations

import numpy as np

from .estimation import Estimate, soc_estimate

# Names for the 14-element detector feature vector (10 base + 4 rolling),
# matching bms.faults.extract_features + RollingFeatureBuffer.stats_vector.
FEATURE_NAMES: list[str] = [
    "v_max", "v_min", "v_spread", "i_max", "abs_i_max",
    "T_max", "T_spread", "dv_max", "abs_dv_max", "dT_max",
    "roll_v_std", "roll_T_std", "roll_v_drift", "roll_T_drift",
]


# ======================================================================
# Fault-detector interpretability
# ======================================================================
def feature_importances(detector) -> dict[str, float]:
    """Return the fault detector's RandomForest importances by feature name.

    Sorted most-important first.  The detector must have been ``fit``.
    """
    if not getattr(detector, "fitted", False):
        raise RuntimeError("detector is not fitted; call detector.fit(X, y) first")
    imp = np.asarray(detector.clf.feature_importances_, float)
    names = FEATURE_NAMES[:len(imp)] + [
        f"feature_{i}" for i in range(len(FEATURE_NAMES), len(imp))]
    pairs = sorted(zip(names, imp), key=lambda kv: -kv[1])
    return {name: float(value) for name, value in pairs}


# ======================================================================
# Multi-estimator agreement + fusion
# ======================================================================
def estimator_agreement(estimates: dict[str, Estimate],
                        disagreement_threshold: float = 0.05) -> dict:
    """Summarise agreement across estimators and fuse them.

    Parameters
    ----------
    estimates : dict[str, Estimate]
        Named estimates (see :func:`bms.soc_estimate`).
    disagreement_threshold : float
        Max-minus-min above which ``disagreement`` is flagged.

    Returns
    -------
    dict with ``mean``, ``spread``, ``fused`` (inverse-variance weighted where
    uncertainty is known, else plain mean), ``disagreement`` (bool), and ``n``.
    """
    if not estimates:
        raise ValueError("no estimates provided")
    values = np.array([e.value for e in estimates.values()], float)
    weighted_v = [e.value for e in estimates.values()
                  if e.has_uncertainty and e.sigma > 0]
    weights = [1.0 / e.sigma ** 2 for e in estimates.values()
               if e.has_uncertainty and e.sigma > 0]
    fused = (float(np.average(weighted_v, weights=weights)) if weights
             else float(values.mean()))
    spread = float(values.max() - values.min())
    return {
        "mean": float(values.mean()),
        "spread": spread,
        "fused": fused,
        "disagreement": bool(spread > disagreement_threshold),
        "n": len(estimates),
    }


def soc_report(estimator, k: float = 1.0) -> str:
    """Render an estimator's SoC with its ±kσ band (or 'uncertainty unknown')."""
    e = soc_estimate(estimator)
    if not e.has_uncertainty:
        return f"SoC {e.value * 100:.1f}% (uncertainty unknown)"
    band = k * e.sigma * 100.0
    return f"SoC {e.value * 100:.1f}% ± {band:.1f}% ({k:.0f}σ)"


# ======================================================================
# Plain-language summaries
# ======================================================================
def explain_state(result: dict, soh: float | None = None) -> str:
    """One-line plain-language summary of a :meth:`BMSSupervisor.step` result."""
    soc = float(np.mean(result["soc"])) * 100.0
    v = float(result["v_pack"])
    power = float(result.get("power_W", 0.0))
    t_max = float(np.max(result["T_cells"]))
    duty = float(result.get("cooling_duty", 0.0)) * 100.0
    fault = result.get("fault_label", "none")
    source = result.get("fault_source", "none")

    parts = [f"State: {str(result.get('state', '?')).upper()}.",
             f"Mean SoC {soc:.1f}% ({v:.1f} V, {power:.0f} W)."]
    if soh is not None:
        parts.append(f"SoH {soh * 100:.1f}%.")
    parts.append(f"Hottest cell {t_max:.0f} °C, cooling {duty:.0f}%.")
    if fault != "none":
        parts.append(f"⚠ Fault: {fault.replace('_', ' ')} (via {source} layer).")
    else:
        parts.append("No active fault.")
    contactor = result.get("contactor_state")
    if contactor:
        parts.append(f"Contactor {contactor}.")
    return " ".join(parts)


def explain_charge(result) -> str:
    """One-line plain-language explanation of a :class:`bms.charging.ChargeResult`."""
    path = "DC" if str(result.method).startswith("dc") else "AC"
    text = (f"{result.method} ({path}, {result.c_rate:.1f}C): "
            f"{result.duration_min:.0f} min, {result.efficiency * 100:.0f}% efficient, "
            f"peak {result.peak_cell_temp_C:.0f} °C. ")
    if result.plating_risk > 0.5:
        text += (f"High lithium-plating risk ({result.plating_risk:.1f}) — this "
                 f"charge ages the cell fast. ")
    elif result.plating_risk > 0.0:
        text += f"Some plating risk ({result.plating_risk:.1f}). "
    else:
        text += "No plating — gentle on the cell. "
    text += f"Capacity fade this charge ≈ {result.capacity_fade_pct:.4f}%."
    return text
