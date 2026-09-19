"""Probabilistic prognostics — from thresholds to *probabilities and lead time*.

Deterministic limits answer "did it fail?"; prognostics answers the more useful
question "how likely is failure, and how soon?".  This module gives three
probabilistic read-outs the rest of the twin can act on:

* :func:`thermal_runaway_probability` — ``P(runaway < t)`` over several horizons,
  from the current temperature and its trend, treating the onset temperature as
  uncertain and letting cell imbalance / internal pressure lower the effective
  onset (a stressed cell runs away sooner).  A Gaussian crossing model, not a
  hard threshold.
* :func:`rul_distribution` — remaining-useful-life as a **distribution** (mean +
  90 % interval) by propagating the uncertainty in the observed fade rate, so
  "612 cycles" becomes "612 cycles (540–704)".
* :func:`predict_failure` — a precursor-fusion failure probability with an
  estimated **lead time**, combining normalised early-warning signals (pressure
  rise, resistance jump, voltage divergence, temperature spread).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _phi(x: float) -> float:
    """Standard-normal CDF via erf (no scipy needed in the hot path)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ── Thermal-runaway probability ─────────────────────────────────────────
@dataclass(frozen=True)
class RunawayForecast:
    """``P(thermal runaway < t)`` over a set of horizons."""

    horizons_s: tuple
    probabilities: dict            # horizon_s -> probability in [0, 1]
    effective_onset_C: float
    projected_temperature_C: dict  # horizon_s -> projected T

    def prob_within(self, horizon_s: float) -> float:
        return float(self.probabilities.get(horizon_s, float("nan")))

    def to_dict(self) -> dict:
        return {"probabilities": self.probabilities,
                "effective_onset_C": self.effective_onset_C,
                "projected_temperature_C": self.projected_temperature_C}


def thermal_runaway_probability(temperature_C: float, dT_dt_C_per_s: float, *,
                                onset_C: float = 70.0, onset_sigma_C: float = 8.0,
                                imbalance: float = 0.0, pressure_kPa: float = 0.0,
                                pressure_ref_kPa: float = 150.0,
                                horizons_s: tuple = (30.0, 300.0, 1800.0)
                                ) -> RunawayForecast:
    """Probability of thermal runaway within each horizon.

    Projects ``T(t) = T + dT/dt · t`` and compares it to an **uncertain** onset
    temperature; imbalance and over-pressure lower the effective onset. Returns a
    monotone-increasing probability curve over ``horizons_s``.
    """
    # A stressed cell lets go sooner: knock the onset down for imbalance / pressure.
    imbalance_penalty = 30.0 * max(0.0, float(imbalance))         # up to 30 °C
    pressure_penalty = 20.0 * max(0.0, (pressure_kPa - pressure_ref_kPa) / pressure_ref_kPa)
    eff_onset = onset_C - imbalance_penalty - pressure_penalty
    sigma = max(onset_sigma_C, 1e-6)

    probs, proj = {}, {}
    for t in horizons_s:
        t_proj = temperature_C + dT_dt_C_per_s * t
        probs[t] = float(_phi((t_proj - eff_onset) / sigma))
        proj[t] = float(t_proj)
    return RunawayForecast(tuple(horizons_s), probs, float(eff_onset), proj)


# ── Probabilistic remaining useful life ─────────────────────────────────
@dataclass(frozen=True)
class RULDistribution:
    """Remaining-useful-life as a distribution (cycles to end-of-life)."""

    mean_cycles: float
    median_cycles: float
    p05_cycles: float
    p95_cycles: float
    std_cycles: float

    def to_dict(self) -> dict:
        return {"mean_cycles": self.mean_cycles, "median_cycles": self.median_cycles,
                "p05_cycles": self.p05_cycles, "p95_cycles": self.p95_cycles,
                "std_cycles": self.std_cycles}


def rul_distribution(soh_now: float, fade_per_cycle: float, *,
                     fade_sigma: float | None = None, eol: float = 0.8,
                     n: int = 5000, seed: int = 0) -> RULDistribution:
    """RUL distribution by propagating uncertainty in the fade rate.

    ``fade_sigma`` (default 25 % of the rate) is the 1-σ uncertainty in the
    per-cycle fade; samples are drawn from a positive-truncated normal so a
    faster-fading draw shortens life and vice-versa.
    """
    if soh_now <= eol:
        return RULDistribution(0.0, 0.0, 0.0, 0.0, 0.0)
    fade_sigma = 0.25 * fade_per_cycle if fade_sigma is None else fade_sigma
    rng = np.random.default_rng(seed)
    fades = rng.normal(fade_per_cycle, fade_sigma, n)
    fades = np.clip(fades, 1e-9, None)                 # fade is strictly positive
    rul = (soh_now - eol) / fades
    return RULDistribution(
        mean_cycles=float(np.mean(rul)), median_cycles=float(np.median(rul)),
        p05_cycles=float(np.percentile(rul, 5)), p95_cycles=float(np.percentile(rul, 95)),
        std_cycles=float(np.std(rul)),
    )


# ── Predictive failure detection ────────────────────────────────────────
@dataclass(frozen=True)
class FailurePrediction:
    """Precursor-fusion failure probability with an estimated lead time."""

    failure_probability: float
    lead_time_s: float
    dominant_precursor: str
    contributions: dict

    def to_dict(self) -> dict:
        return {"failure_probability": self.failure_probability,
                "lead_time_s": self.lead_time_s,
                "dominant_precursor": self.dominant_precursor,
                "contributions": self.contributions}


# Evidence weight of each early-warning precursor (log-odds per unit severity).
_PRECURSOR_WEIGHTS = {
    "pressure_rise": 3.0,
    "resistance_jump": 2.5,
    "voltage_divergence": 2.0,
    "temperature_spread": 2.0,
    "gas_detected": 3.5,
}


def predict_failure(signals: dict, *, lead_times_s: dict | None = None,
                    bias: float = -2.5) -> FailurePrediction:
    """Fuse normalised precursor severities into a failure probability.

    ``signals`` maps precursor name → severity in ``[0, 1]``.  Probability is a
    logistic of the weighted evidence; the lead time is taken from the
    strongest-firing precursor (``lead_times_s`` gives per-precursor horizons,
    scaled shorter as its severity rises).
    """
    contributions, logit = {}, float(bias)
    for name, sev in signals.items():
        w = _PRECURSOR_WEIGHTS.get(name, 1.0)
        c = w * float(np.clip(sev, 0.0, 1.0))
        contributions[name] = float(round(c, 4))
        logit += c
    prob = 1.0 / (1.0 + math.exp(-logit))

    if contributions and any(v > 0 for v in contributions.values()):
        dominant = max(contributions, key=contributions.get)
        base_lead = (lead_times_s or {}).get(dominant, 600.0)
        # A more severe precursor implies less time left.
        sev = float(np.clip(signals.get(dominant, 0.0), 0.0, 1.0))
        lead = base_lead * (1.0 - 0.8 * sev)
    else:
        dominant, lead = "none", float("inf")
    return FailurePrediction(float(prob), float(lead), dominant, contributions)
