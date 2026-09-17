"""Fleet reliability: Monte-Carlo life & warranty curves.

The aging model gives *one* SoH trajectory; a fleet has a **distribution** of
them, because cells vary.  This samples per-cell manufacturing scatter in the
aging rate, runs each cell to end-of-life through :class:`bms.AgingModel`, and
turns the spread into the numbers a warranty desk needs:

* a **SoH band** over time (median with P5–P95),
* a **warranty curve** — ``P(SoH < EoL)`` versus year,
* the **RUL distribution** (years-to-EoL) with a confidence interval,
* the **B10 life** — the year by which 10 % of the fleet has reached EoL.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .aging import AgingModel, AgingParams, AgingState


def monte_carlo_life(n_samples: int = 500, years: int = 12,
                     cycles_per_year: float = 250.0, c_rate: float = 1.0,
                     temperature_C: float = 25.0, dod: float = 0.8,
                     soc_avg: float = 0.5, k_cycle_cov: float = 0.15,
                     params: AgingParams | None = None, eol: float = 0.8,
                     seed: int = 0) -> dict:
    """Monte-Carlo SoH over a fleet with a scattered cycle-aging rate.

    ``k_cycle_cov`` is the coefficient of variation of the per-cell aging rate
    (lognormal, mean 1).  Returns SoH percentiles per year, the warranty curve,
    the RUL distribution, and the B10 life.
    """
    rng = np.random.default_rng(seed)
    base = params or AgingParams()
    sigma = float(np.sqrt(np.log(1.0 + k_cycle_cov ** 2)))
    mult = rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma, size=n_samples)

    yr = np.arange(0, years + 1)
    soh = np.ones((n_samples, years + 1))
    for i in range(n_samples):
        model = AgingModel(replace(base, k_cycle=base.k_cycle * float(mult[i])))
        st = AgingState()
        for y in range(1, years + 1):
            st = model.cycle(st, cycles_per_year, c_rate, temperature_C, dod, soc_avg)
            st = model.calendar(st, 365.0, temperature_C, soc_avg)
            soh[i, y] = st.soh_capacity

    warranty = np.mean(soh < eol, axis=0)                 # P(failed) by year
    # RUL per cell = first year at/below EoL (linear-interp within the year).
    rul = np.full(n_samples, float(years))
    for i in range(n_samples):
        below = np.where(soh[i] <= eol)[0]
        if below.size:
            j = below[0]
            if j > 0:                                     # interpolate the crossing
                s0, s1 = soh[i, j - 1], soh[i, j]
                frac = (s0 - eol) / max(s0 - s1, 1e-9)
                rul[i] = (j - 1) + frac
            else:
                rul[i] = 0.0
    b10 = float(np.interp(0.10, warranty, yr)) if warranty[-1] >= 0.10 else float("inf")

    return {
        "years": yr,
        "soh_p50": np.percentile(soh, 50, axis=0),
        "soh_p05": np.percentile(soh, 5, axis=0),
        "soh_p95": np.percentile(soh, 95, axis=0),
        "warranty_curve": warranty,
        "rul_years_mean": float(np.mean(rul)),
        "rul_years_p05": float(np.percentile(rul, 5)),
        "rul_years_p95": float(np.percentile(rul, 95)),
        "b10_life_years": b10,
        "eol": eol,
    }


def warranty_reserve(mc: dict, warranty_years: float) -> float:
    """Fraction of the fleet expected to need replacement within the warranty.

    Reads the warranty curve from :func:`monte_carlo_life` at ``warranty_years``
    — the provision a manufacturer should reserve for.
    """
    return float(np.interp(warranty_years, mc["years"], mc["warranty_curve"]))
