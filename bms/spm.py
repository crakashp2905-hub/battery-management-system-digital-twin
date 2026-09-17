"""Single Particle Model (SPM) — a lightweight *electrochemical* cell model.

Everything else in the package is an **equivalent-circuit model** (ECM): fast, but
it fakes the physics with lumped R-C.  The SPM is a genuine (reduced) physics
model: each electrode is one spherical particle in which lithium **diffuses**
(Fick's law), and the terminal voltage comes from the *surface* stoichiometry
through each electrode's open-circuit potential.

That surface-vs-bulk distinction is the physics an ECM can't express: at high
C-rate the particle **surface saturates/depletes** faster than its interior, so
the voltage sags and recovers on rest as the gradient relaxes — real diffusion
limitation and rate capability, not a fitted R-C.  This is the SPM (the base of
PyBaMM's model hierarchy), implemented in NumPy with a conservative finite-volume
solid-diffusion solver, so it needs no heavyweight electrochemistry dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _ocp_neg(theta: np.ndarray) -> np.ndarray:
    """Graphite negative-electrode OCP [V] vs Li (monotonic, ~0.05–0.3 V)."""
    theta = np.clip(theta, 1e-4, 1 - 1e-4)
    return 0.08 + 0.20 * np.exp(-25.0 * theta) + 0.05 * (1.0 - theta)


def _ocp_pos(theta: np.ndarray) -> np.ndarray:
    """NMC positive-electrode OCP [V] vs Li (monotonic decreasing, ~3.5–4.3 V)."""
    theta = np.clip(theta, 1e-4, 1 - 1e-4)
    return 4.25 - 0.80 * theta - 0.35 * theta ** 2 + 0.10 * np.exp(-15.0 * theta)


@dataclass
class SPMParams:
    """Reduced-SPM parameters (dimensionless diffusion rates D/R²)."""

    diffusion_rate_neg: float = 6e-4     # D/R² for the negative particle [1/s]
    diffusion_rate_pos: float = 4e-4     # D/R² for the positive particle [1/s]
    R0_ohm: float = 0.03                 # lumped ohmic + kinetic series resistance
    Q_nom_Ah: float = 2.3
    n_shells: int = 10
    # Stoichiometry windows the electrodes cycle over.
    theta_neg_0: float = 0.02            # neg at SoC 0 (delithiated graphite)
    theta_neg_1: float = 0.90            # neg at SoC 1 (lithiated graphite)
    theta_pos_0: float = 0.90            # pos at SoC 0 (lithiated NMC)
    theta_pos_1: float = 0.20            # pos at SoC 1 (delithiated NMC)


@dataclass
class SingleParticleModel:
    """Two-particle SPM with finite-volume spherical diffusion."""

    params: SPMParams = field(default_factory=SPMParams)

    def __post_init__(self) -> None:
        n = self.params.n_shells
        self._edges = np.linspace(0.0, 1.0, n + 1)
        self._centers = 0.5 * (self._edges[:-1] + self._edges[1:])
        self._vol = self._edges[1:] ** 3 - self._edges[:-1] ** 3      # ∝ shell volume, Σ=1
        self._area = self._edges ** 2                                  # face areas (x²)
        self._dx = np.diff(self._centers)                             # centre spacing
        self.reset(1.0)

    def reset(self, soc0: float = 1.0) -> None:
        p = self.params
        s = float(np.clip(soc0, 0.0, 1.0))
        self._neg = np.full(p.n_shells, p.theta_neg_0 + s * (p.theta_neg_1 - p.theta_neg_0))
        self._pos = np.full(p.n_shells, p.theta_pos_0 + s * (p.theta_pos_1 - p.theta_pos_0))

    # -- solid diffusion (conservative FV Laplacian, no-flux both ends) --
    def _laplacian(self, theta: np.ndarray) -> np.ndarray:
        flux = self._area[1:-1] * (theta[1:] - theta[:-1]) / self._dx   # internal faces
        d = np.zeros_like(theta)
        d[:-1] += flux
        d[1:] -= flux
        return d / self._vol

    def _diffuse(self, theta, rate, surface_rate, dt):
        # Sub-step for forward-Euler stability, then inject the surface flux.
        sub = max(1, int(np.ceil(rate * dt / (0.2 * self._dx.min() ** 2))))
        h = dt / sub
        for _ in range(sub):
            theta = theta + rate * h * self._laplacian(theta)
        theta[-1] += surface_rate * dt / self._vol[-1]                  # boundary current
        return np.clip(theta, 1e-4, 1 - 1e-4)

    def step(self, current: float, dt: float) -> float:
        """Advance by ``dt`` (``current > 0`` = discharge); return terminal V."""
        p = self.params
        rate = current / (p.Q_nom_Ah * 3600.0)          # d(mean θ)/dt magnitude
        # Discharge: neg loses Li (−), pos gains Li (+).
        self._neg = self._diffuse(self._neg, p.diffusion_rate_neg, -rate, dt)
        self._pos = self._diffuse(self._pos, p.diffusion_rate_pos, +rate, dt)
        return self.terminal_voltage(current)

    def terminal_voltage(self, current: float = 0.0) -> float:
        p = self.params
        u_pos = float(_ocp_pos(np.array([self._pos[-1]]))[0])
        u_neg = float(_ocp_neg(np.array([self._neg[-1]]))[0])
        return u_pos - u_neg - p.R0_ohm * float(current)

    @property
    def soc(self) -> float:
        """SoC from the negative particle's *mean* stoichiometry."""
        p = self.params
        mean_neg = float(np.sum(self._vol * self._neg))
        return float(np.clip((mean_neg - p.theta_neg_0) / (p.theta_neg_1 - p.theta_neg_0),
                             0.0, 1.0))

    @property
    def surface_soc(self) -> float:
        """SoC from the negative particle's *surface* stoichiometry (< bulk under load)."""
        p = self.params
        return float(np.clip((self._neg[-1] - p.theta_neg_0) / (p.theta_neg_1 - p.theta_neg_0),
                             0.0, 1.0))

    def simulate(self, current: np.ndarray, dt: float, soc0: float | None = None) -> dict:
        if soc0 is not None:
            self.reset(soc0)
        current = np.asarray(current, float)
        n = len(current)
        v, soc, surf = np.empty(n), np.empty(n), np.empty(n)
        for k in range(n):
            v[k] = self.step(float(current[k]), dt)
            soc[k] = self.soc
            surf[k] = self.surface_soc
        return {"v_terminal": v, "soc": soc, "surface_soc": surf}
