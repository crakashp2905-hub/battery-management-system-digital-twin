"""
Second-order RC equivalent-circuit model (ECM).

State vector  x = [SOC, V_RC1, V_RC2]
Input         u = i  (positive on discharge)
Output        y = V_terminal = OCV(SOC) - V_RC1 - V_RC2 - R0 * i

Time-discrete (zero-order hold) recurrences
-------------------------------------------
SOC[k+1]  = SOC[k] - i[k] * dt / (Q_nom_Ah * 3600)
V_RC1[k+1] = a1 * V_RC1[k] + (1 - a1) * R1 * i[k]   ; a1 = exp(-dt/(R1*C1))
V_RC2[k+1] = a2 * V_RC2[k] + (1 - a2) * R2 * i[k]   ; a2 = exp(-dt/(R2*C2))
V_t[k]    = OCV(SOC[k]) - V_RC1[k] - V_RC2[k] - R0 * i[k]

Temperature dependence
----------------------
Resistances follow an Arrhenius law:  R(T) = R_ref * exp(Ea/R * (1/T - 1/T_ref))
with Ea/R ≈ 4 000 K, T_ref = 25 °C. At 0 °C, R roughly doubles; at 45 °C it halves.
Usable capacity de-rates ~0.2 %/°C below 25 °C (simplified linear model).

Parameter identification
------------------------
`fit_ecm_parameters` uses scipy.optimize.least_squares to minimise
||V_meas - V_sim||² over (R0, R1, C1, R2, C2) given a current and voltage
trace and an OCV-SOC characteristic. SOC is anchored from the initial rest
voltage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .ocv_soc import OCVSOC

_T_REF_K: float = 298.15  # 25 °C reference temperature in Kelvin


@dataclass
class ECMParameters:
    """Parameter set for the second-order RC ECM (per cell) at 25 °C.

    Parameters
    ----------
    chemistry : str
        Cell chemistry string (``"nmc"``, ``"lfp"``, ``"lmfp"``, ``"lto"``,
        ``"nca"``, ``"lmo"``).  Controls the Arrhenius activation energy in
        :meth:`at_temperature` and the OCV table used.
    self_discharge_pct_per_month : float
        Monthly self-discharge rate at 25 °C [%].  0.0 = disabled (default).
        Applied in :class:`SecondOrderECM` as a continuous SOC drain during
        both active and idle periods.
    """

    R0: float = 0.025                          # ohmic resistance [Ω]
    R1: float = 0.015                          # fast-dynamic resistance [Ω]
    C1: float = 2_000.0                        # fast-dynamic capacitance [F]
    R2: float = 0.030                          # slow-dynamic resistance [Ω]
    C2: float = 8_000.0                        # slow-dynamic capacitance [F]
    Q_nom_Ah: float = 2.3                      # nominal capacity [Ah]
    chemistry: str = "nmc"                     # cell chemistry
    self_discharge_pct_per_month: float = 0.0  # self-discharge rate [%/month]

    def __post_init__(self) -> None:
        from .chemistry import get_chemistry_props
        self._arrhenius_K: float = get_chemistry_props(self.chemistry)["arrhenius_K"]

    # ------------------------------------------------------------------
    @classmethod
    def for_nmc(cls, Q_nom_Ah: float = 2.3) -> "ECMParameters":
        """Return default NMC 18650 parameters."""
        return cls(R0=0.025, R1=0.015, C1=2000.0, R2=0.030, C2=8000.0,
                   Q_nom_Ah=Q_nom_Ah, chemistry="nmc")

    @classmethod
    def for_lfp(cls, Q_nom_Ah: float = 3.2) -> "ECMParameters":
        """Return default LFP parameters."""
        return cls(R0=0.020, R1=0.012, C1=3000.0, R2=0.025, C2=10000.0,
                   Q_nom_Ah=Q_nom_Ah, chemistry="lfp")

    @classmethod
    def for_lmfp(cls, Q_nom_Ah: float = 3.0) -> "ECMParameters":
        """Return default LMFP (LiMnFePO₄) parameters."""
        return cls(R0=0.018, R1=0.011, C1=2500.0, R2=0.022, C2=9000.0,
                   Q_nom_Ah=Q_nom_Ah, chemistry="lmfp")

    @classmethod
    def for_lto(cls, Q_nom_Ah: float = 3.5) -> "ECMParameters":
        """Return default LTO (Li₄Ti₅O₁₂) parameters. Very low R0 → superb power."""
        return cls(R0=0.008, R1=0.005, C1=5000.0, R2=0.010, C2=15000.0,
                   Q_nom_Ah=Q_nom_Ah, chemistry="lto")

    @classmethod
    def for_nca(cls, Q_nom_Ah: float = 3.0) -> "ECMParameters":
        """Return default NCA (LiNiCoAlO₂) parameters."""
        return cls(R0=0.022, R1=0.013, C1=1800.0, R2=0.028, C2=7500.0,
                   Q_nom_Ah=Q_nom_Ah, chemistry="nca")

    @classmethod
    def for_lmo(cls, Q_nom_Ah: float = 2.0) -> "ECMParameters":
        """Return default LMO (LiMn₂O₄ spinel) parameters."""
        return cls(R0=0.030, R1=0.018, C1=1500.0, R2=0.035, C2=6000.0,
                   Q_nom_Ah=Q_nom_Ah, chemistry="lmo")

    @classmethod
    def for_ssb(cls, Q_nom_Ah: float = 4.0) -> "ECMParameters":
        """Return default Solid-State Battery parameters.

        R0 is higher than liquid-electrolyte cells but rises very steeply at
        cold (Arrhenius_K = 5500 K).  Self-discharge is extremely low (0.3 %/month)
        because there is no liquid electrolyte to enable parasitic reactions.
        """
        return cls(R0=0.030, R1=0.018, C1=1500.0, R2=0.025, C2=5000.0,
                   Q_nom_Ah=Q_nom_Ah, chemistry="ssb",
                   self_discharge_pct_per_month=0.3)

    # ------------------------------------------------------------------
    def as_array(self) -> np.ndarray:
        return np.array([self.R0, self.R1, self.C1, self.R2, self.C2], float)

    @classmethod
    def from_array(cls, arr, Q_nom_Ah: float = 2.3,
                   chemistry: str = "nmc") -> "ECMParameters":
        R0, R1, C1, R2, C2 = arr
        return cls(R0=R0, R1=R1, C1=C1, R2=R2, C2=C2,
                   Q_nom_Ah=Q_nom_Ah, chemistry=chemistry)

    @property
    def tau1(self) -> float:
        return self.R1 * self.C1

    @property
    def tau2(self) -> float:
        return self.R2 * self.C2

    def at_temperature(self, T_C: float) -> "ECMParameters":
        """Return Arrhenius-scaled parameters at *T_C* [°C].

        Resistances scale as exp(Ea/R·(1/T − 1/T_ref)) where Ea/R is
        chemistry-specific (4000 K for NMC, 3500 K for LFP).
        Capacitances are temperature-independent.
        Capacity de-rates 0.2 %/°C below 25 °C.
        """
        T_K = T_C + 273.15
        factor = float(np.exp(self._arrhenius_K * (1.0 / T_K - 1.0 / _T_REF_K)))
        q_scale = 1.0 - 0.002 * max(0.0, 25.0 - T_C)
        return ECMParameters(
            R0=self.R0 * factor,
            R1=self.R1 * factor,
            C1=self.C1,
            R2=self.R2 * factor,
            C2=self.C2,
            Q_nom_Ah=self.Q_nom_Ah * q_scale,
            chemistry=self.chemistry,
            self_discharge_pct_per_month=self.self_discharge_pct_per_month,
        )


@dataclass
class SecondOrderECM:
    """Discrete-time second-order RC equivalent-circuit model."""

    params: ECMParameters = field(default_factory=ECMParameters)
    ocv_curve: OCVSOC = field(default_factory=OCVSOC)
    soc: float = 1.0
    v_rc1: float = 0.0
    v_rc2: float = 0.0

    # ------------------------------------------------------------------
    def reset(self, soc: float = 1.0) -> None:
        self.soc = float(np.clip(soc, 0.0, 1.0))
        self.v_rc1 = 0.0
        self.v_rc2 = 0.0

    def step(self, current: float, dt: float, temperature_C: float = 25.0) -> float:
        """Advance one time-step. `current > 0` denotes discharge.

        Parameters
        ----------
        current : float   Pack current, positive = discharge [A].
        dt : float        Time step [s].
        temperature_C : float
            Cell temperature [°C].  Resistance and capacity are Arrhenius-
            scaled internally; does not affect stored `params`.
        """
        p = self.params.at_temperature(temperature_C) if temperature_C != 25.0 else self.params
        a1 = np.exp(-dt / max(p.tau1, 1e-9))
        a2 = np.exp(-dt / max(p.tau2, 1e-9))

        self.v_rc1 = a1 * self.v_rc1 + (1.0 - a1) * p.R1 * current
        self.v_rc2 = a2 * self.v_rc2 + (1.0 - a2) * p.R2 * current
        # Self-discharge drain (independent of active current)
        sd_per_s = self.params.self_discharge_pct_per_month / (100.0 * 30.0 * 24.0 * 3600.0)
        self.soc = float(np.clip(
            self.soc - current * dt / (p.Q_nom_Ah * 3600.0) - sd_per_s * dt,
            0.0, 1.0,
        ))

        ocv = float(self.ocv_curve.ocv(self.soc, current, T_C=temperature_C))
        return ocv - self.v_rc1 - self.v_rc2 - p.R0 * current

    def simulate(self, current: np.ndarray, dt: float,
                 soc0: float | None = None,
                 temperatures: np.ndarray | None = None) -> dict:
        """Run a full trace and return SOC, V_RC1, V_RC2, V_terminal.

        Parameters
        ----------
        temperatures : (N,) array of float, optional
            Per-step cell temperature [°C].  Defaults to 25 °C throughout.
        """
        if soc0 is not None:
            self.reset(soc0)

        n = len(current)
        T_arr = np.full(n, 25.0) if temperatures is None else np.asarray(temperatures, float)
        soc = np.empty(n)
        vrc1 = np.empty(n)
        vrc2 = np.empty(n)
        v = np.empty(n)
        for k in range(n):
            v[k] = self.step(current[k], dt, temperature_C=float(T_arr[k]))
            soc[k] = self.soc
            vrc1[k] = self.v_rc1
            vrc2[k] = self.v_rc2
        return {"soc": soc, "v_rc1": vrc1, "v_rc2": vrc2, "v_terminal": v}


# ----------------------------------------------------------------------
# Parameter identification
# ----------------------------------------------------------------------
def fit_ecm_parameters(
    current: np.ndarray,
    voltage: np.ndarray,
    dt: float,
    Q_nom_Ah: float = 2.3,
    ocv_curve: OCVSOC | None = None,
    soc0: float | None = None,
    bounds: tuple | None = None,
    initial: ECMParameters | None = None,
) -> tuple[ECMParameters, dict]:
    """Least-squares fit of (R0, R1, C1, R2, C2) to a measured I/V trace."""
    current = np.asarray(current, float)
    voltage = np.asarray(voltage, float)
    if ocv_curve is None:
        ocv_curve = OCVSOC()
    if soc0 is None:
        soc0 = float(ocv_curve.soc(voltage[0]))

    init = (initial or ECMParameters(Q_nom_Ah=Q_nom_Ah)).as_array()
    if bounds is None:
        bounds = (
            np.array([1e-4, 1e-4, 50.0,  1e-4, 200.0]),
            np.array([0.20, 0.20, 5e4,   0.30, 2e5]),
        )

    def residuals(theta: np.ndarray) -> np.ndarray:
        p = ECMParameters.from_array(theta, Q_nom_Ah=Q_nom_Ah)
        model = SecondOrderECM(params=p, ocv_curve=ocv_curve)
        model.reset(soc0)
        sim = model.simulate(current, dt)
        return sim["v_terminal"] - voltage

    res = least_squares(
        residuals, init, bounds=bounds, method="trf",
        x_scale="jac", max_nfev=400,
    )
    fitted = ECMParameters.from_array(res.x, Q_nom_Ah=Q_nom_Ah)
    rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    return fitted, {"success": bool(res.success), "rmse_v": rmse, "result": res, "soc0": soc0}
