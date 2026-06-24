"""
OCV–SOC characteristic for Li-ion cells (NMC or LFP).

The OCV-SOC curve is the open-circuit voltage of a fully relaxed cell as a
function of state of charge. We use a smooth interpolant fit to literature
data points.  A monotonic cubic spline (PCHIP) is used so the inverse map
(V → SOC) is well defined for SOC initialisation.

Temperature correction
----------------------
OCV decreases with temperature.  The coefficient is chemistry-specific:
  • NMC: −0.5 mV/K  (``temp_coeff_V_per_K = −5e-4``)
  • LFP: −0.3 mV/K  (``temp_coeff_V_per_K = −3e-4``)

Use ``OCVSOC.from_chemistry(chemistry)`` to build a correctly configured
curve from a chemistry name or ``CellChemistry`` enum value.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

_T_REF_C: float = 25.0

# Default NMC table (backward compat — matches chemistry._NMC_SOC_OCV)
_DEFAULT_OCV_TABLE = np.array(
    [
        [0.00, 3.000],
        [0.05, 3.350],
        [0.10, 3.520],
        [0.20, 3.610],
        [0.30, 3.660],
        [0.40, 3.710],
        [0.50, 3.760],
        [0.60, 3.820],
        [0.70, 3.890],
        [0.80, 3.960],
        [0.90, 4.060],
        [0.95, 4.130],
        [1.00, 4.200],
    ]
)

# Default NMC temperature coefficient (kept for backward compat).
_TEMP_COEFF_V_PER_K: float = -5e-4


class OCVSOC:
    """Bidirectional OCV ↔ SOC mapping with optional hysteresis.

    Parameters
    ----------
    table : (N, 2) ndarray, optional
        Two-column array of (SOC, OCV).  Defaults to a typical NMC curve.
    hysteresis_v : float, default 0.0
        Symmetric voltage offset added on charge / subtracted on discharge.
        Set ~0.005–0.015 V to reproduce real cell behaviour.
    temp_coeff_V_per_K : float, default -5e-4
        OCV temperature coefficient [V/K].  Use ``-3e-4`` for LFP.
    """

    def __init__(self, table: np.ndarray | None = None, hysteresis_v: float = 0.0,
                 temp_coeff_V_per_K: float = _TEMP_COEFF_V_PER_K):
        table = _DEFAULT_OCV_TABLE if table is None else np.asarray(table, float)
        soc, v = table[:, 0], table[:, 1]
        order = np.argsort(soc)
        self._soc = soc[order]
        self._v = v[order]
        # PCHIP keeps monotonicity → inverse is unambiguous.
        self._ocv_of_soc = PchipInterpolator(self._soc, self._v, extrapolate=True)
        self._soc_of_ocv = PchipInterpolator(self._v, self._soc, extrapolate=True)
        self.hysteresis_v = float(hysteresis_v)
        self._temp_coeff = float(temp_coeff_V_per_K)

    # ------------------------------------------------------------------
    @classmethod
    def from_chemistry(cls, chemistry: str, hysteresis_v: float = 0.0) -> "OCVSOC":
        """Build an OCVSOC configured for *chemistry* (``"nmc"`` or ``"lfp"``)."""
        from .chemistry import get_chemistry_props
        props = get_chemistry_props(chemistry)
        return cls(
            table=props["ocv_table"],
            hysteresis_v=hysteresis_v,
            temp_coeff_V_per_K=props["temp_coeff_V_per_K"],
        )

    # ------------------------------------------------------------------
    def ocv(self, soc: float | np.ndarray,
            current: float | np.ndarray = 0.0,
            T_C: float = 25.0) -> np.ndarray:
        """Open-circuit voltage at a given SOC, hysteresis and temperature.

        Parameters
        ----------
        soc : float or array
            State of charge in [0, 1].
        current : float or array, optional
            Sign determines hysteresis direction (positive = discharge).
        T_C : float, optional
            Cell temperature [°C].  OCV shifts by ``temp_coeff_V_per_K`` per °C
            above 25 °C.
        """
        soc = np.clip(np.asarray(soc, float), 0.0, 1.0)
        v = self._ocv_of_soc(soc)
        if self.hysteresis_v:
            v = v + self.hysteresis_v * np.sign(np.asarray(current, float))
        v = v + self._temp_coeff * (T_C - _T_REF_C)
        return v

    def soc(self, ocv: float | np.ndarray) -> np.ndarray:
        """Inverse map – used for SOC initialisation from a rest voltage."""
        ocv = np.asarray(ocv, float)
        return np.clip(self._soc_of_ocv(ocv), 0.0, 1.0)

    def docv_dsoc(self, soc: float | np.ndarray) -> np.ndarray:
        """Slope dOCV/dSOC – needed by the EKF Jacobian."""
        soc = np.clip(np.asarray(soc, float), 0.0, 1.0)
        return self._ocv_of_soc.derivative()(soc)

    # ------------------------------------------------------------------
    @property
    def table(self) -> np.ndarray:
        return np.column_stack([self._soc, self._v])

    def __repr__(self) -> str:
        return (
            f"OCVSOC(points={len(self._soc)}, "
            f"V[0]={self._v[0]:.3f}, V[1]={self._v[-1]:.3f}, "
            f"hysteresis_v={self.hysteresis_v})"
        )
