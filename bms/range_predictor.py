"""
Physics-based EV range predictor with weather, traffic, and battery coupling.

Theory
------
Energy consumption on each route segment is modelled from first principles:

  Traction    = (F_aero + F_roll + F_grade) × v / η_motor
  Stop-and-go = n_stops × ½mv² × (1/η_motor − η_regen)
  HVAC        = Q_thermal / COP(T)        [heating or cooling load]
  Accessories = P_acc × t                 [constant electrical load]

Battery effects
---------------
Two temperature penalties are applied on top of the rated pack energy:
  capacity_factor(T)  — fraction of rated Ah available at battery temp T
  efficiency_factor(T)— extra ohmic heat from Arrhenius-scaled R0

Weather effects
---------------
  Air density ρ(altitude) = ρ₀ × exp(−h/8500)         [aero drag]
  Headwind component v_wind added to vehicle speed in drag equation
  Rolling resistance multiplied by precipitation factor  [rain/snow/ice]
  HVAC load driven by (T_ambient − T_comfort)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_G = 9.81        # m/s²
_RHO0 = 1.225   # kg/m³  sea-level air density
_H_SCALE = 8500.0   # m  atmospheric scale height

# Road-quality modifiers (India-specific, applied to Cr and effective speed)
_RQ_CR_FACTOR: dict[str, float] = {
    "excellent": 0.95, "good": 1.00, "average": 1.15, "poor": 1.35,
}
_RQ_SPEED_FACTOR: dict[str, float] = {
    "excellent": 1.00, "good": 1.00, "average": 0.95, "poor": 0.80,
}


# ── Vehicle parameters ─────────────────────────────────────────────────────
@dataclass
class VehicleParams:
    """Aerodynamic and mechanical parameters of the EV.

    Parameters
    ----------
    mass_kg : float
        Kerb mass (vehicle + battery) [kg].
    drag_coefficient : float
        Aerodynamic drag coefficient Cd [-].
    frontal_area_m2 : float
        Projected frontal area A [m²].
    rolling_resistance : float
        Rolling-resistance coefficient Cr [-].
    regen_efficiency : float
        Fraction of kinetic energy recovered by regenerative braking [-].
    motor_efficiency : float
        Motor + inverter efficiency at typical operating point [-].
    accessory_load_W : float
        Constant auxiliary electrical load (lights, 12 V system, …) [W].
    hvac_max_W : float
        Maximum HVAC (heating + cooling) electrical power [W].
    comfort_temp_C : float
        Cabin setpoint temperature; HVAC activates outside ±3 °C of this [°C].
    """
    mass_kg: float = 2000.0
    drag_coefficient: float = 0.23
    frontal_area_m2: float = 2.4
    rolling_resistance: float = 0.012
    regen_efficiency: float = 0.65
    motor_efficiency: float = 0.92
    accessory_load_W: float = 400.0
    hvac_max_W: float = 4000.0
    comfort_temp_C: float = 21.0

    @classmethod
    def compact(cls) -> "VehicleParams":
        return cls(mass_kg=1650, drag_coefficient=0.29, frontal_area_m2=2.15,
                   rolling_resistance=0.011, hvac_max_W=3000.0)

    @classmethod
    def sedan(cls) -> "VehicleParams":
        return cls(mass_kg=2100, drag_coefficient=0.23, frontal_area_m2=2.40,
                   rolling_resistance=0.012, hvac_max_W=3500.0)

    @classmethod
    def suv(cls) -> "VehicleParams":
        return cls(mass_kg=2600, drag_coefficient=0.35, frontal_area_m2=2.90,
                   rolling_resistance=0.014, hvac_max_W=5000.0)

    @classmethod
    def truck(cls) -> "VehicleParams":
        return cls(mass_kg=3300, drag_coefficient=0.40, frontal_area_m2=3.50,
                   rolling_resistance=0.015, hvac_max_W=6000.0)

    # ── Two-wheeler presets (India market) ─────────────────────────────────
    @classmethod
    def e_scooter(cls) -> "VehicleParams":
        """Ola S1 / Ather 450X class: lightweight urban e-scooter."""
        return cls(mass_kg=130, drag_coefficient=0.65, frontal_area_m2=0.55,
                   rolling_resistance=0.011, regen_efficiency=0.40,
                   motor_efficiency=0.88, accessory_load_W=45.0, hvac_max_W=0.0)

    @classmethod
    def e_motorcycle(cls) -> "VehicleParams":
        """Ola Roadster / Revolt RV400 class: performance electric motorcycle."""
        return cls(mass_kg=185, drag_coefficient=0.56, frontal_area_m2=0.62,
                   rolling_resistance=0.013, regen_efficiency=0.35,
                   motor_efficiency=0.88, accessory_load_W=80.0, hvac_max_W=0.0)

    @classmethod
    def e_moped(cls) -> "VehicleParams":
        """Hero Electric / Ampere Magnus class: low-speed city moped."""
        return cls(mass_kg=85, drag_coefficient=0.82, frontal_area_m2=0.44,
                   rolling_resistance=0.011, regen_efficiency=0.20,
                   motor_efficiency=0.85, accessory_load_W=28.0, hvac_max_W=0.0)


VEHICLE_PRESETS: dict[str, VehicleParams] = {
    "compact":       VehicleParams.compact(),
    "sedan":         VehicleParams.sedan(),
    "suv":           VehicleParams.suv(),
    "truck":         VehicleParams.truck(),
    "e_scooter":     VehicleParams.e_scooter(),
    "e_motorcycle":  VehicleParams.e_motorcycle(),
    "e_moped":       VehicleParams.e_moped(),
}


# ── Route segment ──────────────────────────────────────────────────────────
@dataclass
class RouteSegment:
    """One leg of a route with uniform driving conditions.

    Parameters
    ----------
    distance_km : float
        Length of this leg [km].
    avg_speed_kmh : float
        Free-flow average speed before traffic scaling [km/h].
    grade_pct : float
        Road grade in percent (positive = uphill, negative = downhill).
    traffic_factor : float
        Traffic density scalar in [0, 1]:
        1.0 = free flow, 0.5 = moderate congestion, 0.0 = gridlock.
        Reduces effective speed and increases stop-and-go cycling.
    name : str
        Human-readable label shown in charts.
    """
    distance_km: float
    avg_speed_kmh: float
    grade_pct: float = 0.0
    traffic_factor: float = 1.0
    name: str = ""
    road_quality: str = "good"   # "excellent", "good", "average", "poor"

    @property
    def effective_speed_kmh(self) -> float:
        return float(self.avg_speed_kmh * float(np.clip(self.traffic_factor, 0.0, 1.0)))

    @property
    def grade_rad(self) -> float:
        return float(np.arctan(self.grade_pct / 100.0))


# ── Weather conditions ──────────────────────────────────────────────────────
@dataclass
class WeatherConditions:
    """Ambient weather affecting aerodynamics, rolling resistance, HVAC, and battery.

    Parameters
    ----------
    temperature_C : float
        Ambient air temperature [°C].  Also used as the initial battery temperature.
    wind_speed_ms : float
        Wind speed magnitude [m/s].
    wind_heading_deg : float
        Angle between wind vector and vehicle heading [°].
        0° = pure headwind, 90° = crosswind, 180° = pure tailwind.
    precipitation : str
        Road surface condition: ``"none"``, ``"rain"``, ``"snow"``, ``"ice"``.
    altitude_m : float
        Elevation above sea level [m].  Reduces air density and aero drag.
    """
    temperature_C: float = 20.0
    wind_speed_ms: float = 0.0
    wind_heading_deg: float = 0.0
    precipitation: str = "none"
    altitude_m: float = 0.0

    @property
    def air_density_kgm3(self) -> float:
        return float(_RHO0 * np.exp(-self.altitude_m / _H_SCALE))

    @property
    def headwind_ms(self) -> float:
        """Component of wind opposing vehicle motion (positive = headwind)."""
        return float(self.wind_speed_ms * np.cos(np.radians(self.wind_heading_deg)))

    @property
    def rolling_factor(self) -> float:
        """Cr multiplier from road surface condition."""
        return {"none": 1.00, "rain": 1.25, "snow": 1.70, "ice": 1.50}.get(
            self.precipitation, 1.0)

    @property
    def speed_factor(self) -> float:
        """Maximum speed fraction imposed by weather conditions."""
        return {"none": 1.00, "rain": 0.90, "snow": 0.65, "ice": 0.50}.get(
            self.precipitation, 1.0)

    @classmethod
    def mild(cls) -> "WeatherConditions":
        return cls(temperature_C=20.0, wind_speed_ms=0.0)

    @classmethod
    def hot_summer(cls) -> "WeatherConditions":
        return cls(temperature_C=38.0, wind_speed_ms=4.0, wind_heading_deg=30.0)

    @classmethod
    def cold_winter(cls) -> "WeatherConditions":
        return cls(temperature_C=-15.0, wind_speed_ms=6.0, wind_heading_deg=10.0,
                   precipitation="snow")

    @classmethod
    def rainy(cls) -> "WeatherConditions":
        return cls(temperature_C=12.0, wind_speed_ms=5.0, wind_heading_deg=45.0,
                   precipitation="rain")

    @classmethod
    def mountain_pass(cls) -> "WeatherConditions":
        return cls(temperature_C=5.0, wind_speed_ms=8.0, wind_heading_deg=20.0,
                   altitude_m=1800.0)

    # ── India-specific weather presets ──────────────────────────────────────
    @classmethod
    def india_summer(cls) -> "WeatherConditions":
        """Peak summer (Apr–Jun) across North India / Deccan plateau (≈42 °C)."""
        return cls(temperature_C=42.0, wind_speed_ms=3.0,
                   wind_heading_deg=30.0, altitude_m=200.0)

    @classmethod
    def india_monsoon(cls) -> "WeatherConditions":
        """Monsoon season (Jul–Sep): warm, wet, gusty (≈28 °C, heavy rain)."""
        return cls(temperature_C=28.0, wind_speed_ms=8.0,
                   wind_heading_deg=45.0, precipitation="rain")

    @classmethod
    def india_winter_north(cls) -> "WeatherConditions":
        """North Indian winter (Dec–Feb): cool mornings, fog (≈8 °C)."""
        return cls(temperature_C=8.0, wind_speed_ms=5.0,
                   wind_heading_deg=30.0, altitude_m=200.0)

    @classmethod
    def india_coastal(cls) -> "WeatherConditions":
        """Coastal cities (Chennai/Mumbai) year-round: hot + sea breeze crosswind."""
        return cls(temperature_C=33.0, wind_speed_ms=9.0,
                   wind_heading_deg=90.0)   # crosswind — reduces drag benefit


# ── Predefined drive-cycle route profiles ──────────────────────────────────
ROUTE_PROFILES: dict[str, list[RouteSegment]] = {
    "wltp": [
        RouteSegment(13.7, 25.0, 0.0, 0.65, "WLTP Urban"),
        RouteSegment(13.5, 44.5, 0.0, 0.85, "WLTP Suburban"),
        RouteSegment(18.9, 70.0, 0.0, 1.00, "WLTP Rural"),
        RouteSegment(23.3, 120.0, 0.0, 1.00, "WLTP Highway"),
    ],
    "city": [
        RouteSegment(5.0, 20.0, 0.0, 0.45, "Dense Urban"),
        RouteSegment(12.0, 35.0, 0.0, 0.65, "Urban Arterial"),
        RouteSegment(5.0, 20.0, 0.0, 0.45, "Dense Urban Return"),
    ],
    "highway": [
        RouteSegment(2.0, 80.0, 0.0, 0.90, "Ramp Merge"),
        RouteSegment(45.0, 120.0, 0.0, 1.00, "Highway Cruise"),
        RouteSegment(2.0, 80.0, 0.0, 0.90, "Exit Ramp"),
    ],
    "mixed": [
        RouteSegment(5.0, 30.0, 0.0, 0.60, "City Start"),
        RouteSegment(15.0, 80.0, 1.5, 0.95, "Rural Uphill"),
        RouteSegment(5.0, 80.0, -1.5, 1.00, "Rural Downhill"),
        RouteSegment(10.0, 110.0, 0.0, 1.00, "Highway"),
        RouteSegment(3.0, 40.0, 0.0, 0.70, "Suburban Finish"),
    ],
    "mountain": [
        RouteSegment(5.0, 50.0, 3.5, 0.90, "Foothills Approach"),
        RouteSegment(10.0, 55.0, 6.0, 0.85, "Mountain Ascent"),
        RouteSegment(3.0, 35.0, 9.0, 0.75, "Steep Summit Push"),
        RouteSegment(12.0, 65.0, -5.5, 0.92, "Mountain Descent (regen)"),
        RouteSegment(5.0, 80.0, -2.0, 0.95, "Valley Exit"),
    ],
    # ── India-specific drive cycles ──────────────────────────────────────
    "midc": [
        # Modified Indian Drive Cycle — IDC urban phase + extra-urban
        RouteSegment(5.6, 19.4, 0.0, 0.40, "MIDC Urban (IDC)", road_quality="average"),
        RouteSegment(14.1, 50.0, 0.0, 0.80, "MIDC Extra-Urban", road_quality="good"),
    ],
    "india_nh": [
        RouteSegment(8.0, 32.0, 0.0, 0.45, "City Egress", road_quality="average"),
        RouteSegment(5.0, 60.0, 0.0, 0.75, "State Highway Approach", road_quality="good"),
        RouteSegment(50.0, 100.0, 0.0, 0.90, "National Highway Cruise", road_quality="excellent"),
        RouteSegment(5.0, 60.0, 0.0, 0.75, "State Highway Exit", road_quality="good"),
        RouteSegment(7.0, 28.0, 0.0, 0.40, "City Ingress", road_quality="average"),
    ],
}


# ── India city route presets ───────────────────────────────────────────────
INDIA_CITY_ROUTES: dict[str, list[RouteSegment]] = {
    "delhi_ncr": [
        RouteSegment(5.0, 22.0, 0.0, 0.35, "Ring Road (dense)", road_quality="average"),
        RouteSegment(8.0, 55.0, 0.0, 0.60, "NH-48 Approach", road_quality="good"),
        RouteSegment(10.0, 75.0, 0.0, 0.72, "DND Flyway", road_quality="excellent"),
        RouteSegment(2.0, 18.0, 0.0, 0.30, "Local Streets", road_quality="poor"),
    ],
    "mumbai": [
        RouteSegment(3.0, 16.0, 0.0, 0.28, "Local Roads (BKC area)", road_quality="average"),
        RouteSegment(12.0, 42.0, 0.0, 0.50, "Western Express Highway", road_quality="good"),
        RouteSegment(5.0, 68.0, 0.0, 0.82, "Mumbai Coastal Road", road_quality="excellent"),
    ],
    "bangalore": [
        RouteSegment(4.0, 20.0, 0.0, 0.33, "Whitefield Local", road_quality="poor"),
        RouteSegment(15.0, 50.0, 0.0, 0.55, "Outer Ring Road", road_quality="good"),
        RouteSegment(6.0, 32.0, 0.0, 0.45, "Sarjapur Arterial", road_quality="average"),
    ],
    "chennai": [
        RouteSegment(5.0, 26.0, 0.0, 0.45, "City Arterial", road_quality="average"),
        RouteSegment(20.0, 62.0, 0.0, 0.65, "Old Mahabalipuram Road (OMR)", road_quality="good"),
        RouteSegment(5.0, 22.0, 0.0, 0.40, "Perungudi Local", road_quality="average"),
    ],
    "pune": [
        RouteSegment(3.0, 24.0, 0.0, 0.50, "MIDC Industrial Zone", road_quality="average"),
        RouteSegment(8.0, 38.0, 0.5, 0.55, "Baner Road (hilly)", road_quality="average"),
        RouteSegment(9.0, 78.0, 0.0, 0.80, "NH-48 Pune-Mumbai", road_quality="excellent"),
    ],
    "hyderabad": [
        RouteSegment(3.0, 28.0, 0.0, 0.55, "Gachibowli Financial District", road_quality="good"),
        RouteSegment(15.0, 88.0, 0.0, 0.82, "PVNR Expressway", road_quality="excellent"),
        RouteSegment(7.0, 38.0, 0.0, 0.50, "Inner Ring Road", road_quality="average"),
    ],
    "kolkata": [
        RouteSegment(2.0, 14.0, 0.0, 0.28, "Park Street / Central", road_quality="poor"),
        RouteSegment(12.0, 52.0, 0.0, 0.60, "EM Bypass", road_quality="good"),
        RouteSegment(6.0, 18.0, 0.0, 0.32, "Gariahat to Shyambazar", road_quality="average"),
    ],
}


# ── India seasonal weather lookup ──────────────────────────────────────────
# Key format: "<season>_<region>"  (season: summer/monsoon/winter/spring)
# region: north, south_deccan, coastal, hilly
INDIA_WEATHER: dict[str, WeatherConditions] = {
    "summer_north":      WeatherConditions(temperature_C=42.0, wind_speed_ms=3.0,
                                           wind_heading_deg=30.0, altitude_m=200.0),
    "monsoon_north":     WeatherConditions(temperature_C=28.0, wind_speed_ms=9.0,
                                           wind_heading_deg=45.0, precipitation="rain"),
    "winter_north":      WeatherConditions(temperature_C=8.0, wind_speed_ms=5.0,
                                           wind_heading_deg=30.0, altitude_m=200.0),
    "spring_north":      WeatherConditions(temperature_C=28.0, wind_speed_ms=4.0),
    "summer_south_deccan": WeatherConditions(temperature_C=36.0, wind_speed_ms=4.0,
                                             altitude_m=900.0),   # Bangalore plateau
    "monsoon_south_deccan": WeatherConditions(temperature_C=22.0, wind_speed_ms=7.0,
                                              precipitation="rain", altitude_m=900.0),
    "winter_south_deccan": WeatherConditions(temperature_C=18.0, wind_speed_ms=3.0,
                                             altitude_m=900.0),
    "spring_south_deccan": WeatherConditions(temperature_C=30.0, wind_speed_ms=3.0,
                                             altitude_m=900.0),
    "summer_coastal":    WeatherConditions(temperature_C=34.0, wind_speed_ms=9.0,
                                           wind_heading_deg=90.0),
    "monsoon_coastal":   WeatherConditions(temperature_C=27.0, wind_speed_ms=13.0,
                                           wind_heading_deg=45.0, precipitation="rain"),
    "winter_coastal":    WeatherConditions(temperature_C=24.0, wind_speed_ms=5.0,
                                           wind_heading_deg=90.0),
    "spring_coastal":    WeatherConditions(temperature_C=32.0, wind_speed_ms=7.0,
                                           wind_heading_deg=90.0),
    "summer_hilly":      WeatherConditions(temperature_C=22.0, wind_speed_ms=5.0,
                                           altitude_m=1500.0),
    "monsoon_hilly":     WeatherConditions(temperature_C=16.0, wind_speed_ms=8.0,
                                           precipitation="rain", altitude_m=1500.0),
    "winter_hilly":      WeatherConditions(temperature_C=4.0, wind_speed_ms=6.0,
                                           altitude_m=1500.0),
    "spring_hilly":      WeatherConditions(temperature_C=18.0, wind_speed_ms=4.0,
                                           altitude_m=1500.0),
}


# ── Prediction result types ────────────────────────────────────────────────
@dataclass
class SegmentResult:
    name: str
    distance_km: float
    energy_traction_Wh: float
    energy_regen_Wh: float
    energy_hvac_Wh: float
    energy_accessories_Wh: float
    soc_end: float
    efficiency_Whkm: float           # net Wh drawn per km


@dataclass
class RangePrediction:
    """Full result from :meth:`RangePredictor.predict`.

    Attributes
    ----------
    estimated_range_km : float
        Distance covered before SOC reaches the reserve.
    route_completable : bool
        True if the battery survives the entire requested route.
    soc_at_destination : float or None
        Remaining SOC if the route is completable, else None.
    usable_energy_Wh : float
        Effective energy available after temperature and SOC-window derating.
    total_consumed_Wh : float
        Net energy drawn from the battery over the completed distance.
    avg_efficiency_Whkm : float
        Total net energy per km.
    segment_results : list[SegmentResult]
        Per-segment breakdown.
    energy_breakdown : dict
        Keys: ``"Traction"``, ``"Stop-Go"``, ``"HVAC"``, ``"Accessories"``,
        ``"Regeneration"`` (negative = saved energy).
    weather_penalty_pct : float
        Extra energy vs. mild reference weather (20 °C, no wind, dry road) [%].
    battery_temp_C : float
        Battery temperature used for capacity / efficiency derating.
    """
    estimated_range_km: float
    route_completable: bool
    soc_at_destination: Optional[float]
    usable_energy_Wh: float
    total_consumed_Wh: float
    avg_efficiency_Whkm: float
    segment_results: list[SegmentResult]
    energy_breakdown: dict[str, float]
    weather_penalty_pct: float
    battery_temp_C: float


# ── Core predictor ─────────────────────────────────────────────────────────
class RangePredictor:
    """Physics-based EV range predictor.

    Parameters
    ----------
    vehicle : VehicleParams, optional
        Vehicle specification.  Defaults to a mid-size sedan.

    Examples
    --------
    ::

        from bms import RangePredictor, WeatherConditions, ROUTE_PROFILES
        pred = RangePredictor()
        result = pred.predict(80_000.0, "nmc", ROUTE_PROFILES["wltp"],
                              WeatherConditions.cold_winter())
        print(f"Range: {result.estimated_range_km:.0f} km")
    """

    SOC_RESERVE = 0.10   # battery reserved (not used for driving)
    _HVAC_COP_HEAT_REF = 2.5   # heat-pump COP at mild cold (0 °C)
    _HVAC_COP_COOL = 3.0       # AC COP (relatively temperature-independent)
    _STOPS_PER_KM_HEAVY = 3.0  # stop rate at traffic_factor=0 (gridlock)

    def __init__(self, vehicle: VehicleParams | None = None) -> None:
        self.vehicle = vehicle or VehicleParams.sedan()

    # ── Internal helpers ────────────────────────────────────────────────────
    def _capacity_factor(self, T_C: float) -> float:
        """Fraction of rated Ah usable at temperature T_C.

        Cold: −0.57 %/°C below 15 °C, floored at 75 % below −30 °C.
        Hot: −0.4 %/°C above 35 °C (accelerated degradation window).
        Both calibrated to match real-world Norwegian/US winter EV data.
        """
        if 15.0 <= T_C <= 35.0:
            return 1.0
        if T_C < 15.0:
            return float(max(0.75, 1.0 - 0.0057 * (15.0 - T_C)))
        return float(max(0.90, 1.0 - 0.004 * (T_C - 35.0)))

    def _battery_efficiency_overhead(self, T_C: float, chemistry: str) -> float:
        """Extra fractional energy overhead from increased R0 at cold temperature.

        Uses a log-scale model: 8 % overhead per decade of Arrhenius resistance
        increase.  This accounts for BMS current-limiting at cold (which prevents
        the extreme quadratic I²R blowup) and is calibrated to give ~8 % overhead
        for NMC at −20 °C, ~11 % for SSB.  Warm temperatures (r_factor < 1)
        return 0 — lower resistance means no penalty.
        """
        from .chemistry import get_chemistry_props
        Ea_R = get_chemistry_props(chemistry)["arrhenius_K"]
        T_K = T_C + 273.15
        r_factor = float(np.exp(Ea_R * (1.0 / T_K - 1.0 / 298.15)))
        if r_factor <= 1.0:
            return 0.0
        return float(min(np.log10(r_factor) * 0.08, 0.25))

    def _hvac_power_W(self, T_ambient_C: float) -> float:
        """Electrical power consumed by HVAC [W]."""
        v = self.vehicle
        dead_band = 3.0   # ± 3 °C comfort dead-band
        T_lo = v.comfort_temp_C - dead_band
        T_hi = v.comfort_temp_C + dead_band

        if T_ambient_C < T_lo:
            delta = T_lo - T_ambient_C
            cop = max(1.0, self._HVAC_COP_HEAT_REF - 0.06 * delta)
            q_heat = min(delta * 180.0, v.hvac_max_W * cop)
            return float(q_heat / cop)
        if T_ambient_C > T_hi:
            delta = T_ambient_C - T_hi
            q_cool = min(delta * 140.0, v.hvac_max_W * self._HVAC_COP_COOL)
            return float(q_cool / self._HVAC_COP_COOL)
        return 0.0

    def _segment_energy_Wh(
        self,
        seg: RouteSegment,
        weather: WeatherConditions,
    ) -> tuple[float, float, float, float, float]:
        """Return (traction, stopgo, regen, hvac, accessories) energy [Wh]."""
        v = self.vehicle
        rho = weather.air_density_kgm3

        # Road quality factors (India-specific; default "good" = 1.0 multiplier)
        rq_cr = _RQ_CR_FACTOR.get(seg.road_quality, 1.0)
        rq_spd = _RQ_SPEED_FACTOR.get(seg.road_quality, 1.0)

        # Effective speed [m/s] after traffic, precipitation, and road quality
        v_kmh = seg.effective_speed_kmh * weather.speed_factor * rq_spd
        v_ms = float(max(v_kmh / 3.6, 0.5))   # minimum 0.5 m/s to avoid /0
        d_m = seg.distance_km * 1000.0
        t_s = d_m / v_ms

        theta = seg.grade_rad
        v_head = weather.headwind_ms
        v_rel = max(v_ms + v_head, 0.0)   # relative air speed [m/s]

        # ── Forces ──────────────────────────────────────────────────────────
        F_aero = 0.5 * rho * v.drag_coefficient * v.frontal_area_m2 * v_rel ** 2
        F_roll = (v.rolling_resistance * weather.rolling_factor * rq_cr
                  * v.mass_kg * _G * float(np.cos(theta)))
        F_grade = v.mass_kg * _G * float(np.sin(theta))

        # ── Constant-speed traction / regen ─────────────────────────────────
        P_net_W = (F_aero + F_roll + F_grade) * v_ms

        if P_net_W >= 0.0:
            E_traction = P_net_W * t_s / 3600.0 / v.motor_efficiency
            E_regen_downhill = 0.0
        else:
            E_traction = 0.0
            E_regen_downhill = abs(P_net_W) * t_s / 3600.0 * v.regen_efficiency

        # ── Stop-and-go energy ───────────────────────────────────────────────
        # stops/km = (1 - traffic_factor) × max_stop_rate
        n_stops = (1.0 - float(np.clip(seg.traffic_factor, 0.0, 1.0))
                   ) * self._STOPS_PER_KM_HEAVY * seg.distance_km
        E_kinetic_per_stop = 0.5 * v.mass_kg * v_ms ** 2 / 3600.0  # Wh
        # Net battery cost per stop: accelerate (E_k/η_motor) − regen (E_k×η_regen)
        E_stopgo = n_stops * E_kinetic_per_stop * (
            1.0 / v.motor_efficiency - v.regen_efficiency)

        # ── HVAC and accessories ─────────────────────────────────────────────
        E_hvac = self._hvac_power_W(weather.temperature_C) * t_s / 3600.0
        E_acc = v.accessory_load_W * t_s / 3600.0

        return (
            float(E_traction),
            float(max(E_stopgo, 0.0)),
            float(E_regen_downhill),
            float(E_hvac),
            float(E_acc),
        )

    # ── Public API ──────────────────────────────────────────────────────────
    def predict(
        self,
        pack_energy_Wh: float,
        chemistry: str,
        route: list[RouteSegment],
        weather: WeatherConditions | None = None,
        initial_soc: float = 1.0,
    ) -> RangePrediction:
        """Simulate the route segment-by-segment and return a full prediction.

        Parameters
        ----------
        pack_energy_Wh : float
            Rated battery pack energy at 25 °C [Wh].
        chemistry : str
            Cell chemistry string (``"nmc"``, ``"ssb"``, …).
        route : list[RouteSegment]
            Ordered list of route segments.
        weather : WeatherConditions, optional
            Ambient conditions.  Defaults to mild (20 °C, no wind).
        initial_soc : float
            Starting state-of-charge [0, 1].  Defaults to 1.0.
        """
        if weather is None:
            weather = WeatherConditions.mild()

        bat_T = weather.temperature_C
        cap_f = self._capacity_factor(bat_T)
        eff_oh = self._battery_efficiency_overhead(bat_T, chemistry)

        # Usable energy after all derating
        usable_Wh = (pack_energy_Wh * cap_f
                     * (initial_soc - self.SOC_RESERVE)
                     / (1.0 + eff_oh))
        remaining_Wh = max(usable_Wh, 0.0)
        soc = initial_soc

        # Mild reference: compute total mild energy for weather penalty
        mild = WeatherConditions.mild()
        mild_total = sum(
            sum(self._segment_energy_Wh(s, mild)[:2])           # traction + stopgo
            - self._segment_energy_Wh(s, mild)[2]               # − regen
            + sum(self._segment_energy_Wh(s, mild)[3:])         # + hvac + acc
            for s in route
        )

        seg_results: list[SegmentResult] = []
        total_traction = total_stopgo = total_regen = 0.0
        total_hvac = total_acc = 0.0
        completed_km = 0.0
        route_done = True

        for seg in route:
            E_tr, E_sg, E_re, E_hv, E_ac = self._segment_energy_Wh(seg, weather)
            E_net = E_tr + E_sg + E_hv + E_ac - E_re

            if E_net > remaining_Wh and E_net > 0:
                # Battery runs out mid-segment — compute partial distance
                frac = float(remaining_Wh / E_net)
                E_tr *= frac; E_sg *= frac; E_re *= frac
                E_hv *= frac; E_ac *= frac
                partial_km = seg.distance_km * frac
                E_net = remaining_Wh
                remaining_Wh = 0.0
                soc = self.SOC_RESERVE
                route_done = False
            else:
                partial_km = seg.distance_km
                remaining_Wh -= max(E_net, 0.0)
                # Recompute SOC from remaining energy
                usable_full = pack_energy_Wh * cap_f * (initial_soc - self.SOC_RESERVE)
                soc = float(self.SOC_RESERVE + remaining_Wh * (1.0 + eff_oh)
                            / max(pack_energy_Wh * cap_f, 1e-6))
                soc = float(np.clip(soc, 0.0, 1.0))

            completed_km += partial_km
            total_traction += E_tr
            total_stopgo += E_sg
            total_regen += E_re
            total_hvac += E_hv
            total_acc += E_ac

            net_for_seg = E_tr + E_sg + E_hv + E_ac - E_re
            seg_results.append(SegmentResult(
                name=seg.name,
                distance_km=partial_km,
                energy_traction_Wh=E_tr,
                energy_regen_Wh=E_re + E_sg * self.vehicle.regen_efficiency,
                energy_hvac_Wh=E_hv,
                energy_accessories_Wh=E_ac,
                soc_end=soc,
                efficiency_Whkm=(net_for_seg / max(partial_km, 1e-6)),
            ))

            if not route_done:
                break

        total_net = total_traction + total_stopgo + total_hvac + total_acc - total_regen
        weather_penalty = ((total_net - mild_total) / max(mild_total, 1.0)) * 100.0

        return RangePrediction(
            estimated_range_km=float(completed_km),
            route_completable=route_done,
            soc_at_destination=float(soc) if route_done else None,
            usable_energy_Wh=float(usable_Wh),
            total_consumed_Wh=float(total_net),
            avg_efficiency_Whkm=(float(total_net / max(completed_km, 1e-6))),
            segment_results=seg_results,
            energy_breakdown={
                "Traction":    float(total_traction),
                "Stop-Go":     float(total_stopgo),
                "HVAC":        float(total_hvac),
                "Accessories": float(total_acc),
                "Regeneration": float(-total_regen),
            },
            weather_penalty_pct=float(weather_penalty),
            battery_temp_C=float(bat_T),
        )

    def predict_max_range_km(
        self,
        pack_energy_Wh: float,
        chemistry: str,
        profile: str = "wltp",
        weather: WeatherConditions | None = None,
        initial_soc: float = 1.0,
    ) -> float:
        """Estimate maximum range by scaling a predefined drive-cycle profile.

        The cycle is run once to get energy-per-km, then the result is
        extrapolated to the full usable battery energy.

        Parameters
        ----------
        pack_energy_Wh : float
            Pack energy at 25 °C [Wh].
        chemistry : str
            Cell chemistry.
        profile : str
            Key in :data:`ROUTE_PROFILES`.
        weather : WeatherConditions, optional
            Ambient conditions.  Defaults to mild.
        initial_soc : float
            Starting SOC.
        """
        if weather is None:
            weather = WeatherConditions.mild()
        route = ROUTE_PROFILES.get(profile, ROUTE_PROFILES["wltp"])

        bat_T = weather.temperature_C
        cap_f = self._capacity_factor(bat_T)
        eff_oh = self._battery_efficiency_overhead(bat_T, chemistry)
        usable_Wh = (pack_energy_Wh * cap_f
                     * (initial_soc - self.SOC_RESERVE)
                     / (1.0 + eff_oh))

        cycle_km = sum(s.distance_km for s in route)
        cycle_net = 0.0
        for seg in route:
            E_tr, E_sg, E_re, E_hv, E_ac = self._segment_energy_Wh(seg, weather)
            cycle_net += E_tr + E_sg + E_hv + E_ac - E_re

        Wh_per_km = cycle_net / max(cycle_km, 1e-6)
        return float(usable_Wh / max(Wh_per_km, 1e-6))

    def temperature_range_sweep(
        self,
        pack_energy_Wh: float,
        chemistry: str,
        profile: str = "wltp",
        temperatures_C: list[float] | None = None,
    ) -> dict[float, float]:
        """Return {temperature_C: range_km} for the given temperature list.

        Useful for plotting a range-vs-temperature curve to show cold/hot penalties.
        """
        if temperatures_C is None:
            temperatures_C = list(range(-30, 51, 5))
        return {
            T: self.predict_max_range_km(
                pack_energy_Wh, chemistry, profile,
                WeatherConditions(temperature_C=T),
            )
            for T in temperatures_C
        }
