"""
BMS Digital Twin — research-grade Battery Management System simulator.

Top-level imports for convenient one-liner use::

    from bms import (
        CellChemistry, get_chemistry_props,
        OCVSOC, SecondOrderECM, ECMParameters,
        BatteryPack, PackConfig,
        ThermalModel, PIDController, PredictiveCoolingController,
        PassiveBalancer, SwitchedCapacitorBalancer, InductorBalancer,
        compare_balancers, CoulombCounter, EKFEstimator, UKFEstimator,
        LSTMEstimator, benchmark_estimators, FaultInjector, FaultMode,
        FaultSpec, HybridFaultDetector, RollingFeatureBuffer,
        build_fmea_table, estimate_rul, estimate_rul_with_resistance,
        BMSSupervisor, generate_load_profile, generate_power_profile,
        generate_cccv_profile, load_nasa_like_dataset, generate_aging_profile,
        BatteryPassport,
        compute_dva, compute_ica, synthetic_discharge_for_dva,
        simulate_eis, compute_crate_map,
        RangePredictor, VehicleParams, RouteSegment, WeatherConditions,
        RangePrediction, ROUTE_PROFILES, VEHICLE_PRESETS,
        INDIA_CITY_ROUTES, INDIA_WEATHER,
    )
"""

__version__ = "0.4.0"

from .chemistry import CellChemistry, get_chemistry_props
from .ocv_soc import OCVSOC
from .ecm import SecondOrderECM, ECMParameters, fit_ecm_parameters
from .pack import BatteryPack, PackConfig
from .thermal import ThermalModel, ThermalParameters, PIDController, PredictiveCoolingController
from .balancing import (
    Balancer,
    PassiveBalancer,
    SwitchedCapacitorBalancer,
    InductorBalancer,
    compare_balancers,
)
from .soc_estimators import (
    CoulombCounter,
    EKFEstimator,
    UKFEstimator,
    LSTMEstimator,
    benchmark_estimators,
)
from .faults import (
    FaultMode,
    FaultSpec,
    FaultInjector,
    HybridFaultDetector,
    RollingFeatureBuffer,
    extract_features,
)
from .fmea import build_fmea_table, estimate_rul, estimate_rul_with_resistance
from .control import BMSSupervisor, BMSState, SupervisorConfig
from .data import (
    generate_load_profile,
    generate_power_profile,
    generate_cccv_profile,
    load_nasa_like_dataset,
    generate_aging_profile,
)
from .passport import BatteryPassport
from .dva import compute_dva, compute_ica, synthetic_discharge_for_dva
from .diagnostics import simulate_eis, compute_crate_map
from .range_predictor import (
    RangePredictor, VehicleParams, RouteSegment, WeatherConditions,
    RangePrediction, ROUTE_PROFILES, VEHICLE_PRESETS,
    INDIA_CITY_ROUTES, INDIA_WEATHER,
)

__all__ = [
    "__version__",
    "CellChemistry", "get_chemistry_props",
    "OCVSOC",
    "SecondOrderECM", "ECMParameters", "fit_ecm_parameters",
    "BatteryPack", "PackConfig",
    "ThermalModel", "ThermalParameters", "PIDController", "PredictiveCoolingController",
    "Balancer", "PassiveBalancer", "SwitchedCapacitorBalancer",
    "InductorBalancer", "compare_balancers",
    "CoulombCounter", "EKFEstimator", "UKFEstimator", "LSTMEstimator",
    "benchmark_estimators",
    "FaultMode", "FaultSpec", "FaultInjector", "HybridFaultDetector",
    "RollingFeatureBuffer", "extract_features",
    "build_fmea_table", "estimate_rul", "estimate_rul_with_resistance",
    "BMSSupervisor", "BMSState", "SupervisorConfig",
    "generate_load_profile", "generate_power_profile",
    "generate_cccv_profile",
    "load_nasa_like_dataset", "generate_aging_profile",
    "BatteryPassport",
    "compute_dva", "compute_ica", "synthetic_discharge_for_dva",
    "simulate_eis", "compute_crate_map",
    "RangePredictor", "VehicleParams", "RouteSegment", "WeatherConditions",
    "RangePrediction", "ROUTE_PROFILES", "VEHICLE_PRESETS",
    "INDIA_CITY_ROUTES", "INDIA_WEATHER",
]
