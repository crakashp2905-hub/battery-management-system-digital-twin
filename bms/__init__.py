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

__version__ = "0.12.0"

from .aging import AgingModel, AgingParams, AgingState
from .balancing import (
    Balancer,
    InductorBalancer,
    PassiveBalancer,
    SwitchedCapacitorBalancer,
    compare_balancers,
)
from .can import BMSCanBus, CANFrame
from .charging import (
    METHOD_POWER_KW,
    ChargeMethod,
    ChargeProtocol,
    ChargeResult,
    ChargingModel,
    compare_methods,
    plating_c_limit,
)
from .chemistry import CellChemistry, get_chemistry_props
from .control import (
    BMSState,
    BMSSupervisor,
    ContactorState,
    PrechargeContactorSequencer,
    SupervisorConfig,
)
from .data import (
    generate_aging_profile,
    generate_cccv_profile,
    generate_load_profile,
    generate_power_profile,
    load_nasa_like_dataset,
)
from .datasets import (
    DATASET_SOURCES,
    LG_COLUMN_MAP,
    DriveCycleData,
    estimator_leaderboard,
    load_capacity_fade_csv,
    load_drivecycle_csv,
    nasa_mat_to_capacity,
    save_drivecycle_csv,
    soh_curve,
    synthetic_drivecycle,
)
from .diagnostics import compute_crate_map, simulate_eis
from .dva import compute_dva, compute_ica, synthetic_discharge_for_dva
from .ecm import ECMParameters, SecondOrderECM, fit_ecm_parameters
from .estimation import (
    Estimate,
    FunctionSocEstimator,
    RecursiveSocEstimator,
    SocEstimator,
    available_soc_estimators,
    make_soc_estimator,
    register_soc_estimator,
    soc_estimate,
)
from .faults import (
    FaultInjector,
    FaultMode,
    FaultSpec,
    HybridFaultDetector,
    RollingFeatureBuffer,
    extract_features,
)
from .fmea import build_fmea_table, estimate_rul, estimate_rul_with_resistance
from .interpret import (
    FEATURE_NAMES,
    estimator_agreement,
    explain_charge,
    explain_state,
    feature_importances,
    soc_report,
)
from .mechanics import (
    CellMechanicalState,
    MechanicalFaultDetector,
    MechanicalParams,
    PressureModel,
    coulombic_efficiency,
)
from .ocv_soc import OCVSOC
from .pack import BatteryPack, PackConfig
from .passport import BatteryPassport
from .range_predictor import (
    INDIA_CITY_ROUTES,
    INDIA_WEATHER,
    ROUTE_PROFILES,
    VEHICLE_PRESETS,
    RangePrediction,
    RangePredictor,
    RouteSegment,
    VehicleParams,
    WeatherConditions,
)
from .soc_estimators import (
    CoulombCounter,
    EKFEstimator,
    LSTMEstimator,
    UKFEstimator,
    benchmark_estimators,
)
from .soh_estimator import JointEKFSoH
from .sop import SOPConfig, SOPLimit, StateOfPower
from .thermal import PIDController, PredictiveCoolingController, ThermalModel, ThermalParameters

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
    "BMSSupervisor", "BMSState", "SupervisorConfig", "ContactorState",
    "PrechargeContactorSequencer", "SOPConfig", "SOPLimit", "StateOfPower",
    "CANFrame", "BMSCanBus",
    "generate_load_profile", "generate_power_profile",
    "generate_cccv_profile",
    "load_nasa_like_dataset", "generate_aging_profile",
    "BatteryPassport",
    "compute_dva", "compute_ica", "synthetic_discharge_for_dva",
    "simulate_eis", "compute_crate_map",
    "RangePredictor", "VehicleParams", "RouteSegment", "WeatherConditions",
    "RangePrediction", "ROUTE_PROFILES", "VEHICLE_PRESETS",
    "INDIA_CITY_ROUTES", "INDIA_WEATHER",
    "AgingModel", "AgingParams", "AgingState",
    "ChargeMethod", "ChargeProtocol", "ChargeResult", "ChargingModel",
    "METHOD_POWER_KW", "compare_methods", "plating_c_limit",
    "Estimate", "SocEstimator", "RecursiveSocEstimator", "FunctionSocEstimator",
    "soc_estimate", "make_soc_estimator", "available_soc_estimators",
    "register_soc_estimator",
    "JointEKFSoH",
    "FEATURE_NAMES", "feature_importances", "estimator_agreement", "soc_report",
    "explain_state", "explain_charge",
    "PressureModel", "MechanicalParams", "CellMechanicalState",
    "MechanicalFaultDetector", "coulombic_efficiency",
    "DriveCycleData", "synthetic_drivecycle", "load_drivecycle_csv",
    "save_drivecycle_csv", "estimator_leaderboard", "load_capacity_fade_csv",
    "nasa_mat_to_capacity", "soh_curve", "DATASET_SOURCES", "LG_COLUMN_MAP",
]
