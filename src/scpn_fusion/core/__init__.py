# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Core Package Init
try:
    from ._rust_compat import FusionKernel, RUST_BACKEND
except ImportError:
    from .fusion_kernel import FusionKernel

    RUST_BACKEND = False
from .fusion_ignition_sim import FusionBurnPhysics
from .equilibrium_3d import FourierMode3D, VMECStyleEquilibrium3D
from .fieldline_3d import FieldLineTrace3D, FieldLineTracer3D, PoincareSection3D
from .heat_ml_shadow_surrogate import (
    HeatMLShadowSurrogate,
    ShadowDataset,
    benchmark_inference_seconds,
    generate_shadow_dataset,
    rmse_percent as shadow_rmse_percent,
    synthetic_shadow_reference,
)
from .gpu_runtime import EquilibriumLatencyBenchmark, GPURuntimeBridge, RuntimeBenchmark
from .gyro_swin_surrogate import (
    GyroSwinLikeSurrogate,
    SpeedBenchmark,
    TurbulenceDataset,
    benchmark_speedup,
    generate_synthetic_gyrokinetic_dataset,
    gene_proxy_predict,
    rmse_percent,
    synthetic_core_turbulence_target,
)
from .pretrained_surrogates import (
    PretrainedMLPSurrogate,
    bundle_pretrained_surrogates,
    evaluate_pretrained_fno,
    evaluate_pretrained_mlp,
    load_pretrained_mlp,
    save_pretrained_mlp,
)
from .integrated_transport_solver import _load_gyro_bohm_coefficient as load_gyro_bohm_coefficient
from .scaling_laws import (
    TransportBenchmarkResult,
    assess_ipb98y2_domain,
    compute_h_factor,
    ipb98y2_tau_e,
    ipb98y2_tau_e_with_metadata,
    load_ipb98y2_coefficients,
)
from .stability_mhd import (
    QProfile,
    MercierResult,
    BallooningResult,
    KruskalShafranovResult,
    TroyonResult,
    NTMResult,
    StabilitySummary,
    compute_q_profile,
    mercier_stability,
    ballooning_stability,
    kruskal_shafranov_stability,
    troyon_beta_limit,
    ntm_stability,
    run_full_stability_check,
)

# Additional modules available via lazy import to avoid circular deps
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "CompactReactorArchitect": (".compact_reactor_optimizer", "CompactReactorArchitect"),
    "ReactorConfig": (".config_schema", "ReactorConfig"),
    "DivertorLab": (".divertor_thermal_sim", "DivertorLab"),
    "EpedPedestalModel": (".eped_pedestal", "EpedPedestalModel"),
    "PedestalResult": (".eped_pedestal", "PedestalResult"),
    "GEqdsk": (".eqdsk", "GEqdsk"),
    "read_geqdsk": (".eqdsk", "read_geqdsk"),
    "write_geqdsk": (".eqdsk", "write_geqdsk"),
    "GlobalDesignExplorer": (".global_design_scanner", "GlobalDesignExplorer"),
    "NeuralEqConfig": (".neural_equilibrium", "NeuralEqConfig"),
    "NeuralTransportModel": (".neural_transport", "NeuralTransportModel"),
    "RFHeatingSystem": (".rf_heating", "RFHeatingSystem"),
    "ECRHHeatingSystem": (".rf_heating", "ECRHHeatingSystem"),
    "StabilityAnalyzer": (".stability_analyzer", "StabilityAnalyzer"),
    "FusionState": (".state_space", "FusionState"),
    "PlasmaScenario": (".uncertainty", "PlasmaScenario"),
    "UQResult": (".uncertainty", "UQResult"),
    "WholeDeviceModel": (".wdm_engine", "WholeDeviceModel"),
    # Current diffusion — Sauter neoclassical resistivity, Crank-Nicolson solver
    "CurrentDiffusionSolver": (".current_diffusion", "CurrentDiffusionSolver"),
    "neoclassical_resistivity": (".current_diffusion", "neoclassical_resistivity"),
    # Current drive — ECCD, NBI, LHCD sources
    "ECCDSource": (".current_drive", "ECCDSource"),
    "NBISource": (".current_drive", "NBISource"),
    "LHCDSource": (".current_drive", "LHCDSource"),
    "CurrentDriveMix": (".current_drive", "CurrentDriveMix"),
    # Sawtooth — Porcelli trigger + Kadomtsev reconnection
    "SawtoothCycler": (".sawtooth", "SawtoothCycler"),
    "SawtoothMonitor": (".sawtooth", "SawtoothMonitor"),
    "kadomtsev_crash": (".sawtooth", "kadomtsev_crash"),
    # NTM dynamics — Modified Rutherford Equation
    "NTMIslandDynamics": (".ntm_dynamics", "NTMIslandDynamics"),
    "NTMController": (".ntm_dynamics", "NTMController"),
    "find_rational_surfaces": (".ntm_dynamics", "find_rational_surfaces"),
    # SOL two-point model — Eich scaling
    "TwoPointSOL": (".sol_model", "TwoPointSOL"),
    "eich_heat_flux_width": (".sol_model", "eich_heat_flux_width"),
    # Phase 5 — Impurity transport (Hirshman & Sigmar 1981)
    "ImpurityTransportSolver": (".impurity_transport", "ImpurityTransportSolver"),
    "neoclassical_impurity_pinch": (".impurity_transport", "neoclassical_impurity_pinch"),
    "total_radiated_power": (".impurity_transport", "total_radiated_power"),
    # Phase 5 — Momentum transport (Waltz ExB shearing)
    "MomentumTransportSolver": (".momentum_transport", "MomentumTransportSolver"),
    "exb_shearing_rate": (".momentum_transport", "exb_shearing_rate"),
    "turbulence_suppression_factor": (".momentum_transport", "turbulence_suppression_factor"),
    # Phase 5 — Runaway electrons (Connor & Hastie 1975, Rosenbluth & Putvinski 1997)
    "RunawayEvolution": (".runaway_electrons", "RunawayEvolution"),
    "RunawayParams": (".runaway_electrons", "RunawayParams"),
    "dreicer_field": (".runaway_electrons", "dreicer_field"),
    "critical_field": (".runaway_electrons", "critical_field"),
    # Phase 5 — Alfven eigenmodes (TAE/RSAE)
    "AlfvenContinuum": (".alfven_eigenmodes", "AlfvenContinuum"),
    "AlfvenStabilityAnalysis": (".alfven_eigenmodes", "AlfvenStabilityAnalysis"),
    # Phase 5 — ELM model (peeling-ballooning + Chirikov)
    "ELMCycler": (".elm_model", "ELMCycler"),
    "PeelingBallooningBoundary": (".elm_model", "PeelingBallooningBoundary"),
    "RMPSuppression": (".elm_model", "RMPSuppression"),
    # Phase 5 — Pellet injection (Parks & Turnbull 1978)
    "PelletTrajectory": (".pellet_injection", "PelletTrajectory"),
    "PelletFuelingController": (".pellet_injection", "PelletFuelingController"),
    "ngs_ablation_rate": (".pellet_injection", "ngs_ablation_rate"),
    # Phase 5 — Plasma wall interaction (Eckstein sputtering)
    "SputteringYield": (".plasma_wall_interaction", "SputteringYield"),
    "WallThermalModel": (".plasma_wall_interaction", "WallThermalModel"),
    "DivertorLifetimeAssessment": (".plasma_wall_interaction", "DivertorLifetimeAssessment"),
    # Phase 5 — Kinetic EFIT (anisotropic fast-ion pressure)
    "KineticEFIT": (".kinetic_efit", "KineticEFIT"),
    "FastIonPressure": (".kinetic_efit", "FastIonPressure"),
    # Phase 6 — Disruption sequence (TQ → CQ → RE → halo)
    "DisruptionConfig": (".disruption_sequence", "DisruptionConfig"),
    # Phase 6 — Locked mode (error field amplification → braking)
    "ErrorFieldSpectrum": (".locked_mode", "ErrorFieldSpectrum"),
    "ResonantFieldAmplification": (".locked_mode", "ResonantFieldAmplification"),
    "RotationEvolution": (".locked_mode", "RotationEvolution"),
    # Phase 6 — Plasma startup (Townsend → burn-through)
    "PaschenBreakdown": (".plasma_startup", "PaschenBreakdown"),
    "TownsendAvalanche": (".plasma_startup", "TownsendAvalanche"),
    # Phase 6 — L-H transition (zonal flow predator-prey)
    "PredatorPreyModel": (".lh_transition", "PredatorPreyModel"),
    "LHTrigger": (".lh_transition", "LHTrigger"),
    # Phase 6 — MARFE (radiation front instability)
    "MARFEFrontModel": (".marfe", "MARFEFrontModel"),
    "DensityLimitPredictor": (".marfe", "DensityLimitPredictor"),
    # Phase 6 — Neural turbulence (QLKNN surrogate)
    "QLKNNSurrogate": (".neural_turbulence", "QLKNNSurrogate"),
    # Phase 6 — Alpha orbit following
    "GuidingCenterOrbit": (".orbit_following", "GuidingCenterOrbit"),
    "OrbitClassifier": (".orbit_following", "OrbitClassifier"),
    # Phase 6 — Tearing mode coupling
    "CoupledTearingModes": (".tearing_mode_coupling", "CoupledTearingModes"),
    "ChirikovOverlap": (".tearing_mode_coupling", "ChirikovOverlap"),
    # Phase 6 — VMEC-lite (3D fixed-boundary MHD)
    "VMECLiteSolver": (".vmec_lite", "VMECLiteSolver"),
    # Phase 6 — Blob transport (SOL filaments)
    "BlobEnsemble": (".blob_transport", "BlobEnsemble"),
    "BlobDynamics": (".blob_transport", "BlobDynamics"),
    # GK three-path — interface + species + geometry
    "GKSolverBase": (".gk_interface", "GKSolverBase"),
    "GKLocalParams": (".gk_interface", "GKLocalParams"),
    "GKOutput": (".gk_interface", "GKOutput"),
    # GK — native linear eigenvalue solver
    "LinearGKResult": (".gk_eigenvalue", "LinearGKResult"),
    # GK — quasilinear flux model
    "quasilinear_fluxes_from_spectrum": (".gk_quasilinear", "quasilinear_fluxes_from_spectrum"),
    # GK — external solver interfaces
    "GyrokineticsParams": (".gyrokinetic_transport", "GyrokineticsParams"),
    "TransportFluxes": (".gyrokinetic_transport", "TransportFluxes"),
    # GK — hybrid surrogate+GK validation
    # GK — OOD detection + correction + scheduling
    # (available via direct import: from scpn_fusion.core.gk_ood_detector import ...)
    # JAX differentiable solvers
    "jax_gs_solve": (".jax_gs_solver", "jax_gs_solve"),
    "thomas_solve": (".jax_solvers", "thomas_solve"),
    # Integrated scenario simulator
    "ScenarioConfig": (".integrated_scenario", "ScenarioConfig"),
    "ScenarioState": (".integrated_scenario", "ScenarioState"),
    # Neoclassical transport (Chang-Hinton + Sauter bootstrap)
    "chang_hinton_chi": (".neoclassical", "chang_hinton_chi"),
    "sauter_bootstrap": (".neoclassical", "sauter_bootstrap"),
    # Vessel eddy current model
    "VesselModel": (".vessel_model", "VesselModel"),
    # Tokamak machine presets (ITER, SPARC, DIII-D, JET)
    "TokamakConfig": (".tokamak_config", "TokamakConfig"),
    # IMAS/OMAS adapter
    "EquilibriumIDS": (".imas_adapter", "EquilibriumIDS"),
    # Ballooning MHD stability
    "BallooningStabilityAnalysis": (".ballooning_solver", "BallooningStabilityAnalysis"),
    # Pedestal profile (mtanh + EPED1 width)
    "PedestalProfile": (".pedestal", "PedestalProfile"),
    "PedestalParams": (".pedestal", "PedestalParams"),
    # Checkpoint/resume
    "save_checkpoint": (".checkpoint", "save_checkpoint"),
    "load_checkpoint": (".checkpoint", "load_checkpoint"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
