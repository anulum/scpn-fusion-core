# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Public core package exports for physics models, solvers, and utilities."""

from typing import Any

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
    "FluxEvolutionTrajectory": (".current_diffusion", "FluxEvolutionTrajectory"),
    "neoclassical_resistivity": (".current_diffusion", "neoclassical_resistivity"),
    "solve_flux_evolution_nonadiabatic": (
        ".current_diffusion",
        "solve_flux_evolution_nonadiabatic",
    ),
    # FRC rigid-rotor equilibrium — Steinhauer no-rotation analytical limit
    "RigidRotorFRCInputs": (".frc_rigid_rotor", "RigidRotorFRCInputs"),
    "FRCEquilibriumState": (".frc_rigid_rotor", "FRCEquilibriumState"),
    "FRCValidationReport": (".frc_rigid_rotor", "FRCValidationReport"),
    "ion_gyroradius_m": (".frc_rigid_rotor", "ion_gyroradius_m"),
    "frc_no_rotation_jax_observables": (
        ".frc_rigid_rotor",
        "frc_no_rotation_jax_observables",
    ),
    "rotating_frc_bvp_acceptance_status": (
        ".frc_rigid_rotor",
        "rotating_frc_bvp_acceptance_status",
    ),
    "solve_frc_equilibrium": (".frc_rigid_rotor", "solve_frc_equilibrium"),
    "ampere_residual": (".frc_rigid_rotor", "ampere_residual"),
    "flux_derivative_residual": (".frc_rigid_rotor", "flux_derivative_residual"),
    "psi_normalized_profile": (".frc_rigid_rotor", "psi_normalized_profile"),
    "pressure_balance_residual": (".frc_rigid_rotor", "pressure_balance_residual"),
    "density_profile": (".frc_rigid_rotor", "density_profile"),
    "beta_profile": (".frc_rigid_rotor", "beta_profile"),
    "force_balance_residual": (".frc_rigid_rotor", "force_balance_residual"),
    "null_radius": (".frc_rigid_rotor", "null_radius"),
    "s_parameter": (".frc_rigid_rotor", "s_parameter"),
    "validate_equilibrium": (".frc_rigid_rotor", "validate_equilibrium"),
    # Public FRC references — C-2U positive-net-heating table
    "C2UPositiveHeatingShot": (".public_frc_reference", "C2UPositiveHeatingShot"),
    "C2UPositiveHeatingSummary": (
        ".public_frc_reference",
        "C2UPositiveHeatingSummary",
    ),
    "c2u_positive_heating_reference_status": (
        ".public_frc_reference",
        "c2u_positive_heating_reference_status",
    ),
    "load_c2u_positive_heating_shots": (
        ".public_frc_reference",
        "load_c2u_positive_heating_shots",
    ),
    "summarise_c2u_positive_heating_shots": (
        ".public_frc_reference",
        "summarise_c2u_positive_heating_shots",
    ),
    # Pulsed Hall-MHD — axisymmetric Ono Eq. 8 flux carrier
    "HallMHDPulsedConfig": (".hall_mhd_pulsed", "HallMHDPulsedConfig"),
    "HallMHDPulsedState": (".hall_mhd_pulsed", "HallMHDPulsedState"),
    "axial_field_from_flux": (".hall_mhd_pulsed", "axial_field_from_flux"),
    "faraday_e_theta_from_b_ramp": (
        ".hall_mhd_pulsed",
        "faraday_e_theta_from_b_ramp",
    ),
    "gkeyll_small_hall_acceptance_status": (
        ".hall_mhd_pulsed",
        "gkeyll_small_hall_acceptance_status",
    ),
    "initial_hall_mhd_pulsed_state": (
        ".hall_mhd_pulsed",
        "initial_hall_mhd_pulsed_state",
    ),
    "ono_fig4_acceptance_status": (".hall_mhd_pulsed", "ono_fig4_acceptance_status"),
    "run_hall_mhd_pulsed": (".hall_mhd_pulsed", "run_hall_mhd_pulsed"),
    "spitzer_resistivity_ohm_m_hall": (
        ".hall_mhd_pulsed",
        "spitzer_resistivity_ohm_m",
    ),
    "step_hall_mhd_pulsed": (".hall_mhd_pulsed", "step_hall_mhd_pulsed"),
    # MRTI — analytical MIF/FRC growth spectrum with magnetic tension
    "MRTISpectrumState": (".mrti", "MRTISpectrumState"),
    "MRTISpectrumTracker": (".mrti", "MRTISpectrumTracker"),
    "effective_acceleration_from_radius_rate": (
        ".mrti",
        "effective_acceleration_from_radius_rate",
    ),
    "effective_acceleration_from_pulsed_compression": (
        ".mrti",
        "effective_acceleration_from_pulsed_compression",
    ),
    "mrti_growth_rate": (".mrti", "mrti_growth_rate"),
    "track_mrti_from_pulsed_compression": (".mrti", "track_mrti_from_pulsed_compression"),
    # Faraday recovery — closed-form MIF/FRC back-EMF over supplied trajectory
    "FaradayRecoveryReport": (".faraday_recovery", "FaradayRecoveryReport"),
    "FaradayRecoverySample": (".faraday_recovery", "FaradayRecoverySample"),
    "FaradayCompressionFluxBudget": (
        ".faraday_recovery",
        "FaradayCompressionFluxBudget",
    ),
    "FaradayCompressionTrajectoryDiagnostics": (
        ".faraday_recovery",
        "FaradayCompressionTrajectoryDiagnostics",
    ),
    "FaradayRecoveryTrajectoryPoint": (
        ".faraday_recovery",
        "FaradayRecoveryTrajectoryPoint",
    ),
    "faraday_back_emf": (".faraday_recovery", "faraday_back_emf"),
    "faraday_back_emf_from_values": (
        ".faraday_recovery",
        "faraday_back_emf_from_values",
    ),
    "faraday_trajectory_from_pulsed_compression": (
        ".faraday_recovery",
        "faraday_trajectory_from_pulsed_compression",
    ),
    "compression_work_from_pulsed_compression": (
        ".faraday_recovery",
        "compression_work_from_pulsed_compression",
    ),
    "compression_flux_budget_from_pulsed_compression": (
        ".faraday_recovery",
        "compression_flux_budget_from_pulsed_compression",
    ),
    "compression_trajectory_diagnostics_from_pulsed_compression": (
        ".faraday_recovery",
        "compression_trajectory_diagnostics_from_pulsed_compression",
    ),
    "faraday_trajectory_from_voltage_driven_compression": (
        ".faraday_recovery",
        "faraday_trajectory_from_voltage_driven_compression",
    ),
    "compression_work_from_voltage_driven_compression": (
        ".faraday_recovery",
        "compression_work_from_voltage_driven_compression",
    ),
    "compression_flux_budget_from_voltage_driven_compression": (
        ".faraday_recovery",
        "compression_flux_budget_from_voltage_driven_compression",
    ),
    "compression_trajectory_diagnostics_from_voltage_driven_compression": (
        ".faraday_recovery",
        "compression_trajectory_diagnostics_from_voltage_driven_compression",
    ),
    "coil_source_work_from_voltage_driven_compression": (
        ".faraday_recovery",
        "coil_source_work_from_voltage_driven_compression",
    ),
    "integrated_recovery_energy": (".faraday_recovery", "integrated_recovery_energy"),
    "magnetic_flux_wb": (".faraday_recovery", "magnetic_flux_wb"),
    # Pulsed compression — pressure-driven MIF/FRC trajectory with flux carrier
    "CoilGeometry": (".pulsed_compression", "CoilGeometry"),
    "CoilCircuitState": (".pulsed_compression", "CoilCircuitState"),
    "PulsedCompressionConfig": (".pulsed_compression", "PulsedCompressionConfig"),
    "PulsedCompressionState": (".pulsed_compression", "PulsedCompressionState"),
    "PulsedCompressionTrajectoryDiagnostics": (
        ".pulsed_compression",
        "PulsedCompressionTrajectoryDiagnostics",
    ),
    "VoltageDrivenCompressionResult": (
        ".pulsed_compression",
        "VoltageDrivenCompressionResult",
    ),
    "adiabatic_temperature_update_eV": (
        ".pulsed_compression",
        "adiabatic_temperature_update_eV",
    ),
    "coil_field_t": (".pulsed_compression", "coil_field_t"),
    "coil_current_interpolator": (".pulsed_compression", "coil_current_interpolator"),
    "initial_coil_circuit_state": (".pulsed_compression", "initial_coil_circuit_state"),
    "initial_pulsed_compression_state": (
        ".pulsed_compression",
        "initial_pulsed_compression_state",
    ),
    "run_coil_circuit": (".pulsed_compression", "run_coil_circuit"),
    "run_pulsed_compression": (".pulsed_compression", "run_pulsed_compression"),
    "run_voltage_driven_pulsed_compression": (
        ".pulsed_compression",
        "run_voltage_driven_pulsed_compression",
    ),
    "pulsed_compression_trajectory_diagnostics": (
        ".pulsed_compression",
        "pulsed_compression_trajectory_diagnostics",
    ),
    "slough_fig5_acceptance_status": (
        ".pulsed_compression",
        "slough_fig5_acceptance_status",
    ),
    "spitzer_resistivity_ohm_m": (".pulsed_compression", "spitzer_resistivity_ohm_m"),
    "step_coil_circuit": (".pulsed_compression", "step_coil_circuit"),
    "step_pulsed_compression": (".pulsed_compression", "step_pulsed_compression"),
    # FRC tilt mode — conservative n=1 MHD diagnostic with blocked Belova parity
    "FRCTiltModeReport": (".tilt_mode_frc", "FRCTiltModeReport"),
    "FRCTiltModeThresholds": (".tilt_mode_frc", "FRCTiltModeThresholds"),
    "FRCTiltModeTrajectoryPoint": (
        ".tilt_mode_frc",
        "FRCTiltModeTrajectoryPoint",
    ),
    "alfven_speed_m_s": (".tilt_mode_frc", "alfven_speed_m_s"),
    "belova_table1_acceptance_status": (
        ".tilt_mode_frc",
        "belova_table1_acceptance_status",
    ),
    "frc_tilt_growth_rate": (".tilt_mode_frc", "frc_tilt_growth_rate"),
    "rigid_body_flr_regime": (".tilt_mode_frc", "rigid_body_flr_regime"),
    "tilt_mode_report": (".tilt_mode_frc", "tilt_mode_report"),
    "tilt_mode_stable": (".tilt_mode_frc", "tilt_mode_stable"),
    "tilt_mode_trajectory_from_pulsed_compression": (
        ".tilt_mode_frc",
        "tilt_mode_trajectory_from_pulsed_compression",
    ),
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
    "AuroraParityCase": (".impurity_transport", "AuroraParityCase"),
    "AuroraParityImpuritySolver": (".impurity_transport", "AuroraParityImpuritySolver"),
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


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
