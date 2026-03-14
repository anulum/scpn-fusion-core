# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Control Module
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from .fusion_sota_mpc import run_sota_simulation, ModelPredictiveController
from .fusion_nmpc_jax import get_nmpc_controller, NonlinearMPC

# Lazy imports to avoid circular dependency chains
# (gpu_runtime -> disruption_predictor -> control.__init__)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AnalyticEquilibriumSolver": (".analytic_solver", "AnalyticEquilibriumSolver"),
    "run_analytic_solver": (".analytic_solver", "run_analytic_solver"),
    "DirectorInterface": (".director_interface", "DirectorInterface"),
    "HybridAnomalyDetector": (".disruption_predictor", "HybridAnomalyDetector"),
    "FuelingSimResult": (".fueling_mode", "FuelingSimResult"),
    "IcePelletFuelingController": (".fueling_mode", "IcePelletFuelingController"),
    "TokamakPhysicsEngine": (".fusion_control_room", "TokamakPhysicsEngine"),
    "OptimalController": (".fusion_optimal_control", "OptimalController"),
    "run_optimal_control": (".fusion_optimal_control", "run_optimal_control"),
    "TokamakEnv": (".gym_tokamak_env", "TokamakEnv"),
    "HInfinityController": (".h_infinity_controller", "HInfinityController"),
    "get_radial_robust_controller": (".h_infinity_controller", "get_radial_robust_controller"),
    "HaloCurrentModel": (".halo_re_physics", "HaloCurrentModel"),
    "RunawayElectronModel": (".halo_re_physics", "RunawayElectronModel"),
    "NeuroCyberneticController": (".neuro_cybernetic_controller", "NeuroCyberneticController"),
    "SpiAblationSolver": (".spi_ablation", "SpiAblationSolver"),
    "ShatteredPelletInjection": (".spi_mitigation", "ShatteredPelletInjection"),
    # Real-time EFIT equilibrium reconstruction
    "RealtimeEFIT": (".realtime_efit", "RealtimeEFIT"),
    "MagneticDiagnostics": (".realtime_efit", "MagneticDiagnostics"),
    # Plasma shape controller — Jacobian + Tikhonov
    "PlasmaShapeController": (".shape_controller", "PlasmaShapeController"),
    "CoilSet": (".shape_controller", "CoilSet"),
    # Super-twisting sliding mode vertical stabilizer
    "SuperTwistingSMC": (".sliding_mode_vertical", "SuperTwistingSMC"),
    "VerticalStabilizer": (".sliding_mode_vertical", "VerticalStabilizer"),
    # Fault-tolerant reconfigurable control
    "FDIMonitor": (".fault_tolerant_control", "FDIMonitor"),
    "ReconfigurableController": (".fault_tolerant_control", "ReconfigurableController"),
    # Constrained safe RL (Lagrangian PPO)
    "LagrangianPPO": (".safe_rl_controller", "LagrangianPPO"),
    "ConstrainedGymTokamakEnv": (".safe_rl_controller", "ConstrainedGymTokamakEnv"),
    # Feedforward scenario scheduler
    "ScenarioSchedule": (".scenario_scheduler", "ScenarioSchedule"),
    "FeedforwardController": (".scenario_scheduler", "FeedforwardController"),
    # Gain-scheduled multi-regime controller
    "GainScheduledController": (".gain_scheduled_controller", "GainScheduledController"),
    "RegimeDetector": (".gain_scheduled_controller", "RegimeDetector"),
    # Phase 5 — Free-boundary tracking (direct kernel + supervisor + EKF)
    "FreeBoundaryTrackingController": (".free_boundary_tracking", "FreeBoundaryTrackingController"),
    # Phase 5 — Extended Kalman Filter state estimator
    "ExtendedKalmanFilter": (".state_estimator", "ExtendedKalmanFilter"),
    # Phase 5 — Volt-second management (flux budget optimization)
    "FluxBudget": (".volt_second_manager", "FluxBudget"),
    "FluxConsumptionMonitor": (".volt_second_manager", "FluxConsumptionMonitor"),
    "VoltSecondOptimizer": (".volt_second_manager", "VoltSecondOptimizer"),
    # Phase 5 — RWM feedback (sensor-coil stabilization)
    "RWMFeedbackController": (".rwm_feedback", "RWMFeedbackController"),
    "RWMPhysics": (".rwm_feedback", "RWMPhysics"),
    # Phase 5 — Mu-synthesis (D-K iteration)
    "MuSynthesisController": (".mu_synthesis", "MuSynthesisController"),
    # Phase 6 — Detachment control (impurity seeding)
    "DetachmentController": (".detachment_controller", "DetachmentController"),
    # Phase 6 — Density control (fueling + pumping)
    "DensityController": (".density_controller", "DensityController"),
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


__all__ = [
    "AnalyticEquilibriumSolver",
    "CoilSet",
    "ConstrainedGymTokamakEnv",
    "DensityController",
    "DetachmentController",
    "DirectorInterface",
    "ExtendedKalmanFilter",
    "FDIMonitor",
    "FeedforwardController",
    "FluxBudget",
    "FluxConsumptionMonitor",
    "FreeBoundaryTrackingController",
    "FuelingSimResult",
    "GainScheduledController",
    "get_nmpc_controller",
    "get_radial_robust_controller",
    "HaloCurrentModel",
    "HInfinityController",
    "HybridAnomalyDetector",
    "IcePelletFuelingController",
    "LagrangianPPO",
    "MagneticDiagnostics",
    "ModelPredictiveController",
    "MuSynthesisController",
    "NeuroCyberneticController",
    "NonlinearMPC",
    "OptimalController",
    "PlasmaShapeController",
    "RealtimeEFIT",
    "ReconfigurableController",
    "RegimeDetector",
    "run_analytic_solver",
    "run_optimal_control",
    "run_sota_simulation",
    "RunawayElectronModel",
    "RWMFeedbackController",
    "RWMPhysics",
    "ScenarioSchedule",
    "ShatteredPelletInjection",
    "SpiAblationSolver",
    "SuperTwistingSMC",
    "TokamakEnv",
    "TokamakPhysicsEngine",
    "VerticalStabilizer",
    "VoltSecondOptimizer",
]
