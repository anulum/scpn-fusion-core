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
    "DirectorInterface",
    "FuelingSimResult",
    "get_nmpc_controller",
    "get_radial_robust_controller",
    "HaloCurrentModel",
    "HInfinityController",
    "HybridAnomalyDetector",
    "IcePelletFuelingController",
    "ModelPredictiveController",
    "NeuroCyberneticController",
    "NonlinearMPC",
    "OptimalController",
    "run_analytic_solver",
    "run_optimal_control",
    "run_sota_simulation",
    "RunawayElectronModel",
    "ShatteredPelletInjection",
    "SpiAblationSolver",
    "TokamakEnv",
    "TokamakPhysicsEngine",
]
