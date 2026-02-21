# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Core Package Init
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
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
