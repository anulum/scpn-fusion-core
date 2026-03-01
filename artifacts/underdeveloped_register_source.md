# Underdeveloped Register

- Generated at: `2026-03-01T06:27:28.842836+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 31 |
| P0 + P1 entries | 0 |
| Source-domain entries | 31 |
| Source-domain P0 + P1 entries | 0 |
| Docs-claims entries | 0 |
| Domains affected | 5 |

## Marker Distribution

| Key | Count |
|---|---:|
| `FALLBACK` | 31 |

## Domain Distribution

| Key | Count |
|---|---:|
| `core_physics` | 16 |
| `control` | 7 |
| `compiler_runtime` | 6 |
| `diagnostics_io` | 1 |
| `nuclear` | 1 |

## Top Priority Backlog (Top 31)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P2 | 75 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:813` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | Predict disruption risk with checkpoint path if available, else deterministic fallback. |
| P2 | 75 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:819` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | ``metadata`` includes whether fallback mode was used. |
| P2 | 75 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:822` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | failures instead of returning fallback risk from ``predict_disruption_risk``. |
| P2 | 75 | `control` | `FALLBACK` | `src/scpn_fusion/control/fusion_control_room.py:391` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | logger.warning("Kernel init failed, fallback to analytic Psi: %s", kernel_error) |
| P2 | 75 | `control` | `FALLBACK` | `src/scpn_fusion/control/fusion_control_room.py:423` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | "Kernel coil update failed; continuing with fallback controls: %s", |
| P2 | 75 | `control` | `FALLBACK` | `src/scpn_fusion/control/h_infinity_controller.py:166` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | self._Fd: np.ndarray = self.F # fallback: continuous gain |
| P2 | 75 | `control` | `FALLBACK` | `src/scpn_fusion/control/neuro_cybernetic_controller.py:58` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | Push-pull spiking control population with deterministic fallback. |
| P2 | 74 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_rust_compat.py:276` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | Seed used by deterministic NumPy fallback backend. |
| P2 | 74 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_rust_compat.py:346` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | Seed used by deterministic NumPy fallback backend. |
| P2 | 74 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_rust_compat.py:500` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | "Use Python multigrid fallback in FusionKernel._multigrid_vcycle()." |
| P2 | 74 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_rust_compat.py:509` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | "Use Python multigrid fallback." |
| P2 | 74 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/fusion_kernel.py:198` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | "HPC Acceleration UNAVAILABLE (using Python fallback)." |
| P2 | 74 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/fusion_kernel.py:1644` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | fallback = np.asarray(coils.currents, dtype=np.float64).copy() |
| P2 | 74 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/fusion_kernel.py:1645` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | return np.clip(fallback, lb, ub).astype(np.float64) |
| P2 | 74 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/gpu_runtime.py:114` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | raise RuntimeError("PyTorch fallback requested but torch is not installed.") |
| P2 | 74 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/gpu_runtime.py:300` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | "PyTorch fallback requested but torch is not installed." |
| P2 | 74 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:74` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | _GYRO_BOHM_DEFAULT = 0.1 # Fallback if JSON not found |
| P2 | 74 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:1148` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | fallback: np.ndarray, |
| P2 | 74 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:1155` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | fb = np.asarray(fallback, dtype=np.float64) |
| P2 | 74 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/neural_transport.py:360` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | "critical-gradient fallback", |
| P2 | 74 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/quasi_3d_contracts.py:178` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | fallback = _require_finite_float("fallback_asymmetry", fallback_asymmetry) |
| P2 | 74 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/quasi_3d_contracts.py:183` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | ratio = float(np.clip(0.06 + 0.40 * abs(fallback), 0.0, 0.9)) |
| P2 | 74 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/quasi_3d_contracts.py:200` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | ratio = float(np.clip(0.06 + 0.40 * abs(fallback), 0.0, 0.9)) |
| P2 | 73 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/compiler.py:14` | Runtime WG | Measure fallback hit-rate and retire fallback from default lane. | - Float-path fallback when sc_neurocore is not installed. |
| P2 | 73 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/compiler.py:127` | Runtime WG | Measure fallback hit-rate and retire fallback from default lane. | Holds both the dense float matrices (for validation / fallback) and |
| P2 | 73 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/compiler.py:189` | Runtime WG | Measure fallback hit-rate and retire fallback from default lane. | "Use dense_forward_float for the numpy fallback." |
| P2 | 73 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/safety_interlocks.py:98` | Runtime WG | Measure fallback hit-rate and retire fallback from default lane. | def _safe_float(state: Mapping[str, float], key: str, fallback: float) -> float: |
| P2 | 73 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/safety_interlocks.py:99` | Runtime WG | Measure fallback hit-rate and retire fallback from default lane. | value = float(state.get(key, fallback)) |
| P2 | 73 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/safety_interlocks.py:100` | Runtime WG | Measure fallback hit-rate and retire fallback from default lane. | return value if np.isfinite(value) else float(fallback) |
| P2 | 73 | `nuclear` | `FALLBACK` | `src/scpn_fusion/nuclear/blanket_neutronics.py:159` | Nuclear WG | Measure fallback hit-rate and retire fallback from default lane. | else: # pragma: no cover - legacy NumPy fallback |
| P2 | 72 | `diagnostics_io` | `FALLBACK` | `src/scpn_fusion/io/tokamak_archive.py:362` | Diagnostics/IO WG | Measure fallback hit-rate and retire fallback from default lane. | Poll live MDSplus feed snapshots with deterministic merge + fallback metadata. |

## Full Register (Top 31)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:813` | Predict disruption risk with checkpoint path if available, else deterministic fallback. |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:819` | ``metadata`` includes whether fallback mode was used. |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:822` | failures instead of returning fallback risk from ``predict_disruption_risk``. |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/fusion_control_room.py:391` | logger.warning("Kernel init failed, fallback to analytic Psi: %s", kernel_error) |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/fusion_control_room.py:423` | "Kernel coil update failed; continuing with fallback controls: %s", |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/h_infinity_controller.py:166` | self._Fd: np.ndarray = self.F # fallback: continuous gain |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/neuro_cybernetic_controller.py:58` | Push-pull spiking control population with deterministic fallback. |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_rust_compat.py:276` | Seed used by deterministic NumPy fallback backend. |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_rust_compat.py:346` | Seed used by deterministic NumPy fallback backend. |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_rust_compat.py:500` | "Use Python multigrid fallback in FusionKernel._multigrid_vcycle()." |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_rust_compat.py:509` | "Use Python multigrid fallback." |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/fusion_kernel.py:198` | "HPC Acceleration UNAVAILABLE (using Python fallback)." |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/fusion_kernel.py:1644` | fallback = np.asarray(coils.currents, dtype=np.float64).copy() |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/fusion_kernel.py:1645` | return np.clip(fallback, lb, ub).astype(np.float64) |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/gpu_runtime.py:114` | raise RuntimeError("PyTorch fallback requested but torch is not installed.") |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/gpu_runtime.py:300` | "PyTorch fallback requested but torch is not installed." |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:74` | _GYRO_BOHM_DEFAULT = 0.1 # Fallback if JSON not found |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:1148` | fallback: np.ndarray, |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:1155` | fb = np.asarray(fallback, dtype=np.float64) |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/neural_transport.py:360` | "critical-gradient fallback", |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/quasi_3d_contracts.py:178` | fallback = _require_finite_float("fallback_asymmetry", fallback_asymmetry) |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/quasi_3d_contracts.py:183` | ratio = float(np.clip(0.06 + 0.40 * abs(fallback), 0.0, 0.9)) |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/quasi_3d_contracts.py:200` | ratio = float(np.clip(0.06 + 0.40 * abs(fallback), 0.0, 0.9)) |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/compiler.py:14` | - Float-path fallback when sc_neurocore is not installed. |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/compiler.py:127` | Holds both the dense float matrices (for validation / fallback) and |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/compiler.py:189` | "Use dense_forward_float for the numpy fallback." |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/safety_interlocks.py:98` | def _safe_float(state: Mapping[str, float], key: str, fallback: float) -> float: |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/safety_interlocks.py:99` | value = float(state.get(key, fallback)) |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/safety_interlocks.py:100` | return value if np.isfinite(value) else float(fallback) |
| P2 | `nuclear` | `FALLBACK` | `src/scpn_fusion/nuclear/blanket_neutronics.py:159` | else: # pragma: no cover - legacy NumPy fallback |
| P2 | `diagnostics_io` | `FALLBACK` | `src/scpn_fusion/io/tokamak_archive.py:362` | Poll live MDSplus feed snapshots with deterministic merge + fallback metadata. |
