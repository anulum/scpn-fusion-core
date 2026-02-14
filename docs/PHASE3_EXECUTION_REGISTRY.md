<!--
SCPN Fusion Core — Phase 3 Execution Registry
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# Phase 3 Execution Registry and Sprint S2 Queue

Date: 2026-02-13

This registry tracks the high-impact enhancement backlog imported after Phase 2 closure.

## Registry Size

| Source Pack | Imported Tasks |
|---|---|
| Control and simulation enhancement plan | 20 |
| HPC solver enhancement plan | 21 |
| Nuclear simulation enhancement plan | 19 |
| SCPN compiler enhancement plan | 25 |
| Total imported Phase 3 backlog | 85 |

Current tracker baseline (`docs/PHASE2_ADVANCED_RFC_TRACKER.md`): 20/20 tasks complete (`Done`).

## Prioritization Guardrails

- Keep strict green gates before each merge:
  - `cargo clippy --all-targets --all-features -- -D warnings`
  - `cargo test --all-features`
  - `python -m pytest -v`
- Maximum active work-in-progress: 2 tasks.
- Prefer changes that improve determinism, testability, and runtime safety before model scope expansion.
- External-data-dependent work stays blocked until data/license readiness is explicit in RFC.

## Sprint S2 Execution Queue (Ordered)

| ID | Priority | Track | Task | Target Files | Definition of Done | Validation |
|---|---|---|---|---|---|---|
| S2-001 | P0 | HPC | Add in-place zero-allocation bridge solve paths (`solve_into`, converged variant) | `src/scpn_fusion/hpc/hpc_bridge.py`, `tests/test_hpc_bridge.py` | Reusable output buffer paths exist with strict shape/layout checks and tests | `python -m pytest tests/test_hpc_bridge.py -v` |
| S2-002 | P0 | HPC | Harden C++ SOR convergence API with explicit bad-input guards | `src/scpn_fusion/hpc/solver.cpp`, `tests/test_hpc_bridge.py` | Null/size/iteration guards verified by tests and no clippy/format regressions | `python -m pytest tests/test_hpc_bridge.py -v` |
| S2-003 | P0 | SCPN | Add stochastic-vs-float equivalence benchmark gate (`<=5%` error) | `src/scpn_fusion/scpn/controller.py`, `validation/gneu_01_benchmark.py`, `tests/` | Deterministic campaign metric exposed and thresholded in tests | `python -m pytest tests/test_gneu_01_benchmark.py -v` |
| S2-004 | P1 | Control | Add robust model-loading fallback path in disruption predictor | `src/scpn_fusion/control/disruption_predictor.py`, `tests/` | Missing-model path no longer crashes and emits deterministic fallback score | `python -m pytest tests/test_gneu_02_anomaly.py -v` |
| S2-005 | P1 | Nuclear | Increase PWI coverage for angle-energy invariants + redeposition bounds | `src/scpn_fusion/nuclear/pwi_erosion.py`, `tests/test_pwi_erosion.py` | Invariants captured in tests with deterministic tolerances | `python -m pytest tests/test_pwi_erosion.py -v` |
| S2-006 | P1 | Nuclear | Extend TEMHD solver regression with pathological edge cases | `src/scpn_fusion/nuclear/temhd_peltier.py`, `tests/test_temhd_peltier.py` | Singular/near-singular stability behavior documented and tested | `python -m pytest tests/test_temhd_peltier.py -v` |
| S2-007 | P2 | Docs | Normalize path mappings for all imported tasks to repository modules | `docs/3d_gaps.md`, `docs/PHASE3_EXECUTION_REGISTRY.md` | Every queued task maps to real paths only | Docs review |
| S2-008 | P2 | Release | Add queue-level release gate summary for S2 | `validation/gdep_05_release_readiness.py`, `tests/test_gdep_05_release_readiness.py` | Report includes S2 queue health section without breaking current checks | `python -m pytest tests/test_gdep_05_release_readiness.py -v` |

## Sprint S3 Execution Queue (Ordered)

| ID | Priority | Track | Task | Target Files | Definition of Done | Validation |
|---|---|---|---|---|---|---|
| S3-001 | P0 | SCPN | Add topology diagnostics for dead nodes and unseeded place cycles | `src/scpn_fusion/scpn/structure.py`, `tests/test_scpn_compiler.py` | `validate_topology()` + optional strict compile validation implemented and tested | `python -m pytest tests/test_scpn_compiler.py tests/test_hypothesis_properties.py -v` |
| S3-002 | P0 | SCPN | Add inhibitor-arc support with explicit opt-in semantics | `src/scpn_fusion/scpn/structure.py`, `tests/test_scpn_compiler.py` | Inhibitor arc definition supported without breaking default positive-weight semantics | `python -m pytest tests/test_scpn_compiler.py -v` |
| S3-003 | P1 | SCPN | Add compact artifact serialization mode for packed bitstreams | `src/scpn_fusion/scpn/artifact.py`, `tests/test_controller.py` | Optional compact export/import path round-trips deterministically | `python -m pytest tests/test_controller.py -v` |
| S3-004 | P1 | Control | Normalize control simulation imports and deterministic fallback entry points | `src/scpn_fusion/control/disruption_predictor.py`, `src/scpn_fusion/control/tokamak_digital_twin.py`, `tests/test_gneu_02_anomaly.py` | No hard crash on missing model/dependency paths in core control entry points | `python -m pytest tests/test_gneu_02_anomaly.py -v` |
| S3-005 | P1 | HPC | Extend bridge/solver validation for invalid stride/shape edge paths | `src/scpn_fusion/hpc/hpc_bridge.py`, `src/scpn_fusion/hpc/solver.cpp`, `tests/test_hpc_bridge.py` | Additional boundary/fuzz-style guards and tests for invalid buffer geometry | `python -m pytest tests/test_hpc_bridge.py -v` |
| S3-006 | P2 | Release | Add S3 queue health visibility in release-readiness markdown | `validation/gdep_05_release_readiness.py`, `tests/test_gdep_05_release_readiness.py` | Release report captures S3 progress without weakening existing hard gates | `python -m pytest tests/test_gdep_05_release_readiness.py -v` |

## Sprint S4 Execution Queue (Ordered)

| ID | Priority | Track | Task | Target Files | Definition of Done | Validation |
|---|---|---|---|---|---|---|
| S4-001 | P0 | SCPN | Wire topology/inhibitor compile controls through `FusionCompiler` | `src/scpn_fusion/scpn/compiler.py`, `tests/test_scpn_compiler.py` | Compiler can enforce topology checks and inhibitor opt-in at compile entry point | `python -m pytest tests/test_scpn_compiler.py tests/test_hypothesis_properties.py -v` |
| S4-002 | P1 | SCPN | Add deterministic compact-packed artifact codec smoke checks independent of runtime packed availability | `src/scpn_fusion/scpn/artifact.py`, `tests/test_controller.py` | Compact codec validated directly for deterministic roundtrip of u64 payloads | `python -m pytest tests/test_controller.py -v` |
| S4-003 | P1 | Control | Add deterministic safe runtime summary path for `tokamak_flight_sim` | `src/scpn_fusion/control/tokamak_flight_sim.py`, `tests/` | Control script callable in CI without interactive plotting dependency | `python -m pytest -v` |
| S4-004 | P2 | Release | Extend release-readiness queue health to include S4 lane | `validation/gdep_05_release_readiness.py`, `tests/test_gdep_05_release_readiness.py` | Release report includes S2/S3/S4 queue snapshots | `python -m pytest tests/test_gdep_05_release_readiness.py -v` |

## Post-S4 Hardening Queue (Ad Hoc)

| ID | Priority | Track | Task | Target Files | Definition of Done | Validation |
|---|---|---|---|---|---|---|
| H5-001 | P1 | SCPN | Generalize controller feature passthrough sources from artifact injections | `src/scpn_fusion/scpn/controller.py`, `tests/test_controller.py` | Non-default injection sources are consumed without hardcoded feature keys; missing passthrough keys fail deterministically | `python -m pytest tests/test_controller.py -v`, `python -m mypy --strict src/scpn_fusion/scpn/controller.py` |
| H5-002 | P1 | SCPN | Add strict topology guard for positive input-weight overflow | `src/scpn_fusion/scpn/structure.py`, `tests/test_scpn_compiler.py` | Topology diagnostics surface transitions with positive input sum >1.0 and strict compile rejects them | `python -m pytest tests/test_scpn_compiler.py tests/test_hypothesis_properties.py -v`, `python -m mypy --strict src/scpn_fusion/scpn/structure.py` |
| H5-003 | P1 | SCPN | Add optional oracle-path bypass for runtime controller loops + generic observation mapping | `src/scpn_fusion/scpn/controller.py`, `tests/test_controller.py` | Controller accepts arbitrary observation mappings for feature passthrough and can skip oracle diagnostics path when disabled | `python -m pytest tests/test_controller.py -v`, `python -m mypy --strict src/scpn_fusion/scpn/controller.py` |
| H5-004 | P1 | SCPN | Wire explicit custom feature-axis mapping into controller runtime | `src/scpn_fusion/scpn/controller.py`, `tests/test_controller.py` | Controller can map non-`R_axis_m`/`Z_axis_m` observations into existing feature places via `feature_axes` and fails deterministically on missing keys | `python -m pytest tests/test_controller.py -v`, `python -m mypy --strict src/scpn_fusion/scpn/controller.py` |
| H5-005 | P0 | SCPN | Promote adaptive SCPN runtime defaults and add optional Rust dense-runtime kernels | `src/scpn_fusion/scpn/controller.py`, `scpn-fusion-rs/crates/fusion-python/src/lib.rs`, `tests/test_controller.py` | Controller default profile is adaptive (binary probabilistic margin), supports runtime backend selection (`auto`/`numpy`/`rust`), and uses Rust dense kernels when available for large nets | `python -m pytest tests/test_controller.py tests/test_scpn_pid_mpc_benchmark.py tests/test_gneu_01_benchmark.py -v`, `python -m mypy --strict src/scpn_fusion/scpn/controller.py`, `cargo test -p scpn-fusion-rs`, `cargo clippy -p scpn-fusion-rs --all-targets --all-features -- -D warnings` |
| H5-006 | P1 | SCPN | Reduce Python overhead in SCPN tick loop via vectorized injection/action decode and array-native stepping | `src/scpn_fusion/scpn/controller.py`, `tests/test_controller.py`, `tests/test_scpn_pid_mpc_benchmark.py` | Controller step avoids redundant list↔array conversions, uses vectorized place injection + action decode, and keeps benchmark thresholds unchanged | `python -m pytest tests/test_controller.py tests/test_scpn_pid_mpc_benchmark.py tests/test_gneu_01_benchmark.py -v`, `python -m mypy --strict src/scpn_fusion/scpn/controller.py` |
| H5-007 | P1 | SCPN | Keep controller state in internal NumPy marking array with compatibility accessors | `src/scpn_fusion/scpn/controller.py`, `tests/test_controller.py` | Internal runtime state remains array-native across ticks while external `marking` API remains compatible, preserving benchmark thresholds | `python -m pytest tests/test_controller.py tests/test_scpn_pid_mpc_benchmark.py tests/test_gneu_01_benchmark.py -v`, `python -m mypy --strict src/scpn_fusion/scpn/controller.py` |
| H5-008 | P1 | SCPN | Add compiled feature-evaluation path for injection sources to remove per-tick dict churn | `src/scpn_fusion/scpn/controller.py`, `tests/test_controller.py` | Controller computes feature components once per tick from compiled axis metadata and injects directly into marking with deterministic error semantics and unchanged benchmark thresholds | `python -m pytest tests/test_controller.py tests/test_scpn_pid_mpc_benchmark.py tests/test_gneu_01_benchmark.py -v`, `python -m mypy --strict src/scpn_fusion/scpn/controller.py` |
| H5-009 | P1 | SCPN | Add preallocated dense/update work buffers and explicit Rust-kernel execution regression | `src/scpn_fusion/scpn/controller.py`, `tests/test_controller.py` | Numpy backend reuses work buffers in dense/update path and tests verify Rust backend path executes when available (mocked), preserving deterministic behavior and benchmark thresholds | `python -m pytest tests/test_controller.py tests/test_scpn_pid_mpc_benchmark.py tests/test_gneu_01_benchmark.py -v`, `python -m mypy --strict src/scpn_fusion/scpn/controller.py` |
| H5-010 | P1 | SCPN | Remove per-tick observation-map churn and keep marking updates buffer-backed without aliasing | `src/scpn_fusion/scpn/controller.py`, `tests/test_controller.py` | Controller feature extraction uses incoming observation mapping directly, marking update writes into dedicated work buffers, and `marking` accessors preserve non-aliasing API behavior | `python -m pytest tests/test_controller.py tests/test_scpn_pid_mpc_benchmark.py tests/test_gneu_01_benchmark.py -v`, `python -m mypy --strict src/scpn_fusion/scpn/controller.py`, `python validation/scpn_pid_mpc_benchmark.py --seed 42 --steps 240 --strict` |
| H5-011 | P1 | SCPN | Convert feature/injection/input-marking hot path to fully reusable in-place scratch buffers | `src/scpn_fusion/scpn/controller.py` | Step path avoids per-tick allocation for observation vectors, feature components, injection values, and input-marking prep while preserving deterministic outputs and benchmark gates | `python -m pytest tests/test_controller.py tests/test_scpn_pid_mpc_benchmark.py tests/test_gneu_01_benchmark.py -v`, `python -m mypy --strict src/scpn_fusion/scpn/controller.py`, `python validation/scpn_pid_mpc_benchmark.py --seed 42 --steps 240 --strict` |
| H5-012 | P1 | SCPN | Replace high-allocation stochastic sampling path with binomial and low-allocation antithetic counting | `src/scpn_fusion/scpn/controller.py` | SC firing estimation avoids mirrored draw tensors, uses binomial counts for plain sampling, preserves deterministic replay, and keeps benchmark thresholds unchanged | `python -m pytest tests/test_controller.py tests/test_scpn_pid_mpc_benchmark.py tests/test_gneu_01_benchmark.py -v`, `python -m mypy --strict src/scpn_fusion/scpn/controller.py`, `python validation/scpn_pid_mpc_benchmark.py --seed 42 --steps 240 --strict` |
| H5-013 | P1 | SCPN | Add Rust stochastic-firing kernel bridge for runtime backend and validate Rust sample path execution | `src/scpn_fusion/scpn/controller.py`, `tests/test_controller.py`, `scpn-fusion-rs/crates/fusion-python/src/lib.rs`, `scpn-fusion-rs/crates/fusion-python/Cargo.toml` | Rust backend can offload SC firing-sampling counts (when bit-flip faults are disabled), Python keeps deterministic fallback, and tests prove Rust sample hook execution | `python -m pytest tests/test_controller.py tests/test_scpn_pid_mpc_benchmark.py tests/test_gneu_01_benchmark.py -v`, `python -m mypy --strict src/scpn_fusion/scpn/controller.py`, `cargo test -p scpn-fusion-rs`, `cargo clippy -p scpn-fusion-rs --all-targets --all-features -- -D warnings`, `python validation/scpn_pid_mpc_benchmark.py --seed 42 --steps 240 --strict` |
| H5-014 | P1 | SCPN | Add regression lock that default runtime profile remains adaptive (non-oracle binary) | `tests/test_controller.py` | Default controller construction (without runtime_profile override) provably diverges from strict oracle binary threshold path under identical seed/obs inputs | `python -m pytest tests/test_controller.py -v` |
| H5-015 | P1 | SCPN | Expand Rust stochastic-sampling offload coverage to bit-flip campaigns via dedicated fault RNG stream | `src/scpn_fusion/scpn/controller.py` | Rust backend sampling remains active even when bit-flip injection is enabled; faults use separate deterministic `sc_flip` RNG stream and benchmark gates remain unchanged | `python -m pytest tests/test_controller.py tests/test_scpn_pid_mpc_benchmark.py tests/test_gneu_01_benchmark.py -v`, `python -m mypy --strict src/scpn_fusion/scpn/controller.py`, `python validation/scpn_pid_mpc_benchmark.py --seed 42 --steps 240 --strict` |
| H5-016 | P0 | HPC | Restrict native solver library loading to trusted package paths + explicit env override | `src/scpn_fusion/hpc/hpc_bridge.py`, `tests/test_hpc_bridge.py` | HPC bridge no longer searches CWD for native libraries, supports `SCPN_SOLVER_LIB` opt-in override, and defaults to package-local paths only | `python -m pytest tests/test_hpc_bridge.py -v` |
| H5-017 | P0 | SCPN | Add bounded compact-payload decompression with strict base64 and size caps | `src/scpn_fusion/scpn/artifact.py`, `tests/test_controller.py` | Compact artifact decode validates payload type, compressed size, decompressed size, and count bounds to prevent zip-bomb / memory abuse paths | `python -m pytest tests/test_controller.py -v`, `python -m mypy --strict src/scpn_fusion/scpn/artifact.py` |
| H5-018 | P1 | CI | Pin optional `sc-neurocore` bootstrap dependency to immutable commit | `.github/workflows/ci.yml` | CI optional pip install references `sc-neurocore` with explicit commit SHA in both Python 3.12 bootstrap lanes for reproducibility | CI workflow run |
| H5-019 | P1 | Runtime | Remove hardcoded path + `sys.path` hack from realtime simulation entrypoint | `run_realtime_simulation.py` | Realtime script resolves config relative to script root and supports explicit `--config`, without mutating `sys.path` at runtime | `python run_realtime_simulation.py --help` |
| H5-020 | P1 | Container | Harden runtime image with prod-by-default dependency install and non-root user | `Dockerfile` | Runtime image installs `.` by default (`INSTALL_DEV=1` opt-in for dev extras) and runs as unprivileged `appuser` | Docker build smoke |
| H5-021 | P0 | Core | Add solver fail-fast mode for divergence and enable for validation configs | `src/scpn_fusion/core/fusion_kernel.py`, `validation/*.json`, `tests/test_fusion_kernel_fail_on_diverge.py` | Equilibrium solver raises on NaN/Inf when `solver.fail_on_diverge=true`; validation configs enable flag; tests cover both guarded and permissive behavior | `python -m pytest tests/test_fusion_kernel_fail_on_diverge.py -v` |
| H5-022 | P0 | SCPN | Harden py312 strict typing for stochastic binomial counts path | `src/scpn_fusion/scpn/controller.py` | Controller binomial branch is strictly array-typed under mypy py312 lane, eliminating scalar-or-array assignment ambiguity in CI | `python -m mypy --strict src/scpn_fusion/scpn/contracts.py src/scpn_fusion/scpn/controller.py src/scpn_fusion/scpn/structure.py src/scpn_fusion/scpn/artifact.py src/scpn_fusion/scpn/compiler.py src/scpn_fusion/scpn/__init__.py` |
| H5-023 | P1 | SCPN | Precompute transition-delay indices to remove per-tick timing mask/index churn | `src/scpn_fusion/scpn/controller.py` | Controller caches immediate/delayed transition index vectors and reuses delay-slot buffers in `_apply_transition_timing`, preserving deterministic outputs and benchmark thresholds | `python -m pytest tests/test_controller.py tests/test_scpn_pid_mpc_benchmark.py tests/test_gneu_01_benchmark.py -v`, `python -m mypy --strict src/scpn_fusion/scpn/controller.py`, `python validation/scpn_pid_mpc_benchmark.py --seed 42 --steps 240 --strict` |
| H5-024 | P1 | Control | Stabilize GAI-02 latency metric with deterministic hardware-normalized proxy | `src/scpn_fusion/control/torax_hybrid_loop.py` | TORAX-hybrid campaign reports deterministic loop-latency proxy independent of host jitter while preserving parity/avoidance thresholds and strict validation pass | `python -m pytest tests/test_gai_02_torax_hybrid.py -v`, `python validation/gai_02_torax_hybrid.py --strict`, `python -m pytest tests/ -q -x` |
| H5-025 | P2 | Tests | Remove path-hack/cwd coupling from legacy physics smoke tests | `tests/test_physics.py` | Physics tests run via package imports and `tmp_path` fixture config files, with no `sys.path` mutation or cwd file writes | `python -m pytest tests/test_physics.py -v` |
| H5-026 | P2 | Validation | Remove path-hack and hardcoded root path from ITER validation runner | `validation/validate_iter.py` | ITER validation script resolves config relative to repository root, exposes `--config`, and runs without `sys.path` mutation | `python validation/validate_iter.py --help` |
| H5-027 | P2 | Tests | Remove path-hack/cwd dependency from legacy unittest aggregation harness | `tests/run_all_tests.py` | `run_all_tests.py` relies on package imports only and writes temp config into an isolated temporary directory (no cwd mutation) | `python tests/run_all_tests.py` |
| H5-028 | P2 | Diagnostics | Remove path-hack/hardcoded root dependency from diagnostics demo runner | `src/scpn_fusion/diagnostics/run_diagnostics.py` | Diagnostics demo resolves config/output paths via `Path` defaults and exposes CLI flags without `sys.path` mutation | `python src/scpn_fusion/diagnostics/run_diagnostics.py --help` |
| H5-029 | P2 | Validation | Remove path injection from SPARC validation CLI and rely on package import path | `validation/validate_against_sparc.py` | SPARC validation script runs without mutating `sys.path`, preserving existing metrics/report behavior | `python validation/validate_against_sparc.py` |
| H5-030 | P2 | Validation | Remove path injection from G* validation runners and rely on package import path | `validation/control_resilience_campaign.py`, `validation/gai_01_turbulence_surrogate.py`, `validation/gai_02_torax_hybrid.py`, `validation/gai_03_heat_ml_shadow.py`, `validation/gdep_01_digital_twin_hook.py`, `validation/gdep_02_gpu_integration.py`, `validation/gmvr_03_stellarator_extension.py` | G-series validation runners execute without mutating `sys.path`, preserving existing CLI/report behavior via package import resolution | `python -m pytest tests/test_gai_01_turbulence_surrogate.py tests/test_gai_02_torax_hybrid.py tests/test_gai_03_heat_ml_shadow.py tests/test_gdep_01_digital_twin_hook.py tests/test_gdep_02_gpu_integration.py tests/test_gmvr_03_stellarator_extension.py -q` |

## Task Accounting

- Total imported tasks: 85
- Tasks currently queued for Sprint S2: 8
- Tasks currently queued for Sprint S3: 6
- Tasks currently queued for Sprint S4: 4
- Post-S4 hardening tasks delivered: 30
- Remaining in deferred pool after queue selection: 67

## Active Task

- Completed: `S2-001`
- Completed: `S2-002`
- Completed: `S2-003`
- Completed: `S2-004`
- Completed: `S2-005`
- Completed: `S2-006`
- Completed: `S2-007`
- Completed: `S2-008`
- Completed: `S3-001`
- Completed: `S3-002`
- Completed: `S3-003`
- Completed: `S3-004`
- Completed: `S3-005`
- Completed: `S3-006`
- Completed: `S4-001`
- Completed: `S4-002`
- Completed: `S4-003`
- Completed: `S4-004`
- Completed: `H5-001`
- Completed: `H5-002`
- Completed: `H5-003`
- Completed: `H5-004`
- Completed: `H5-005`
- Completed: `H5-006`
- Completed: `H5-007`
- Completed: `H5-008`
- Completed: `H5-009`
- Completed: `H5-010`
- Completed: `H5-011`
- Completed: `H5-012`
- Completed: `H5-013`
- Completed: `H5-014`
- Completed: `H5-015`
- Completed: `H5-016`
- Completed: `H5-017`
- Completed: `H5-018`
- Completed: `H5-019`
- Completed: `H5-020`
- Completed: `H5-021`
- Completed: `H5-022`
- Completed: `H5-023`
- Completed: `H5-024`
- Completed: `H5-025`
- Completed: `H5-026`
- Completed: `H5-027`
- Completed: `H5-028`
- Completed: `H5-029`
- Completed: `H5-030`
- Next active task: none (Sprint S4 queue baseline closed; deferred pool unchanged at 67 pending next sprint cut).
