# Underdeveloped Register

- Generated at: `2026-03-07T23:37:50.507711+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: production code + docs claims markers (tests/reports/html excluded)

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 66 |
| P0 + P1 entries | 0 |
| Source-domain entries | 2 |
| Source-domain P0 + P1 entries | 0 |
| Docs-claims entries | 64 |
| Domains affected | 2 |

## Marker Distribution

| Key | Count |
|---|---:|
| `FALLBACK` | 29 |
| `PLANNED` | 21 |
| `DEPRECATED` | 5 |
| `EXPERIMENTAL` | 5 |
| `NOT_VALIDATED` | 3 |
| `SIMPLIFIED` | 3 |

## Domain Distribution

| Key | Count |
|---|---:|
| `docs_claims` | 64 |
| `validation` | 2 |

## Source-Centric Priority Backlog (Top 0)

_Filtered to implementation domains to reduce docs/claims noise during hardening triage._

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|

## Top Priority Backlog (Top 66)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P2 | 73 | `validation` | `FALLBACK` | `tools/download_diiid_data.py:230` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | logger.debug("tokamak_archive fallback failed: %s", exc) |
| P2 | 73 | `validation` | `FALLBACK` | `tools/generate_fno_qlknn_spatial.py:143` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | print(" Will use critical-gradient fallback (less accurate).") |
| P2 | 68 | `docs_claims` | `NOT_VALIDATED` | `docs/HONEST_SCOPE.md:37` | Docs WG | Add real-data validation campaign and publish error bars. | \| FNO turbulence \| Synthetic-data trained; **not validated against gyrokinetics** \| Proxy mapping only; archived from release lane in v3.9 \| |
| P3 | 65 | `docs_claims` | `DEPRECATED` | `docs/PHYSICS_VALIDATION_STATUS.md:278` | Docs WG | Replace default path or remove lane before next major release. | \| `fno_eurofusion_jet` \| 2026-02-16 \| rel_L2=0.79 \| No (DEPRECATED) \| |
| P3 | 65 | `docs_claims` | `DEPRECATED` | `docs/PHYSICS_VALIDATION_STATUS.md:283` | Docs WG | Replace default path or remove lane before next major release. | The FNO is already DEPRECATED (v4.0 decision: retrain on real data or remove). |
| P3 | 65 | `docs_claims` | `DEPRECATED` | `docs/PHYSICS_VALIDATION_STATUS.md:326` | Docs WG | Replace default path or remove lane before next major release. | \| FNO turbulence \| Synthetic Hasegawa-Wakatani \| **No** (DEPRECATED) \| |
| P3 | 56 | `docs_claims` | `SIMPLIFIED` | `docs/assets/generate_header.py:27` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified Grad-Shafranov flux function with Shafranov shift + triangularity |
| P3 | 56 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/control.rst:197` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | simplified equilibrium and transport problems, used primarily for: |
| P3 | 56 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/transport.rst:82` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | using simplified ray-tracing: |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md:47` | Docs WG | Gate behind explicit flag and define validation exit criteria. | --experimental \ |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md:48` | Docs WG | Gate behind explicit flag and define validation exit criteria. | --experimental-ack I_UNDERSTAND_EXPERIMENTAL \ |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/PHYSICS_VALIDATION_STATUS.md:224` | Docs WG | Gate behind explicit flag and define validation exit criteria. | \| Multi-Regime FNO \| `src/.../fno_training_multi_regime.py` \| NumPy \| Synthetic H-W \| 30-90 min \| Experimental \| |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/PHYSICS_VALIDATION_STATUS.md:290` | Docs WG | Gate behind explicit flag and define validation exit criteria. | 4. **Multi-Regime FNO** — experimental, synthetic data only. |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/api/core.rst:225` | Docs WG | Gate behind explicit flag and define validation exit criteria. | Experimental Bridges |
| P3 | 47 | `docs_claims` | `FALLBACK` | `README.md:130` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | \| Graceful degradation \| Every path has a pure-Python fallback \| |
| P3 | 47 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:167` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | - Transparent fallback to analytic model when no weights are available |
| P3 | 47 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:259` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | with fallback to CPU SIMD for systems without GPU support. |
| P3 | 47 | `docs_claims` | `FALLBACK` | `docs/VALIDATION_GATE_MATRIX.md:19` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | | `freegs-strict` | FreeGS-only strict backend parity lane with runtime-fallback disallowed and artifact contract checks (`mode=freegs`, ... |
| P3 | 47 | `docs_claims` | `FALLBACK` | `docs/VALIDATION_GATE_MATRIX.md:29` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | | `freegs-strict` (`freegs-strict.yml`, manual dispatch) | Strict FreeGS backend parity lane; fails on any fallback or non-FreeGS referen... |
| P3 | 47 | `docs_claims` | `FALLBACK` | `docs/VALIDATION_GATE_MATRIX.md:36` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | enforces a no-fallback contract when invoked. |
| P3 | 37 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:41` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| GPU support \| Planned \| Yes (JAX) \| No \| No \| |
| P3 | 37 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:242` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| SOR red-black sweep \| wgpu compute shader \| 20–50× (65×65), 100–200× (256×256) \| P0 \| Planned \| |
| P3 | 37 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:243` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| Multigrid V-cycle \| wgpu + host orchestration \| 10–30× \| P1 \| Planned \| |
| P3 | 37 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:245` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| MLP batch inference \| wgpu or cuBLAS \| 2–5× (small H) \| P3 \| Planned \| |
| P3 | 37 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:246` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| FNO turbulence (FFT) \| cuFFT / wgpu FFT \| 50–100× (64×64) \| P3 \| Planned \| |
| P3 | 37 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:282` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| SCPN (planned) \| Quadtree, gradient-based \| 2D GS + 3D extension \| |
| P3 | 37 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:284` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | The planned quadtree AMR is simpler than JOREK's h-p adaptivity but |
| P3 | 37 | `docs_claims` | `DEPRECATED` | `docs/HARDENING_30_DAY_EXECUTION_PLAN.md:16` | Docs WG | Replace default path or remove lane before next major release. | - Lock default turbulence path to non-deprecated lanes only. |
| P3 | 37 | `docs_claims` | `DEPRECATED` | `docs/HARDENING_30_DAY_EXECUTION_PLAN.md:17` | Docs WG | Replace default path or remove lane before next major release. | - Keep deprecated FNO paths non-default and explicitly gated. |
| P3 | 37 | `docs_claims` | `PLANNED` | `docs/HONEST_SCOPE.md:25` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| Full 3D MHD \| Not planned for real-time loop \| Use NIMROD/M3D-C1 externally \| |
| P3 | 37 | `docs_claims` | `PLANNED` | `docs/competitive_analysis.md:77` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| GPU acceleration \| Planned (wgpu) \| Yes (JAX) \| No \| No \| JAX \| No \| |
| P3 | 29 | `docs_claims` | `FALLBACK` | `docs/3d_gaps.md:92` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | - fallback if external dependency is unavailable. |
| P3 | 29 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:93` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | │ ╱ Fallback (analytic) |
| P3 | 29 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:124` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | \| Fallback vectorised \| ~2 ms \| ~7× slower \| |
| P3 | 29 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:125` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | \| Fallback point-by-point loop \| ~200 ms \| ~670× slower \| |
| P3 | 29 | `docs_claims` | `FALLBACK` | `docs/CI_FAILURE_LEDGER.md:26` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | | CI-2026-03-03-006 | 2026-03-03 19:18 | 22638800884 (Python 3.9/3.10) | `ModuleNotFoundError: No module named 'tomllib'` from `tools/che... |
| P3 | 29 | `docs_claims` | `FALLBACK` | `docs/NEURAL_TRANSPORT_TRAINING.md:112` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | - fallback to analytic critical-gradient model on any contract failure |
| P3 | 29 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:337` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | **`dense_forward_float(W, inputs)`** — The float path: a simple `W @ inputs` using numpy. Used for validation and as a fallback when sc_n... |
| P3 | 29 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:502` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | **Rationale:** The Logic Compiler is a component of SCPN-Fusion-Core, which may be installed in environments where sc_neurocore is not pr... |
| P3 | 29 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:660` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | | **Graceful sc_neurocore fallback** | SCPN-Fusion-Core may be used without neuromorphic hardware. The Petri Net API should always be ava... |
| P3 | 29 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:740` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | The "Tokens are Bits" philosophy creates a direct isomorphism between Petri Net semantics and stochastic computing primitives, guaranteei... |
| P3 | 29 | `docs_claims` | `FALLBACK` | `docs/NEXT_SPRINT_EXECUTION_QUEUE.md:37` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | - Completed: `S1-003` (added low-point LCFS fallback regression test and VMEC-like geometry CI smoke coverage). |
| P3 | 29 | `docs_claims` | `FALLBACK` | `docs/PHYSICS_VALIDATION_STATUS.md:13` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | \| Solov'ev (fallback) \| psi_nrmse < 0.11 \| **PASS** (avg 0.076) \| |
| P3 | 28 | `docs_claims` | `NOT_VALIDATED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:1731` | Docs WG | Add real-data validation campaign and publish error bars. | 3. **Schema validation.** The JSON Schema (`scpnctl.schema.json`) is not validated against the artifact files in the test suite. Schema v... |
| P3 | 28 | `docs_claims` | `NOT_VALIDATED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:3084` | Docs WG | Add real-data validation campaign and publish error bars. | 5. **No schema validation.** The JSON Schema exists but is not validated at load time. This means a malformed artifact that satisfies the... |
| P3 | 19 | `docs_claims` | `FALLBACK` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:103` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | - Transparent fallback to full physics solve when surrogate confidence is below threshold |
| P3 | 19 | `docs_claims` | `FALLBACK` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:276` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | - f32 precision with tolerance-guarded fallback to f64 CPU path |
| P3 | 19 | `docs_claims` | `PLANNED` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:690` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | **Packet C — `scpn/runtime.py` — PetriNetEngine** (planned): |
| P3 | 19 | `docs_claims` | `PLANNED` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:704` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | **Packet D — `examples/01_traffic_light.py` — Hello World Demo** (planned): |
| P3 | 19 | `docs_claims` | `FALLBACK` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:779` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | commit SC state (or oracle fallback) |
| P3 | 19 | `docs_claims` | `FALLBACK` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:786` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | The oracle path is always available and always correct. The SC path currently returns the oracle result (fallback) but is structured so t... |
| P3 | 19 | `docs_claims` | `FALLBACK` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:1721` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | The untested lines in `artifact.py` are error-handling paths for malformed JSON that would require intentionally corrupted artifacts to e... |
| P3 | 19 | `docs_claims` | `FALLBACK` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:2904` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | 4. **Fallback policy.** If the divergence exceeds a threshold, the controller should revert to the oracle path. This provides a safety ne... |
| P3 | 19 | `docs_claims` | `PLANNED` | `docs/VALIDATION_AGAINST_ITER.md:238` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | Planned follow-up after WP-E1: |
| P3 | 19 | `docs_claims` | `FALLBACK` | `docs/promotions/HN_SHOW.md:11` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | SCPN Fusion Core is an open-source tokamak plasma physics simulator that covers the full lifecycle of a fusion reactor: Grad-Shafranov eq... |
| P3 | 19 | `docs_claims` | `FALLBACK` | `docs/promotions/REDDIT_PROGRAMMING.md:13` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | - Transparent fallback: `try: import scpn_fusion_rs` -- if the Rust extension isn't built, NumPy kicks in |
| P3 | 19 | `docs_claims` | `FALLBACK` | `docs/rfc/GPHY-06_RFC.md:70` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | - add deterministic tests for cache reuse, hot-swap behavior, and identity fallback. |
| P3 | 19 | `docs_claims` | `PLANNED` | `docs/sphinx/userguide/hpc.rst:6` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | native backend, C++ FFI bridge, and a planned GPU acceleration path. |
| P3 | 19 | `docs_claims` | `PLANNED` | `docs/sphinx/userguide/hpc.rst:96` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | GPU support is planned in three phases (tracked in |
| P3 | 9 | `docs_claims` | `PLANNED` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:284` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| Multigrid V-cycle (65x65) \| 10-30x \| Phase 3 planned \| |
| P3 | 9 | `docs_claims` | `PLANNED` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:285` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| FNO turbulence (FFT, 64x64) \| 50-100x \| Phase 3 planned \| |
| P3 | 9 | `docs_claims` | `PLANNED` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:286` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| MLP batch inference \| 2-5x \| Phase 3 planned \| |
| P3 | 9 | `docs_claims` | `PLANNED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:212` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | > "Packet C — `scpn/runtime.py` — PetriNetEngine (planned): The runtime engine wraps the `CompiledNet` in a simulation loop." |
| P3 | 9 | `docs_claims` | `PLANNED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:2675` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | Online learning is planned as future work (Section 34) with appropriate safety constraints (bounded weight deltas, Lyapunov stability mon... |
| P3 | 9 | `docs_claims` | `PLANNED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:2684` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | Plant model integration is planned as future work (Section 31). |
| P3 | 9 | `docs_claims` | `PLANNED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:2693` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | Hierarchical multi-controller composition is planned as future work (Section 33). |

## Full Register (Top 66)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P2 | `validation` | `FALLBACK` | `tools/download_diiid_data.py:230` | logger.debug("tokamak_archive fallback failed: %s", exc) |
| P2 | `validation` | `FALLBACK` | `tools/generate_fno_qlknn_spatial.py:143` | print(" Will use critical-gradient fallback (less accurate).") |
| P2 | `docs_claims` | `NOT_VALIDATED` | `docs/HONEST_SCOPE.md:37` | \| FNO turbulence \| Synthetic-data trained; **not validated against gyrokinetics** \| Proxy mapping only; archived from release lane in v3.9 \| |
| P3 | `docs_claims` | `DEPRECATED` | `docs/PHYSICS_VALIDATION_STATUS.md:278` | \| `fno_eurofusion_jet` \| 2026-02-16 \| rel_L2=0.79 \| No (DEPRECATED) \| |
| P3 | `docs_claims` | `DEPRECATED` | `docs/PHYSICS_VALIDATION_STATUS.md:283` | The FNO is already DEPRECATED (v4.0 decision: retrain on real data or remove). |
| P3 | `docs_claims` | `DEPRECATED` | `docs/PHYSICS_VALIDATION_STATUS.md:326` | \| FNO turbulence \| Synthetic Hasegawa-Wakatani \| **No** (DEPRECATED) \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/assets/generate_header.py:27` | # Simplified Grad-Shafranov flux function with Shafranov shift + triangularity |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/control.rst:197` | simplified equilibrium and transport problems, used primarily for: |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/transport.rst:82` | using simplified ray-tracing: |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md:47` | --experimental \ |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md:48` | --experimental-ack I_UNDERSTAND_EXPERIMENTAL \ |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/PHYSICS_VALIDATION_STATUS.md:224` | \| Multi-Regime FNO \| `src/.../fno_training_multi_regime.py` \| NumPy \| Synthetic H-W \| 30-90 min \| Experimental \| |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/PHYSICS_VALIDATION_STATUS.md:290` | 4. **Multi-Regime FNO** — experimental, synthetic data only. |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/api/core.rst:225` | Experimental Bridges |
| P3 | `docs_claims` | `FALLBACK` | `README.md:130` | \| Graceful degradation \| Every path has a pure-Python fallback \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:167` | - Transparent fallback to analytic model when no weights are available |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:259` | with fallback to CPU SIMD for systems without GPU support. |
| P3 | `docs_claims` | `FALLBACK` | `docs/VALIDATION_GATE_MATRIX.md:19` | | `freegs-strict` | FreeGS-only strict backend parity lane with runtime-fallback disallowed and artifact contract checks (`mode=freegs`, ... |
| P3 | `docs_claims` | `FALLBACK` | `docs/VALIDATION_GATE_MATRIX.md:29` | | `freegs-strict` (`freegs-strict.yml`, manual dispatch) | Strict FreeGS backend parity lane; fails on any fallback or non-FreeGS referen... |
| P3 | `docs_claims` | `FALLBACK` | `docs/VALIDATION_GATE_MATRIX.md:36` | enforces a no-fallback contract when invoked. |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:41` | \| GPU support \| Planned \| Yes (JAX) \| No \| No \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:242` | \| SOR red-black sweep \| wgpu compute shader \| 20–50× (65×65), 100–200× (256×256) \| P0 \| Planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:243` | \| Multigrid V-cycle \| wgpu + host orchestration \| 10–30× \| P1 \| Planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:245` | \| MLP batch inference \| wgpu or cuBLAS \| 2–5× (small H) \| P3 \| Planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:246` | \| FNO turbulence (FFT) \| cuFFT / wgpu FFT \| 50–100× (64×64) \| P3 \| Planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:282` | \| SCPN (planned) \| Quadtree, gradient-based \| 2D GS + 3D extension \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:284` | The planned quadtree AMR is simpler than JOREK's h-p adaptivity but |
| P3 | `docs_claims` | `DEPRECATED` | `docs/HARDENING_30_DAY_EXECUTION_PLAN.md:16` | - Lock default turbulence path to non-deprecated lanes only. |
| P3 | `docs_claims` | `DEPRECATED` | `docs/HARDENING_30_DAY_EXECUTION_PLAN.md:17` | - Keep deprecated FNO paths non-default and explicitly gated. |
| P3 | `docs_claims` | `PLANNED` | `docs/HONEST_SCOPE.md:25` | \| Full 3D MHD \| Not planned for real-time loop \| Use NIMROD/M3D-C1 externally \| |
| P3 | `docs_claims` | `PLANNED` | `docs/competitive_analysis.md:77` | \| GPU acceleration \| Planned (wgpu) \| Yes (JAX) \| No \| No \| JAX \| No \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/3d_gaps.md:92` | - fallback if external dependency is unavailable. |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:93` | │ ╱ Fallback (analytic) |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:124` | \| Fallback vectorised \| ~2 ms \| ~7× slower \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:125` | \| Fallback point-by-point loop \| ~200 ms \| ~670× slower \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/CI_FAILURE_LEDGER.md:26` | | CI-2026-03-03-006 | 2026-03-03 19:18 | 22638800884 (Python 3.9/3.10) | `ModuleNotFoundError: No module named 'tomllib'` from `tools/che... |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURAL_TRANSPORT_TRAINING.md:112` | - fallback to analytic critical-gradient model on any contract failure |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:337` | **`dense_forward_float(W, inputs)`** — The float path: a simple `W @ inputs` using numpy. Used for validation and as a fallback when sc_n... |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:502` | **Rationale:** The Logic Compiler is a component of SCPN-Fusion-Core, which may be installed in environments where sc_neurocore is not pr... |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:660` | | **Graceful sc_neurocore fallback** | SCPN-Fusion-Core may be used without neuromorphic hardware. The Petri Net API should always be ava... |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:740` | The "Tokens are Bits" philosophy creates a direct isomorphism between Petri Net semantics and stochastic computing primitives, guaranteei... |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEXT_SPRINT_EXECUTION_QUEUE.md:37` | - Completed: `S1-003` (added low-point LCFS fallback regression test and VMEC-like geometry CI smoke coverage). |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHYSICS_VALIDATION_STATUS.md:13` | \| Solov'ev (fallback) \| psi_nrmse < 0.11 \| **PASS** (avg 0.076) \| |
| P3 | `docs_claims` | `NOT_VALIDATED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:1731` | 3. **Schema validation.** The JSON Schema (`scpnctl.schema.json`) is not validated against the artifact files in the test suite. Schema v... |
| P3 | `docs_claims` | `NOT_VALIDATED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:3084` | 5. **No schema validation.** The JSON Schema exists but is not validated at load time. This means a malformed artifact that satisfies the... |
| P3 | `docs_claims` | `FALLBACK` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:103` | - Transparent fallback to full physics solve when surrogate confidence is below threshold |
| P3 | `docs_claims` | `FALLBACK` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:276` | - f32 precision with tolerance-guarded fallback to f64 CPU path |
| P3 | `docs_claims` | `PLANNED` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:690` | **Packet C — `scpn/runtime.py` — PetriNetEngine** (planned): |
| P3 | `docs_claims` | `PLANNED` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:704` | **Packet D — `examples/01_traffic_light.py` — Hello World Demo** (planned): |
| P3 | `docs_claims` | `FALLBACK` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:779` | commit SC state (or oracle fallback) |
| P3 | `docs_claims` | `FALLBACK` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:786` | The oracle path is always available and always correct. The SC path currently returns the oracle result (fallback) but is structured so t... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:1721` | The untested lines in `artifact.py` are error-handling paths for malformed JSON that would require intentionally corrupted artifacts to e... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:2904` | 4. **Fallback policy.** If the divergence exceeds a threshold, the controller should revert to the oracle path. This provides a safety ne... |
| P3 | `docs_claims` | `PLANNED` | `docs/VALIDATION_AGAINST_ITER.md:238` | Planned follow-up after WP-E1: |
| P3 | `docs_claims` | `FALLBACK` | `docs/promotions/HN_SHOW.md:11` | SCPN Fusion Core is an open-source tokamak plasma physics simulator that covers the full lifecycle of a fusion reactor: Grad-Shafranov eq... |
| P3 | `docs_claims` | `FALLBACK` | `docs/promotions/REDDIT_PROGRAMMING.md:13` | - Transparent fallback: `try: import scpn_fusion_rs` -- if the Rust extension isn't built, NumPy kicks in |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GPHY-06_RFC.md:70` | - add deterministic tests for cache reuse, hot-swap behavior, and identity fallback. |
| P3 | `docs_claims` | `PLANNED` | `docs/sphinx/userguide/hpc.rst:6` | native backend, C++ FFI bridge, and a planned GPU acceleration path. |
| P3 | `docs_claims` | `PLANNED` | `docs/sphinx/userguide/hpc.rst:96` | GPU support is planned in three phases (tracked in |
| P3 | `docs_claims` | `PLANNED` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:284` | \| Multigrid V-cycle (65x65) \| 10-30x \| Phase 3 planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:285` | \| FNO turbulence (FFT, 64x64) \| 50-100x \| Phase 3 planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:286` | \| MLP batch inference \| 2-5x \| Phase 3 planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:212` | > "Packet C — `scpn/runtime.py` — PetriNetEngine (planned): The runtime engine wraps the `CompiledNet` in a simulation loop." |
| P3 | `docs_claims` | `PLANNED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:2675` | Online learning is planned as future work (Section 34) with appropriate safety constraints (bounded weight deltas, Lyapunov stability mon... |
| P3 | `docs_claims` | `PLANNED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:2684` | Plant model integration is planned as future work (Section 31). |
| P3 | `docs_claims` | `PLANNED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:2693` | Hierarchical multi-controller composition is planned as future work (Section 33). |
