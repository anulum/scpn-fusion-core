# Underdeveloped Register

- Generated at: `2026-03-02T17:26:30.031195+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: docs-claims-only markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 154 |
| P0 + P1 entries | 15 |
| Source-domain entries | 0 |
| Source-domain P0 + P1 entries | 0 |
| Docs-claims entries | 154 |
| Domains affected | 1 |

## Marker Distribution

| Key | Count |
|---|---:|
| `FALLBACK` | 80 |
| `EXPERIMENTAL` | 28 |
| `PLANNED` | 22 |
| `SIMPLIFIED` | 13 |
| `DEPRECATED` | 7 |
| `NOT_VALIDATED` | 4 |

## Domain Distribution

| Key | Count |
|---|---:|
| `docs_claims` | 154 |

## Top Priority Backlog (Top 80)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P1 | 89 | `docs_claims` | `DEPRECATED` | `docs/VALIDATION_GATE_MATRIX.md:15` | Docs WG | Replace default path or remove lane before next major release. | | `release` | Version/claims integrity, underdeveloped/source-backlog drift checks, untested-module linkage guard, deprecated-default-lan... |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `README.md:18` | Docs WG | Gate behind explicit flag and define validation exit criteria. | > This repo is the full physics + experimental suite. |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `README.md:255` | Docs WG | Gate behind explicit flag and define validation exit criteria. | ### Experimental |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/BENCHMARKS.md:42` | Docs WG | Gate behind explicit flag and define validation exit criteria. | \| Experimental validation \| SPARC, ITPA, JET \| DIII-D \| ITER, DEMO \| JET \| |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/BENCHMARKS.md:327` | Docs WG | Gate behind explicit flag and define validation exit criteria. | scpn-fusion all --surrogate --experimental |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:8` | Docs WG | Gate behind explicit flag and define validation exit criteria. | - Preserve visibility on experimental/research lanes without letting them silently contaminate release acceptance. |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:15` | Docs WG | Gate behind explicit flag and define validation exit criteria. | | `release` | Version/claims integrity, underdeveloped/source-backlog drift checks, untested-module linkage guard, deprecated-default-lan... |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:16` | Docs WG | Gate behind explicit flag and define validation exit criteria. | \| `research` \| Experimental-only pytest lane (`pytest -m experimental`). \| `python tools/run_python_preflight.py --gate research` \| |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:24` | Docs WG | Gate behind explicit flag and define validation exit criteria. | \| `python-research-gate` \| Experimental validation lane (3.12) \| `research` \| |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:25` | Docs WG | Gate behind explicit flag and define validation exit criteria. | \| `validation-regression` \| Cross-language physics validation lane \| `release` (`pytest -m "not experimental"`) \| |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:27` | Docs WG | Gate behind explicit flag and define validation exit criteria. | ## Experimental Marker Contract |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:29` | Docs WG | Gate behind explicit flag and define validation exit criteria. | Tests marked with `@pytest.mark.experimental` are considered research-only and are excluded from release acceptance runs. As of v3.9.x th... |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:42` | Docs WG | Gate behind explicit flag and define validation exit criteria. | 1. Remove `@pytest.mark.experimental` from the test module(s). |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/competitive_analysis.md:79` | Docs WG | Gate behind explicit flag and define validation exit criteria. | \| Experimental validation \| SPARC, ITPA, JET \| DIII-D \| ITER, DEMO \| JET \| DIII-D \| ITER \| |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/userguide/validation.rst:5` | Docs WG | Gate behind explicit flag and define validation exit criteria. | SCPN-Fusion-Core is validated against published experimental data from |
| P2 | 80 | `docs_claims` | `NOT_VALIDATED` | `docs/HONEST_SCOPE.md:37` | Docs WG | Add real-data validation campaign and publish error bars. | \| FNO turbulence \| Synthetic-data trained; **not validated against gyrokinetics** \| Proxy mapping only; archived from release lane in v3.9 \| |
| P2 | 72 | `docs_claims` | `EXPERIMENTAL` | `docs/SOLVER_TUNING_GUIDE.md:74` | Docs WG | Gate behind explicit flag and define validation exit criteria. | 4. For real experimental data, start with `0.01` and adjust based on |
| P2 | 72 | `docs_claims` | `EXPERIMENTAL` | `docs/SOLVER_TUNING_GUIDE.md:98` | Docs WG | Gate behind explicit flag and define validation exit criteria. | \| **0.05–0.5** \| Standard robust range \| Real experimental data with occasional probe failures or calibration drift \| |
| P2 | 72 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/api/core.rst:225` | Docs WG | Gate behind explicit flag and define validation exit criteria. | Experimental Bridges |
| P2 | 71 | `docs_claims` | `DEPRECATED` | `docs/sphinx/changelog.rst:27` | Docs WG | Replace default path or remove lane before next major release. | - FNO turbulence surrogate deprecated (runtime ``FutureWarning``) |
| P2 | 69 | `docs_claims` | `DEPRECATED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:26` | Docs WG | Replace default path or remove lane before next major release. | \| `DEPRECATED` \| 17 \| |
| P2 | 69 | `docs_claims` | `DEPRECATED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:51` | Docs WG | Replace default path or remove lane before next major release. | - The critical residual gaps are targeted: physics fidelity upgrades, deprecated FNO lane retirement, and validation/claim rigor. |
| P2 | 69 | `docs_claims` | `DEPRECATED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:59` | Docs WG | Replace default path or remove lane before next major release. | | `src/scpn_fusion/core/fno_turbulence_suppressor.py` | 3 (`DEPRECATED`, `NOT_VALIDATED`, `SIMPLIFIED`) | Highest-credibility physics lan... |
| P2 | 69 | `docs_claims` | `DEPRECATED` | `docs/HARDENING_30_DAY_EXECUTION_PLAN.md:16` | Docs WG | Replace default path or remove lane before next major release. | - Lock default turbulence path to non-deprecated lanes only. |
| P2 | 69 | `docs_claims` | `DEPRECATED` | `docs/HARDENING_30_DAY_EXECUTION_PLAN.md:17` | Docs WG | Replace default path or remove lane before next major release. | - Keep deprecated FNO paths non-default and explicitly gated. |
| P2 | 68 | `docs_claims` | `SIMPLIFIED` | `docs/assets/generate_header.py:27` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified Grad-Shafranov flux function with Shafranov shift + triangularity |
| P2 | 68 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/control.rst:197` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | simplified equilibrium and transport problems, used primarily for: |
| P2 | 68 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/transport.rst:82` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | using simplified ray-tracing: |
| P3 | 62 | `docs_claims` | `EXPERIMENTAL` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:23` | Docs WG | Gate behind explicit flag and define validation exit criteria. | \| `EXPERIMENTAL` \| 55 \| |
| P3 | 62 | `docs_claims` | `EXPERIMENTAL` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:73` | Docs WG | Gate behind explicit flag and define validation exit criteria. | - `EXPERIMENTAL`: 15 mentions |
| P3 | 62 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:102` | Docs WG | Gate behind explicit flag and define validation exit criteria. | - Training on real experimental data via `train_from_geqdsk()` with profile perturbations, not synthetic data only |
| P3 | 62 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:343` | Docs WG | Gate behind explicit flag and define validation exit criteria. | ## 6. Experimental Validation Portfolio |
| P3 | 62 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:478` | Docs WG | Gate behind explicit flag and define validation exit criteria. | - **HSX** (University of Wisconsin): 4 field periods, QHS configuration, US-based experimental access |
| P3 | 62 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:546` | Docs WG | Gate behind explicit flag and define validation exit criteria. | \| Q4 \| DIII-D and JET GEQDSK validation (5+5 shots) \| Extended validation database \| Axis error < 10 mm on experimental shots \| |
| P3 | 62 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:554` | Docs WG | Gate behind explicit flag and define validation exit criteria. | - Experimental data access (DIII-D, JET): $50K |
| P3 | 62 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:611` | Docs WG | Gate behind explicit flag and define validation exit criteria. | \| **Experimental Access** \| \| \| \| \| |
| P3 | 62 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:634` | Docs WG | Gate behind explicit flag and define validation exit criteria. | **Experimental Access:** DIII-D and JET GEQDSK data are needed for expanding the validation database beyond the current 8 SPARC files. Co... |
| P3 | 62 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:788` | Docs WG | Gate behind explicit flag and define validation exit criteria. | 2. **General Atomics** — DIII-D experimental data access, validation collaboration |
| P3 | 62 | `docs_claims` | `EXPERIMENTAL` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:2534` | Docs WG | Gate behind explicit flag and define validation exit criteria. | Each paper would require additional experimental results (plant model integration, hardware benchmarks, comparison against conventional c... |
| P3 | 60 | `docs_claims` | `NOT_VALIDATED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:59` | Docs WG | Add real-data validation campaign and publish error bars. | | `src/scpn_fusion/core/fno_turbulence_suppressor.py` | 3 (`DEPRECATED`, `NOT_VALIDATED`, `SIMPLIFIED`) | Highest-credibility physics lan... |
| P3 | 60 | `docs_claims` | `NOT_VALIDATED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:1731` | Docs WG | Add real-data validation campaign and publish error bars. | 3. **Schema validation.** The JSON Schema (`scpnctl.schema.json`) is not validated against the artifact files in the test suite. Schema v... |
| P3 | 60 | `docs_claims` | `NOT_VALIDATED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:3084` | Docs WG | Add real-data validation campaign and publish error bars. | 5. **No schema validation.** The JSON Schema exists but is not validated at load time. This means a malformed artifact that satisfies the... |
| P3 | 59 | `docs_claims` | `FALLBACK` | `README.md:108` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | \| Graceful degradation \| Every path has a pure-Python fallback \| |
| P3 | 59 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:167` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | - Transparent fallback to analytic model when no weights are available |
| P3 | 59 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:259` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | with fallback to CPU SIMD for systems without GPU support. |
| P3 | 59 | `docs_claims` | `FALLBACK` | `docs/HONEST_SCOPE.md:18` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | \| Graceful degradation (no Rust / no GPU / no SC-NeuroCore) \| Every module has a pure-Python fallback \| |
| P3 | 54 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:24` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | \| `SIMPLIFIED` \| 27 \| |
| P3 | 54 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:59` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | | `src/scpn_fusion/core/fno_turbulence_suppressor.py` | 3 (`DEPRECATED`, `NOT_VALIDATED`, `SIMPLIFIED`) | Highest-credibility physics lan... |
| P3 | 54 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:60` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | \| `src/scpn_fusion/core/integrated_transport_solver.py` \| 1 (`SIMPLIFIED`) \| Largest physics file, central to transport claims \| |
| P3 | 54 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:61` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | \| `src/scpn_fusion/core/eped_pedestal.py` \| 1 (`SIMPLIFIED`) \| Pedestal closure fidelity bottleneck \| |
| P3 | 54 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:62` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | \| `src/scpn_fusion/core/stability_mhd.py` \| 1 (`SIMPLIFIED`) \| NTM/MHD reliability for disruption-oriented claims \| |
| P3 | 54 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:63` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | \| `src/scpn_fusion/control/fokker_planck_re.py` \| 1 (`SIMPLIFIED`) \| Runaway electron fidelity limits hard-safety narratives \| |
| P3 | 54 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:64` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | \| `src/scpn_fusion/control/spi_ablation.py` \| 1 (`SIMPLIFIED`) \| SPI mitigation realism in disruption campaigns \| |
| P3 | 54 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:65` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | \| `src/scpn_fusion/nuclear/blanket_neutronics.py` \| 1 (`SIMPLIFIED`) \| TBR confidence boundary \| |
| P3 | 54 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:66` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | \| `src/scpn_fusion/nuclear/nuclear_wall_interaction.py` \| 1 (`SIMPLIFIED`) \| PWI claims bounded by reduced model assumptions \| |
| P3 | 54 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:72` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | - `SIMPLIFIED`: 19 mentions |
| P3 | 49 | `docs_claims` | `FALLBACK` | `docs/3d_gaps.md:92` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | - fallback if external dependency is unavailable. |
| P3 | 49 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:41` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| GPU support \| Planned \| Yes (JAX) \| No \| No \| |
| P3 | 49 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:242` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| SOR red-black sweep \| wgpu compute shader \| 20–50× (65×65), 100–200× (256×256) \| P0 \| Planned \| |
| P3 | 49 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:243` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| Multigrid V-cycle \| wgpu + host orchestration \| 10–30× \| P1 \| Planned \| |
| P3 | 49 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:245` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| MLP batch inference \| wgpu or cuBLAS \| 2–5× (small H) \| P3 \| Planned \| |
| P3 | 49 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:246` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| FNO turbulence (FFT) \| cuFFT / wgpu FFT \| 50–100× (64×64) \| P3 \| Planned \| |
| P3 | 49 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:282` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| SCPN (planned) \| Quadtree, gradient-based \| 2D GS + 3D extension \| |
| P3 | 49 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:284` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | The planned quadtree AMR is simpler than JOREK's h-p adaptivity but |
| P3 | 49 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:93` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | │ ╱ Fallback (analytic) |
| P3 | 49 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:124` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | \| Fallback vectorised \| ~2 ms \| ~7× slower \| |
| P3 | 49 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:125` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | \| Fallback point-by-point loop \| ~200 ms \| ~670× slower \| |
| P3 | 49 | `docs_claims` | `FALLBACK` | `docs/GPU_ACCELERATION_ROADMAP.md:35` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | - Host orchestration in Rust with deterministic CPU fallback |
| P3 | 49 | `docs_claims` | `FALLBACK` | `docs/GPU_ACCELERATION_ROADMAP.md:64` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | - CPU fallback for unsupported nodes |
| P3 | 49 | `docs_claims` | `FALLBACK` | `docs/GPU_ACCELERATION_ROADMAP.md:122` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | - automatic CPU fallback |
| P3 | 49 | `docs_claims` | `PLANNED` | `docs/HONEST_SCOPE.md:25` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| Full 3D MHD \| Not planned for real-time loop \| Use NIMROD/M3D-C1 externally \| |
| P3 | 49 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:41` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | - [6.2 Import Boundary and Fallback Strategy](#62-import-boundary-and-fallback-strategy) |
| P3 | 49 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:269` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | **Dependencies:** numpy, sc_neurocore (optional — graceful fallback) |
| P3 | 49 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:337` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | **`dense_forward_float(W, inputs)`** — The float path: a simple `W @ inputs` using numpy. Used for validation and as a fallback when sc_n... |
| P3 | 49 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:502` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | **Rationale:** The Logic Compiler is a component of SCPN-Fusion-Core, which may be installed in environments where sc_neurocore is not pr... |
| P3 | 49 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:660` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | | **Graceful sc_neurocore fallback** | SCPN-Fusion-Core may be used without neuromorphic hardware. The Petri Net API should always be ava... |
| P3 | 49 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:740` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | The "Tokens are Bits" philosophy creates a direct isomorphism between Petri Net semantics and stochastic computing primitives, guaranteei... |
| P3 | 49 | `docs_claims` | `FALLBACK` | `docs/NEXT_SPRINT_EXECUTION_QUEUE.md:37` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | - Completed: `S1-003` (added low-point LCFS fallback regression test and VMEC-like geometry CI smoke coverage). |
| P3 | 49 | `docs_claims` | `FALLBACK` | `docs/PHASE2_ADVANCED_RFC_TEMPLATE.md:36` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | - Offline fallback path: |
| P3 | 49 | `docs_claims` | `FALLBACK` | `docs/SOLVER_TUNING_GUIDE.md:187` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | \| 1 \| 1× (serial fallback) \| Same as before \| |

## Full Register (Top 154)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P1 | `docs_claims` | `DEPRECATED` | `docs/VALIDATION_GATE_MATRIX.md:15` | | `release` | Version/claims integrity, underdeveloped/source-backlog drift checks, untested-module linkage guard, deprecated-default-lan... |
| P1 | `docs_claims` | `EXPERIMENTAL` | `README.md:18` | > This repo is the full physics + experimental suite. |
| P1 | `docs_claims` | `EXPERIMENTAL` | `README.md:255` | ### Experimental |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/BENCHMARKS.md:42` | \| Experimental validation \| SPARC, ITPA, JET \| DIII-D \| ITER, DEMO \| JET \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/BENCHMARKS.md:327` | scpn-fusion all --surrogate --experimental |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:8` | - Preserve visibility on experimental/research lanes without letting them silently contaminate release acceptance. |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:15` | | `release` | Version/claims integrity, underdeveloped/source-backlog drift checks, untested-module linkage guard, deprecated-default-lan... |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:16` | \| `research` \| Experimental-only pytest lane (`pytest -m experimental`). \| `python tools/run_python_preflight.py --gate research` \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:24` | \| `python-research-gate` \| Experimental validation lane (3.12) \| `research` \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:25` | \| `validation-regression` \| Cross-language physics validation lane \| `release` (`pytest -m "not experimental"`) \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:27` | ## Experimental Marker Contract |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:29` | Tests marked with `@pytest.mark.experimental` are considered research-only and are excluded from release acceptance runs. As of v3.9.x th... |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:42` | 1. Remove `@pytest.mark.experimental` from the test module(s). |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/competitive_analysis.md:79` | \| Experimental validation \| SPARC, ITPA, JET \| DIII-D \| ITER, DEMO \| JET \| DIII-D \| ITER \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/userguide/validation.rst:5` | SCPN-Fusion-Core is validated against published experimental data from |
| P2 | `docs_claims` | `NOT_VALIDATED` | `docs/HONEST_SCOPE.md:37` | \| FNO turbulence \| Synthetic-data trained; **not validated against gyrokinetics** \| Proxy mapping only; archived from release lane in v3.9 \| |
| P2 | `docs_claims` | `EXPERIMENTAL` | `docs/SOLVER_TUNING_GUIDE.md:74` | 4. For real experimental data, start with `0.01` and adjust based on |
| P2 | `docs_claims` | `EXPERIMENTAL` | `docs/SOLVER_TUNING_GUIDE.md:98` | \| **0.05–0.5** \| Standard robust range \| Real experimental data with occasional probe failures or calibration drift \| |
| P2 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/api/core.rst:225` | Experimental Bridges |
| P2 | `docs_claims` | `DEPRECATED` | `docs/sphinx/changelog.rst:27` | - FNO turbulence surrogate deprecated (runtime ``FutureWarning``) |
| P2 | `docs_claims` | `DEPRECATED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:26` | \| `DEPRECATED` \| 17 \| |
| P2 | `docs_claims` | `DEPRECATED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:51` | - The critical residual gaps are targeted: physics fidelity upgrades, deprecated FNO lane retirement, and validation/claim rigor. |
| P2 | `docs_claims` | `DEPRECATED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:59` | | `src/scpn_fusion/core/fno_turbulence_suppressor.py` | 3 (`DEPRECATED`, `NOT_VALIDATED`, `SIMPLIFIED`) | Highest-credibility physics lan... |
| P2 | `docs_claims` | `DEPRECATED` | `docs/HARDENING_30_DAY_EXECUTION_PLAN.md:16` | - Lock default turbulence path to non-deprecated lanes only. |
| P2 | `docs_claims` | `DEPRECATED` | `docs/HARDENING_30_DAY_EXECUTION_PLAN.md:17` | - Keep deprecated FNO paths non-default and explicitly gated. |
| P2 | `docs_claims` | `SIMPLIFIED` | `docs/assets/generate_header.py:27` | # Simplified Grad-Shafranov flux function with Shafranov shift + triangularity |
| P2 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/control.rst:197` | simplified equilibrium and transport problems, used primarily for: |
| P2 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/transport.rst:82` | using simplified ray-tracing: |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:23` | \| `EXPERIMENTAL` \| 55 \| |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:73` | - `EXPERIMENTAL`: 15 mentions |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:102` | - Training on real experimental data via `train_from_geqdsk()` with profile perturbations, not synthetic data only |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:343` | ## 6. Experimental Validation Portfolio |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:478` | - **HSX** (University of Wisconsin): 4 field periods, QHS configuration, US-based experimental access |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:546` | \| Q4 \| DIII-D and JET GEQDSK validation (5+5 shots) \| Extended validation database \| Axis error < 10 mm on experimental shots \| |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:554` | - Experimental data access (DIII-D, JET): $50K |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:611` | \| **Experimental Access** \| \| \| \| \| |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:634` | **Experimental Access:** DIII-D and JET GEQDSK data are needed for expanding the validation database beyond the current 8 SPARC files. Co... |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:788` | 2. **General Atomics** — DIII-D experimental data access, validation collaboration |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:2534` | Each paper would require additional experimental results (plant model integration, hardware benchmarks, comparison against conventional c... |
| P3 | `docs_claims` | `NOT_VALIDATED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:59` | | `src/scpn_fusion/core/fno_turbulence_suppressor.py` | 3 (`DEPRECATED`, `NOT_VALIDATED`, `SIMPLIFIED`) | Highest-credibility physics lan... |
| P3 | `docs_claims` | `NOT_VALIDATED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:1731` | 3. **Schema validation.** The JSON Schema (`scpnctl.schema.json`) is not validated against the artifact files in the test suite. Schema v... |
| P3 | `docs_claims` | `NOT_VALIDATED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:3084` | 5. **No schema validation.** The JSON Schema exists but is not validated at load time. This means a malformed artifact that satisfies the... |
| P3 | `docs_claims` | `FALLBACK` | `README.md:108` | \| Graceful degradation \| Every path has a pure-Python fallback \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:167` | - Transparent fallback to analytic model when no weights are available |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:259` | with fallback to CPU SIMD for systems without GPU support. |
| P3 | `docs_claims` | `FALLBACK` | `docs/HONEST_SCOPE.md:18` | \| Graceful degradation (no Rust / no GPU / no SC-NeuroCore) \| Every module has a pure-Python fallback \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:24` | \| `SIMPLIFIED` \| 27 \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:59` | | `src/scpn_fusion/core/fno_turbulence_suppressor.py` | 3 (`DEPRECATED`, `NOT_VALIDATED`, `SIMPLIFIED`) | Highest-credibility physics lan... |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:60` | \| `src/scpn_fusion/core/integrated_transport_solver.py` \| 1 (`SIMPLIFIED`) \| Largest physics file, central to transport claims \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:61` | \| `src/scpn_fusion/core/eped_pedestal.py` \| 1 (`SIMPLIFIED`) \| Pedestal closure fidelity bottleneck \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:62` | \| `src/scpn_fusion/core/stability_mhd.py` \| 1 (`SIMPLIFIED`) \| NTM/MHD reliability for disruption-oriented claims \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:63` | \| `src/scpn_fusion/control/fokker_planck_re.py` \| 1 (`SIMPLIFIED`) \| Runaway electron fidelity limits hard-safety narratives \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:64` | \| `src/scpn_fusion/control/spi_ablation.py` \| 1 (`SIMPLIFIED`) \| SPI mitigation realism in disruption campaigns \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:65` | \| `src/scpn_fusion/nuclear/blanket_neutronics.py` \| 1 (`SIMPLIFIED`) \| TBR confidence boundary \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:66` | \| `src/scpn_fusion/nuclear/nuclear_wall_interaction.py` \| 1 (`SIMPLIFIED`) \| PWI claims bounded by reduced model assumptions \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:72` | - `SIMPLIFIED`: 19 mentions |
| P3 | `docs_claims` | `FALLBACK` | `docs/3d_gaps.md:92` | - fallback if external dependency is unavailable. |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:41` | \| GPU support \| Planned \| Yes (JAX) \| No \| No \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:242` | \| SOR red-black sweep \| wgpu compute shader \| 20–50× (65×65), 100–200× (256×256) \| P0 \| Planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:243` | \| Multigrid V-cycle \| wgpu + host orchestration \| 10–30× \| P1 \| Planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:245` | \| MLP batch inference \| wgpu or cuBLAS \| 2–5× (small H) \| P3 \| Planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:246` | \| FNO turbulence (FFT) \| cuFFT / wgpu FFT \| 50–100× (64×64) \| P3 \| Planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:282` | \| SCPN (planned) \| Quadtree, gradient-based \| 2D GS + 3D extension \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:284` | The planned quadtree AMR is simpler than JOREK's h-p adaptivity but |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:93` | │ ╱ Fallback (analytic) |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:124` | \| Fallback vectorised \| ~2 ms \| ~7× slower \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:125` | \| Fallback point-by-point loop \| ~200 ms \| ~670× slower \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/GPU_ACCELERATION_ROADMAP.md:35` | - Host orchestration in Rust with deterministic CPU fallback |
| P3 | `docs_claims` | `FALLBACK` | `docs/GPU_ACCELERATION_ROADMAP.md:64` | - CPU fallback for unsupported nodes |
| P3 | `docs_claims` | `FALLBACK` | `docs/GPU_ACCELERATION_ROADMAP.md:122` | - automatic CPU fallback |
| P3 | `docs_claims` | `PLANNED` | `docs/HONEST_SCOPE.md:25` | \| Full 3D MHD \| Not planned for real-time loop \| Use NIMROD/M3D-C1 externally \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:41` | - [6.2 Import Boundary and Fallback Strategy](#62-import-boundary-and-fallback-strategy) |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:269` | **Dependencies:** numpy, sc_neurocore (optional — graceful fallback) |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:337` | **`dense_forward_float(W, inputs)`** — The float path: a simple `W @ inputs` using numpy. Used for validation and as a fallback when sc_n... |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:502` | **Rationale:** The Logic Compiler is a component of SCPN-Fusion-Core, which may be installed in environments where sc_neurocore is not pr... |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:660` | | **Graceful sc_neurocore fallback** | SCPN-Fusion-Core may be used without neuromorphic hardware. The Petri Net API should always be ava... |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:740` | The "Tokens are Bits" philosophy creates a direct isomorphism between Petri Net semantics and stochastic computing primitives, guaranteei... |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEXT_SPRINT_EXECUTION_QUEUE.md:37` | - Completed: `S1-003` (added low-point LCFS fallback regression test and VMEC-like geometry CI smoke coverage). |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE2_ADVANCED_RFC_TEMPLATE.md:36` | - Offline fallback path: |
| P3 | `docs_claims` | `FALLBACK` | `docs/SOLVER_TUNING_GUIDE.md:187` | \| 1 \| 1× (serial fallback) \| Same as before \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/SOLVER_TUNING_GUIDE.md:258` | \| Discontinuity at threshold \| Fallback model uses hard cutoff \| Switch to MLP weights (smooth transition) \| |
| P3 | `docs_claims` | `PLANNED` | `docs/competitive_analysis.md:75` | \| GPU acceleration \| Planned (wgpu) \| Yes (JAX) \| No \| No \| JAX \| No \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/gpu_roadmap.rst:18` | 1. ``wgpu`` SOR kernel (red-black stencil + deterministic CPU fallback) |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/gpu_roadmap.rst:19` | 2. GPU-backed GMRES preconditioning (CUDA/ROCm adapters with CPU fallback) |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/gpu_roadmap.rst:39` | - Operations: runtime capability detection + automatic CPU fallback |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/index.rst:54` | speedups with pure-Python fallback |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/installation.rst:42` | INFO: scpn_fusion_rs not found -- using NumPy fallback. |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/userguide/control.rst:89` | - **Checkpoint fallback** ensuring graceful degradation if the model |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/userguide/diagnostics.rst:97` | with automatic fallback for environments where SciPy is not optimised. |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/userguide/hpc.rst:102` | WebGPU) with deterministic CPU fallback. |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/userguide/hpc.rst:110` | CUDA/ROCm adapters for the GMRES linear solver with CPU fallback |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/userguide/hpc.rst:129` | - Operations: runtime capability detection + automatic CPU fallback |
| P3 | `docs_claims` | `FALLBACK` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:22` | \| `FALLBACK` \| 172 \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:71` | - `FALLBACK`: 67 mentions |
| P3 | `docs_claims` | `FALLBACK` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:103` | - Transparent fallback to full physics solve when surrogate confidence is below threshold |
| P3 | `docs_claims` | `FALLBACK` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:276` | - f32 precision with tolerance-guarded fallback to f64 CPU path |
| P3 | `docs_claims` | `FALLBACK` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:508` | **2. Rust Performance with Python Accessibility:** The dual-language architecture provides 10-50x performance over pure-Python alternativ... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:779` | commit SC state (or oracle fallback) |
| P3 | `docs_claims` | `FALLBACK` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:786` | The oracle path is always available and always correct. The SC path currently returns the oracle result (fallback) but is structured so t... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:1721` | The untested lines in `artifact.py` are error-handling paths for malformed JSON that would require intentionally corrupted artifacts to e... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:2904` | 4. **Fallback policy.** If the divergence exceeds a threshold, the controller should revert to the oracle path. This provides a safety ne... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:44` | | S2-004 | P1 | Control | Add robust model-loading fallback path in disruption predictor | `src/scpn_fusion/control/disruption_predictor.... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:57` | | S3-004 | P1 | Control | Normalize control simulation imports and deterministic fallback entry points | `src/scpn_fusion/control/disrupt... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:86` | | H5-013 | P1 | SCPN | Add Rust stochastic-firing kernel bridge for runtime backend and validate Rust sample path execution | `src/scpn_f... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:142` | | H7-005 | P1 | Control | Add deterministic NumPy LIF fallback backend for neuro-cybernetic spiking pool when sc-neurocore is unavailable... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:143` | | H7-006 | P1 | Control | Harden Director interface with deterministic built-in fallback oversight and CI-safe summary execution | `src/s... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:145` | | H7-008 | P1 | Nuclear | Restore NumPy 2.4 compatibility for blanket TBR integration in Python 3.11 CI lane | `src/scpn_fusion/nuclear/b... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:152` | | H7-015 | P1 | Diagnostics | Harden tomography inversion with strict signal guards and deterministic non-SciPy fallback solve path | `sr... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:165` | | H7-028 | P1 | Control | Remove residual global NumPy RNG mutation from tearing-mode simulation fallback path | `src/scpn_fusion/control... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:198` | | H7-061 | P1 | Control | Harden director-interface runtime mission with strict duration/glitch input guards | `src/scpn_fusion/control/d... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:200` | | H7-063 | P1 | Control | Harden disruption-predictor sequence-length handling with strict guardrails | `src/scpn_fusion/control/disrupti... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:201` | | H7-064 | P1 | Control | Harden fallback-director constructor with strict entropy/window input guards | `src/scpn_fusion/control/directo... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:202` | | H7-065 | P1 | Control | Harden disruption-predictor training campaign with strict shot/epoch input guards | `src/scpn_fusion/control/di... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:205` | | H7-068 | P1 | HPC | Harden native convergence bridge with strict max-iteration/tolerance/omega guards | `src/scpn_fusion/hpc/hpc_bridge... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:208` | | H7-071 | P1 | Control | Harden spiking-controller constructor with strict neuron/window/timebase/noise guards | `src/scpn_fusion/contro... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:209` | | H7-072 | P1 | Diagnostics | Harden tomography constructor with strict grid/regularization input guards | `src/scpn_fusion/diagnostics/t... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:216` | | H7-079 | P1 | Control | Harden disruption predictor sequence/training integer knobs with strict type/range guards | `src/scpn_fusion/co... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:252` | | H8-025 | P1 | Control | Harden analytic Shafranov and coil-solve APIs with strict finite/conditioning validation | `scpn-fusion-rs/crat... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:256` | | H8-029 | P1 | Operations | Harden digital-twin bit-flip fault injection with strict input/index validation | `scpn-fusion-rs/crates/fus... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:261` | | H8-034 | P1 | Runtime | Harden JIT kernel runtime execution with strict active-kernel and vector validation | `scpn-fusion-rs/crates/fu... |
| P3 | `docs_claims` | `FALLBACK` | `docs/promotions/HN_SHOW.md:11` | SCPN Fusion Core is an open-source tokamak plasma physics simulator that covers the full lifecycle of a fusion reactor: Grad-Shafranov eq... |
| P3 | `docs_claims` | `FALLBACK` | `docs/promotions/REDDIT_COMPILERS.md:15` | 2. **Compilation** (`scpn/compiler.py`) -- Each Petri net transition maps to a stochastic leaky-integrate-and-fire (LIF) neuron. Places b... |
| P3 | `docs_claims` | `FALLBACK` | `docs/promotions/REDDIT_PROGRAMMING.md:13` | - Transparent fallback: `try: import scpn_fusion_rs` -- if the Rust extension isn't built, NumPy kicks in |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GAI-01_RFC.md:43` | - Offline fallback path: benchmark remains fully offline (`numpy`, existing project modules). |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GAI-02_RFC.md:42` | - Offline fallback path: full campaign runs with stdlib + NumPy + in-repo modules. |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GAI-03_RFC.md:40` | - Offline fallback path: full benchmark and scanner integration remain NumPy/Pandas only. |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GDEP-01_RFC.md:40` | - Offline fallback path: campaign remains fully deterministic and local. |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GDEP-02_RFC.md:39` | - Offline fallback path: deterministic CPU and `gpu_sim` lanes run locally. |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GDEP-03_RFC.md:41` | - Offline fallback path: deterministic local report generation using bundled synthetic references. |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GDEP-05_RFC.md:42` | - Offline fallback path: deterministic local parsing of tracker/changelog files. |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GMVR-01_RFC.md:40` | - Offline fallback path: deterministic NumPy/Pandas scan remains available. |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GMVR-02_RFC.md:39` | - Offline fallback path: deterministic model and toroidal sweep remain in-repo. |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GMVR-03_RFC.md:40` | - Offline fallback path: deterministic field-line and Poincare diagnostics remain local. |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GNEU-01_RFC.md:43` | - Offline fallback path: benchmark remains fully offline and deterministic (`numpy` + existing repo modules). |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GPHY-01_RFC.md:38` | - Offline fallback path: particle overlay remains optional and disabled by default. |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GPHY-02_RFC.md:37` | - Offline fallback path: deterministic local trajectory integration and drift checks. |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GPHY-03_RFC.md:36` | - Offline fallback path: deterministic in-repo lookup/interpolation only. |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GPHY-04_RFC.md:38` | - Offline fallback path: existing analytic wall generator remains available. |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GPHY-05_RFC.md:37` | - Offline fallback path: additive APIs; existing controller paths remain available. |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GPHY-06_RFC.md:37` | - Offline fallback path: if no active compiled kernel is available, execution is identity/stable. |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GPHY-06_RFC.md:70` | - add deterministic tests for cache reuse, hot-swap behavior, and identity fallback. |
| P3 | `docs_claims` | `PLANNED` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:690` | **Packet C — `scpn/runtime.py` — PetriNetEngine** (planned): |
| P3 | `docs_claims` | `PLANNED` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:704` | **Packet D — `examples/01_traffic_light.py` — Hello World Demo** (planned): |
| P3 | `docs_claims` | `PLANNED` | `docs/VALIDATION_AGAINST_ITER.md:238` | Planned follow-up after WP-E1: |
| P3 | `docs_claims` | `PLANNED` | `docs/sphinx/userguide/hpc.rst:6` | native backend, C++ FFI bridge, and a planned GPU acceleration path. |
| P3 | `docs_claims` | `PLANNED` | `docs/sphinx/userguide/hpc.rst:96` | GPU support is planned in three phases (tracked in |
| P3 | `docs_claims` | `PLANNED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:25` | \| `PLANNED` \| 22 \| |
| P3 | `docs_claims` | `PLANNED` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:284` | \| Multigrid V-cycle (65x65) \| 10-30x \| Phase 3 planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:285` | \| FNO turbulence (FFT, 64x64) \| 50-100x \| Phase 3 planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:286` | \| MLP batch inference \| 2-5x \| Phase 3 planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:212` | > "Packet C — `scpn/runtime.py` — PetriNetEngine (planned): The runtime engine wraps the `CompiledNet` in a simulation loop." |
| P3 | `docs_claims` | `PLANNED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:2675` | Online learning is planned as future work (Section 34) with appropriate safety constraints (bounded weight deltas, Lyapunov stability mon... |
| P3 | `docs_claims` | `PLANNED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:2684` | Plant model integration is planned as future work (Section 31). |
| P3 | `docs_claims` | `PLANNED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:2693` | Hierarchical multi-controller composition is planned as future work (Section 33). |
