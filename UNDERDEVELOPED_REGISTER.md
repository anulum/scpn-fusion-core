# Underdeveloped Register

- Generated at: `2026-03-04T01:15:14.711150+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: production code + docs claims markers (tests/reports/html excluded)

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 204 |
| P0 + P1 entries | 16 |
| Source-domain entries | 41 |
| Source-domain P0 + P1 entries | 16 |
| Docs-claims entries | 142 |
| Domains affected | 6 |

## Marker Distribution

| Key | Count |
|---|---:|
| `FALLBACK` | 116 |
| `PLANNED` | 23 |
| `EXPERIMENTAL` | 21 |
| `SIMPLIFIED` | 18 |
| `MONOLITH` | 11 |
| `DEPRECATED` | 10 |
| `NOT_VALIDATED` | 4 |
| `FALLBACK_DENSITY` | 1 |

## Domain Distribution

| Key | Count |
|---|---:|
| `docs_claims` | 142 |
| `validation` | 29 |
| `other` | 21 |
| `core_physics` | 6 |
| `control` | 4 |
| `compiler_runtime` | 2 |

## Source-Centric Priority Backlog (Top 16)

_Filtered to implementation domains to reduce docs/claims noise during hardening triage._

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 102 | `control` | `MONOLITH` | `src/scpn_fusion/control/disruption_contracts.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=649 exceeds monolith threshold (500+). |
| P0 | 102 | `control` | `MONOLITH` | `src/scpn_fusion/control/halo_re_physics.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=651 exceeds monolith threshold (500+). |
| P0 | 102 | `control` | `MONOLITH` | `src/scpn_fusion/control/hil_harness.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=547 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fno_training.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=572 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=590 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel_newton_solver.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=632 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_equilibrium.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=595 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=509 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=517 exceeds monolith threshold (500+). |
| P0 | 100 | `compiler_runtime` | `MONOLITH` | `src/scpn_fusion/scpn/artifact.py:1` | Runtime WG | Split module into focused subcomponents and lock interface contracts. | module LOC=583 exceeds monolith threshold (500+). |
| P0 | 100 | `compiler_runtime` | `MONOLITH` | `src/scpn_fusion/scpn/controller.py:1` | Runtime WG | Split module into focused subcomponents and lock interface contracts. | module LOC=599 exceeds monolith threshold (500+). |
| P0 | 96 | `control` | `FALLBACK_DENSITY` | `src/scpn_fusion/control/disruption_predictor.py:1` | Control WG | Reduce fallback concentration and enforce strict-backend parity checks. | fallback risk signals=6 across LOC=437; high fallback concentration in runtime code paths. |
| P1 | 82 | `validation` | `SIMPLIFIED` | `tools/train_fno_qlknn_spatial.py:120` | Validation WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified: spectral conv in truncated mode space |
| P1 | 82 | `validation` | `SIMPLIFIED` | `validation/stress_test_campaign.py:391` | Validation WG | Upgrade with higher-fidelity closure or tighten domain contract. | implements a 10 kHz PID control loop with simplified linear plasma |
| P1 | 82 | `validation` | `SIMPLIFIED` | `validation/test_full_pulse_scenario.py:62` | Validation WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Step Physics (Single step of simulate logic simplified) |
| P1 | 82 | `validation` | `SIMPLIFIED` | `validation/validate_iter.py:59` | Validation WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Our model is simplified (L-mode profiles mostly), so if we get > 100MW it's a "physics pass" |

## Top Priority Backlog (Top 80)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 102 | `control` | `MONOLITH` | `src/scpn_fusion/control/disruption_contracts.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=649 exceeds monolith threshold (500+). |
| P0 | 102 | `control` | `MONOLITH` | `src/scpn_fusion/control/halo_re_physics.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=651 exceeds monolith threshold (500+). |
| P0 | 102 | `control` | `MONOLITH` | `src/scpn_fusion/control/hil_harness.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=547 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fno_training.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=572 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=590 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel_newton_solver.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=632 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_equilibrium.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=595 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=509 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=517 exceeds monolith threshold (500+). |
| P0 | 100 | `compiler_runtime` | `MONOLITH` | `src/scpn_fusion/scpn/artifact.py:1` | Runtime WG | Split module into focused subcomponents and lock interface contracts. | module LOC=583 exceeds monolith threshold (500+). |
| P0 | 100 | `compiler_runtime` | `MONOLITH` | `src/scpn_fusion/scpn/controller.py:1` | Runtime WG | Split module into focused subcomponents and lock interface contracts. | module LOC=599 exceeds monolith threshold (500+). |
| P0 | 96 | `control` | `FALLBACK_DENSITY` | `src/scpn_fusion/control/disruption_predictor.py:1` | Control WG | Reduce fallback concentration and enforce strict-backend parity checks. | fallback risk signals=6 across LOC=437; high fallback concentration in runtime code paths. |
| P1 | 82 | `validation` | `SIMPLIFIED` | `tools/train_fno_qlknn_spatial.py:120` | Validation WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified: spectral conv in truncated mode space |
| P1 | 82 | `validation` | `SIMPLIFIED` | `validation/stress_test_campaign.py:391` | Validation WG | Upgrade with higher-fidelity closure or tighten domain contract. | implements a 10 kHz PID control loop with simplified linear plasma |
| P1 | 82 | `validation` | `SIMPLIFIED` | `validation/test_full_pulse_scenario.py:62` | Validation WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Step Physics (Single step of simulate logic simplified) |
| P1 | 82 | `validation` | `SIMPLIFIED` | `validation/validate_iter.py:59` | Validation WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Our model is simplified (L-mode profiles mostly), so if we get > 100MW it's a "physics pass" |
| P2 | 78 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:383` | Validation WG | Gate behind explicit flag and define validation exit criteria. | "Experimental-only pytest suite", |
| P2 | 78 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:384` | Validation WG | Gate behind explicit flag and define validation exit criteria. | [sys.executable, "-m", "pytest", "tests/", "-q", "-m", "experimental"], |
| P2 | 78 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:503` | Validation WG | Gate behind explicit flag and define validation exit criteria. | "'release' excludes experimental-only lanes, " |
| P2 | 78 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:504` | Validation WG | Gate behind explicit flag and define validation exit criteria. | "'research' runs experimental-only lanes, " |
| P2 | 78 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:676` | Validation WG | Gate behind explicit flag and define validation exit criteria. | help="Skip pytest experimental-only lane (tests/ -m experimental).", |
| P2 | 77 | `other` | `DEPRECATED` | `CHANGELOG.md:35` | Architecture WG | Replace default path or remove lane before next major release. | - Runtime hardening: deprecated FNO suppressor now logs missing-weight fallback via standard `logging` without raising when optional weig... |
| P2 | 77 | `other` | `DEPRECATED` | `CHANGELOG.md:72` | Architecture WG | Replace default path or remove lane before next major release. | - Regression test for deprecated FNO-controller missing-weight fallback path. |
| P2 | 77 | `validation` | `DEPRECATED` | `tools/generate_source_p0p1_issue_backlog.py:78` | Validation WG | Replace default path or remove lane before next major release. | if "DEPRECATED" in issue.markers: |
| P2 | 77 | `validation` | `DEPRECATED` | `tools/generate_source_p0p1_issue_backlog.py:192` | Validation WG | Replace default path or remove lane before next major release. | items.append("Remove deprecated runtime-default path or replace with validated default lane.") |
| P2 | 74 | `other` | `SIMPLIFIED` | `CHANGELOG.md:155` | Architecture WG | Upgrade with higher-fidelity closure or tighten domain contract. | - **D-T Reactivity Fix**: Replaced simplified Huba fit with NRL Plasma Formulary 5-coefficient Bosch-Hale parameterisation for `sigma_v_d... |
| P2 | 73 | `validation` | `FALLBACK` | `tools/download_diiid_data.py:230` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | logger.debug("tokamak_archive fallback failed: %s", exc) |
| P2 | 73 | `validation` | `FALLBACK` | `tools/fallback_budget_guard.py:329` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | help="Optional runtime fallback telemetry snapshot JSON.", |
| P2 | 73 | `validation` | `FALLBACK` | `tools/generate_fno_qlknn_spatial.py:142` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | print(" Will use critical-gradient fallback (less accurate).") |
| P2 | 73 | `validation` | `FALLBACK` | `tools/generate_source_p0p1_issue_backlog.py:82` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | if "FALLBACK" in issue.markers: |
| P2 | 73 | `validation` | `FALLBACK` | `tools/generate_source_p0p1_issue_backlog.py:83` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | metrics.append("Fallback budget guard remains green for impacted lane (tools/fallback_budget_guard.py).") |
| P2 | 73 | `validation` | `FALLBACK` | `tools/generate_source_p0p1_issue_backlog.py:198` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | items.append("Record fallback telemetry and enforce strict-backend parity where applicable.") |
| P2 | 73 | `validation` | `FALLBACK` | `validation/benchmark_disturbance_rejection.py:26` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | ``get_radial_robust_controller()`` with LQR fallback. |
| P2 | 73 | `validation` | `FALLBACK` | `validation/benchmark_disturbance_rejection.py:729` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | f" [H-infinity] ARE failed ({exc}); using LQR fallback" |
| P2 | 73 | `validation` | `FALLBACK` | `validation/benchmark_sparc_geqdsk_rmse.py:207` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | except Exception as exc: # pragma: no cover - import failure handled via synthetic fallback |
| P2 | 73 | `validation` | `FALLBACK` | `validation/benchmark_vs_freegs.py:628` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | fallback = run_solovev_case(case) |
| P2 | 73 | `validation` | `FALLBACK` | `validation/benchmark_vs_freegs.py:630` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | "psi": fallback["psi"], |
| P2 | 73 | `validation` | `FALLBACK` | `validation/benchmark_vs_freegs.py:631` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | "R": fallback["R"], |
| P2 | 73 | `validation` | `FALLBACK` | `validation/benchmark_vs_freegs.py:632` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | "Z": fallback["Z"], |
| P2 | 73 | `validation` | `FALLBACK` | `validation/benchmark_vs_freegs.py:633` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | "R_axis": fallback["R_axis"], |
| P2 | 73 | `validation` | `FALLBACK` | `validation/benchmark_vs_freegs.py:634` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | "Z_axis": fallback["Z_axis"], |
| P2 | 73 | `validation` | `FALLBACK` | `validation/benchmark_vs_freegs.py:635` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | "q_proxy": fallback["q_proxy"], |
| P2 | 70 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:238` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | - Added research-only pytest marker contract (`@pytest.mark.experimental`) |
| P2 | 70 | `validation` | `EXPERIMENTAL` | `tools/generate_source_p0p1_issue_backlog.py:200` | Validation WG | Gate behind explicit flag and define validation exit criteria. | items.append("Ensure release lane remains experimental-excluded unless explicitly opted in.") |
| P2 | 68 | `docs_claims` | `NOT_VALIDATED` | `docs/HONEST_SCOPE.md:37` | Docs WG | Add real-data validation campaign and publish error bars. | \| FNO turbulence \| Synthetic-data trained; **not validated against gyrokinetics** \| Proxy mapping only; archived from release lane in v3.9 \| |
| P3 | 65 | `other` | `FALLBACK` | `CHANGELOG.md:21` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | - Runtime hardening: compiler git SHA probe now uses bounded subprocess timeout with deterministic fallback. |
| P3 | 65 | `other` | `FALLBACK` | `CHANGELOG.md:29` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | - Tooling hardening: claims audit git file discovery now uses a bounded subprocess timeout with safe fallback. |
| P3 | 65 | `other` | `FALLBACK` | `CHANGELOG.md:35` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | - Runtime hardening: deprecated FNO suppressor now logs missing-weight fallback via standard `logging` without raising when optional weig... |
| P3 | 65 | `other` | `FALLBACK` | `CHANGELOG.md:40` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | - Validation hardening: TORAX cross-validation fallback now uses deterministic stable seeding (BLAKE2-based) instead of Python process-ra... |
| P3 | 65 | `other` | `FALLBACK` | `CHANGELOG.md:41` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | - Validation observability: TORAX benchmark artifacts now expose per-case backend, fallback reason, and fallback seed fields. |
| P3 | 65 | `other` | `FALLBACK` | `CHANGELOG.md:42` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | - Validation hardening: TORAX benchmark adds strict backend gating (`--strict-backend`) to fail when reduced-order fallback is used. |
| P3 | 65 | `other` | `FALLBACK` | `CHANGELOG.md:43` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | - Validation hardening: SPARC GEQDSK RMSE benchmark fallback no longer uses identity-like reconstruction; it now uses a deterministic red... |
| P3 | 65 | `other` | `FALLBACK` | `CHANGELOG.md:44` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | - Validation observability: SPARC GEQDSK RMSE benchmark now records surrogate backend and fallback reason per case, plus strict backend r... |
| P3 | 65 | `other` | `FALLBACK` | `CHANGELOG.md:60` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | - Validation hardening: FreeGS benchmark artifacts now include backend-availability metadata, and fallback-budget guard is availability-a... |
| P3 | 65 | `other` | `FALLBACK` | `CHANGELOG.md:72` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | - Regression test for deprecated FNO-controller missing-weight fallback path. |
| P3 | 65 | `other` | `FALLBACK` | `CHANGELOG.md:73` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | - Regression tests for TORAX benchmark deterministic fallback seeding and backend/fallback metadata fields. |
| P3 | 65 | `other` | `FALLBACK` | `CHANGELOG.md:76` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | - Validation hardening: added `tools/fallback_budget_guard.py` + thresholds config and wired CI fallback-budget contract on TORAX/SPARC/F... |
| P3 | 65 | `other` | `FALLBACK` | `CHANGELOG.md:82` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | - Regression tests for expanded real-shot and FreeGS fallback-budget guard contracts. |
| P3 | 65 | `other` | `FALLBACK` | `CHANGELOG.md:114` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | - **Rust Transport Delegation**: `chang_hinton_chi_profile()` → Rust fast-path (4.7x speedup), `calculate_sauter_bootstrap_current_full()... |
| P3 | 65 | `other` | `FALLBACK` | `CHANGELOG.md:186` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | - Robust fallback to SciPy L-BFGS-B when JAX is unavailable. |
| P3 | 65 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:32` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | raise ValueError("fallback domain must be non-empty") |
| P3 | 65 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:114` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | "Fallback budget exceeded: " |
| P3 | 63 | `validation` | `PLANNED` | `validation/validate_real_shots.py:592` | Validation WG | Convert roadmap note into scheduled milestone task + owner. | f"({THRESHOLDS['disruption_fpr_max']:.0%}); tuning planned for v2.1" |
| P3 | 56 | `docs_claims` | `SIMPLIFIED` | `docs/assets/generate_header.py:27` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified Grad-Shafranov flux function with Shafranov shift + triangularity |
| P3 | 56 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/control.rst:197` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | simplified equilibrium and transport problems, used primarily for: |
| P3 | 56 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/transport.rst:82` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | using simplified ray-tracing: |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/SOLVER_TUNING_GUIDE.md:74` | Docs WG | Gate behind explicit flag and define validation exit criteria. | 4. For real experimental data, start with `0.01` and adjust based on |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/SOLVER_TUNING_GUIDE.md:98` | Docs WG | Gate behind explicit flag and define validation exit criteria. | \| **0.05–0.5** \| Standard robust range \| Real experimental data with occasional probe failures or calibration drift \| |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/api/core.rst:225` | Docs WG | Gate behind explicit flag and define validation exit criteria. | Experimental Bridges |
| P3 | 47 | `docs_claims` | `FALLBACK` | `README.md:123` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | \| Graceful degradation \| Every path has a pure-Python fallback \| |
| P3 | 47 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:167` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | - Transparent fallback to analytic model when no weights are available |
| P3 | 47 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:259` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | with fallback to CPU SIMD for systems without GPU support. |
| P3 | 47 | `docs_claims` | `FALLBACK` | `docs/HONEST_SCOPE.md:18` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | \| Graceful degradation (no Rust / no GPU / no SC-NeuroCore) \| Every module has a pure-Python fallback \| |
| P3 | 47 | `docs_claims` | `FALLBACK` | `docs/VALIDATION_GATE_MATRIX.md:29` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | until convergence hardening closes the remaining fallback/instability cases. |
| P3 | 47 | `docs_claims` | `DEPRECATED` | `docs/sphinx/changelog.rst:27` | Docs WG | Replace default path or remove lane before next major release. | - FNO turbulence surrogate deprecated (runtime ``FutureWarning``) |
| P3 | 37 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:41` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| GPU support \| Planned \| Yes (JAX) \| No \| No \| |
| P3 | 37 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:242` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| SOR red-black sweep \| wgpu compute shader \| 20–50× (65×65), 100–200× (256×256) \| P0 \| Planned \| |
| P3 | 37 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:243` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| Multigrid V-cycle \| wgpu + host orchestration \| 10–30× \| P1 \| Planned \| |
| P3 | 37 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:245` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| MLP batch inference \| wgpu or cuBLAS \| 2–5× (small H) \| P3 \| Planned \| |
| P3 | 37 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:246` | Docs WG | Convert roadmap note into scheduled milestone task + owner. | \| FNO turbulence (FFT) \| cuFFT / wgpu FFT \| 50–100× (64×64) \| P3 \| Planned \| |

## Full Register (Top 204)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/disruption_contracts.py:1` | module LOC=649 exceeds monolith threshold (500+). |
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/halo_re_physics.py:1` | module LOC=651 exceeds monolith threshold (500+). |
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/hil_harness.py:1` | module LOC=547 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fno_training.py:1` | module LOC=572 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel.py:1` | module LOC=590 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel_newton_solver.py:1` | module LOC=632 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_equilibrium.py:1` | module LOC=595 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | module LOC=509 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | module LOC=517 exceeds monolith threshold (500+). |
| P0 | `compiler_runtime` | `MONOLITH` | `src/scpn_fusion/scpn/artifact.py:1` | module LOC=583 exceeds monolith threshold (500+). |
| P0 | `compiler_runtime` | `MONOLITH` | `src/scpn_fusion/scpn/controller.py:1` | module LOC=599 exceeds monolith threshold (500+). |
| P0 | `control` | `FALLBACK_DENSITY` | `src/scpn_fusion/control/disruption_predictor.py:1` | fallback risk signals=6 across LOC=437; high fallback concentration in runtime code paths. |
| P1 | `validation` | `SIMPLIFIED` | `tools/train_fno_qlknn_spatial.py:120` | # Simplified: spectral conv in truncated mode space |
| P1 | `validation` | `SIMPLIFIED` | `validation/stress_test_campaign.py:391` | implements a 10 kHz PID control loop with simplified linear plasma |
| P1 | `validation` | `SIMPLIFIED` | `validation/test_full_pulse_scenario.py:62` | # Step Physics (Single step of simulate logic simplified) |
| P1 | `validation` | `SIMPLIFIED` | `validation/validate_iter.py:59` | # Our model is simplified (L-mode profiles mostly), so if we get > 100MW it's a "physics pass" |
| P2 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:383` | "Experimental-only pytest suite", |
| P2 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:384` | [sys.executable, "-m", "pytest", "tests/", "-q", "-m", "experimental"], |
| P2 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:503` | "'release' excludes experimental-only lanes, " |
| P2 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:504` | "'research' runs experimental-only lanes, " |
| P2 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:676` | help="Skip pytest experimental-only lane (tests/ -m experimental).", |
| P2 | `other` | `DEPRECATED` | `CHANGELOG.md:35` | - Runtime hardening: deprecated FNO suppressor now logs missing-weight fallback via standard `logging` without raising when optional weig... |
| P2 | `other` | `DEPRECATED` | `CHANGELOG.md:72` | - Regression test for deprecated FNO-controller missing-weight fallback path. |
| P2 | `validation` | `DEPRECATED` | `tools/generate_source_p0p1_issue_backlog.py:78` | if "DEPRECATED" in issue.markers: |
| P2 | `validation` | `DEPRECATED` | `tools/generate_source_p0p1_issue_backlog.py:192` | items.append("Remove deprecated runtime-default path or replace with validated default lane.") |
| P2 | `other` | `SIMPLIFIED` | `CHANGELOG.md:155` | - **D-T Reactivity Fix**: Replaced simplified Huba fit with NRL Plasma Formulary 5-coefficient Bosch-Hale parameterisation for `sigma_v_d... |
| P2 | `validation` | `FALLBACK` | `tools/download_diiid_data.py:230` | logger.debug("tokamak_archive fallback failed: %s", exc) |
| P2 | `validation` | `FALLBACK` | `tools/fallback_budget_guard.py:329` | help="Optional runtime fallback telemetry snapshot JSON.", |
| P2 | `validation` | `FALLBACK` | `tools/generate_fno_qlknn_spatial.py:142` | print(" Will use critical-gradient fallback (less accurate).") |
| P2 | `validation` | `FALLBACK` | `tools/generate_source_p0p1_issue_backlog.py:82` | if "FALLBACK" in issue.markers: |
| P2 | `validation` | `FALLBACK` | `tools/generate_source_p0p1_issue_backlog.py:83` | metrics.append("Fallback budget guard remains green for impacted lane (tools/fallback_budget_guard.py).") |
| P2 | `validation` | `FALLBACK` | `tools/generate_source_p0p1_issue_backlog.py:198` | items.append("Record fallback telemetry and enforce strict-backend parity where applicable.") |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_disturbance_rejection.py:26` | ``get_radial_robust_controller()`` with LQR fallback. |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_disturbance_rejection.py:729` | f" [H-infinity] ARE failed ({exc}); using LQR fallback" |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_sparc_geqdsk_rmse.py:207` | except Exception as exc: # pragma: no cover - import failure handled via synthetic fallback |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_vs_freegs.py:628` | fallback = run_solovev_case(case) |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_vs_freegs.py:630` | "psi": fallback["psi"], |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_vs_freegs.py:631` | "R": fallback["R"], |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_vs_freegs.py:632` | "Z": fallback["Z"], |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_vs_freegs.py:633` | "R_axis": fallback["R_axis"], |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_vs_freegs.py:634` | "Z_axis": fallback["Z_axis"], |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_vs_freegs.py:635` | "q_proxy": fallback["q_proxy"], |
| P2 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:238` | - Added research-only pytest marker contract (`@pytest.mark.experimental`) |
| P2 | `validation` | `EXPERIMENTAL` | `tools/generate_source_p0p1_issue_backlog.py:200` | items.append("Ensure release lane remains experimental-excluded unless explicitly opted in.") |
| P2 | `docs_claims` | `NOT_VALIDATED` | `docs/HONEST_SCOPE.md:37` | \| FNO turbulence \| Synthetic-data trained; **not validated against gyrokinetics** \| Proxy mapping only; archived from release lane in v3.9 \| |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:21` | - Runtime hardening: compiler git SHA probe now uses bounded subprocess timeout with deterministic fallback. |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:29` | - Tooling hardening: claims audit git file discovery now uses a bounded subprocess timeout with safe fallback. |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:35` | - Runtime hardening: deprecated FNO suppressor now logs missing-weight fallback via standard `logging` without raising when optional weig... |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:40` | - Validation hardening: TORAX cross-validation fallback now uses deterministic stable seeding (BLAKE2-based) instead of Python process-ra... |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:41` | - Validation observability: TORAX benchmark artifacts now expose per-case backend, fallback reason, and fallback seed fields. |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:42` | - Validation hardening: TORAX benchmark adds strict backend gating (`--strict-backend`) to fail when reduced-order fallback is used. |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:43` | - Validation hardening: SPARC GEQDSK RMSE benchmark fallback no longer uses identity-like reconstruction; it now uses a deterministic red... |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:44` | - Validation observability: SPARC GEQDSK RMSE benchmark now records surrogate backend and fallback reason per case, plus strict backend r... |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:60` | - Validation hardening: FreeGS benchmark artifacts now include backend-availability metadata, and fallback-budget guard is availability-a... |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:72` | - Regression test for deprecated FNO-controller missing-weight fallback path. |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:73` | - Regression tests for TORAX benchmark deterministic fallback seeding and backend/fallback metadata fields. |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:76` | - Validation hardening: added `tools/fallback_budget_guard.py` + thresholds config and wired CI fallback-budget contract on TORAX/SPARC/F... |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:82` | - Regression tests for expanded real-shot and FreeGS fallback-budget guard contracts. |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:114` | - **Rust Transport Delegation**: `chang_hinton_chi_profile()` → Rust fast-path (4.7x speedup), `calculate_sauter_bootstrap_current_full()... |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:186` | - Robust fallback to SciPy L-BFGS-B when JAX is unavailable. |
| P3 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:32` | raise ValueError("fallback domain must be non-empty") |
| P3 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:114` | "Fallback budget exceeded: " |
| P3 | `validation` | `PLANNED` | `validation/validate_real_shots.py:592` | f"({THRESHOLDS['disruption_fpr_max']:.0%}); tuning planned for v2.1" |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/assets/generate_header.py:27` | # Simplified Grad-Shafranov flux function with Shafranov shift + triangularity |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/control.rst:197` | simplified equilibrium and transport problems, used primarily for: |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/transport.rst:82` | using simplified ray-tracing: |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/SOLVER_TUNING_GUIDE.md:74` | 4. For real experimental data, start with `0.01` and adjust based on |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/SOLVER_TUNING_GUIDE.md:98` | \| **0.05–0.5** \| Standard robust range \| Real experimental data with occasional probe failures or calibration drift \| |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/api/core.rst:225` | Experimental Bridges |
| P3 | `docs_claims` | `FALLBACK` | `README.md:123` | \| Graceful degradation \| Every path has a pure-Python fallback \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:167` | - Transparent fallback to analytic model when no weights are available |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:259` | with fallback to CPU SIMD for systems without GPU support. |
| P3 | `docs_claims` | `FALLBACK` | `docs/HONEST_SCOPE.md:18` | \| Graceful degradation (no Rust / no GPU / no SC-NeuroCore) \| Every module has a pure-Python fallback \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/VALIDATION_GATE_MATRIX.md:29` | until convergence hardening closes the remaining fallback/instability cases. |
| P3 | `docs_claims` | `DEPRECATED` | `docs/sphinx/changelog.rst:27` | - FNO turbulence surrogate deprecated (runtime ``FutureWarning``) |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:41` | \| GPU support \| Planned \| Yes (JAX) \| No \| No \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:242` | \| SOR red-black sweep \| wgpu compute shader \| 20–50× (65×65), 100–200× (256×256) \| P0 \| Planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:243` | \| Multigrid V-cycle \| wgpu + host orchestration \| 10–30× \| P1 \| Planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:245` | \| MLP batch inference \| wgpu or cuBLAS \| 2–5× (small H) \| P3 \| Planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:246` | \| FNO turbulence (FFT) \| cuFFT / wgpu FFT \| 50–100× (64×64) \| P3 \| Planned \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:282` | \| SCPN (planned) \| Quadtree, gradient-based \| 2D GS + 3D extension \| |
| P3 | `docs_claims` | `PLANNED` | `docs/BENCHMARKS.md:284` | The planned quadtree AMR is simpler than JOREK's h-p adaptivity but |
| P3 | `docs_claims` | `DEPRECATED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:26` | \| `DEPRECATED` \| 17 \| |
| P3 | `docs_claims` | `DEPRECATED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:51` | - The critical residual gaps are targeted: physics fidelity upgrades, deprecated FNO lane retirement, and validation/claim rigor. |
| P3 | `docs_claims` | `DEPRECATED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:59` | | `src/scpn_fusion/core/fno_turbulence_suppressor.py` | 3 (`DEPRECATED`, `NOT_VALIDATED`, `SIMPLIFIED`) | Highest-credibility physics lan... |
| P3 | `docs_claims` | `DEPRECATED` | `docs/HARDENING_30_DAY_EXECUTION_PLAN.md:16` | - Lock default turbulence path to non-deprecated lanes only. |
| P3 | `docs_claims` | `DEPRECATED` | `docs/HARDENING_30_DAY_EXECUTION_PLAN.md:17` | - Keep deprecated FNO paths non-default and explicitly gated. |
| P3 | `docs_claims` | `PLANNED` | `docs/HONEST_SCOPE.md:25` | \| Full 3D MHD \| Not planned for real-time loop \| Use NIMROD/M3D-C1 externally \| |
| P3 | `docs_claims` | `PLANNED` | `docs/competitive_analysis.md:77` | \| GPU acceleration \| Planned (wgpu) \| Yes (JAX) \| No \| No \| JAX \| No \| |
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
| P3 | `docs_claims` | `FALLBACK` | `docs/3d_gaps.md:92` | - fallback if external dependency is unavailable. |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:93` | │ ╱ Fallback (analytic) |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:124` | \| Fallback vectorised \| ~2 ms \| ~7× slower \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:125` | \| Fallback point-by-point loop \| ~200 ms \| ~670× slower \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/CI_FAILURE_LEDGER.md:25` | | CI-2026-03-03-006 | 2026-03-03 19:18 | 22638800884 (Python 3.9/3.10) | `ModuleNotFoundError: No module named 'tomllib'` from `tools/che... |
| P3 | `docs_claims` | `FALLBACK` | `docs/GPU_ACCELERATION_ROADMAP.md:35` | - Host orchestration in Rust with deterministic CPU fallback |
| P3 | `docs_claims` | `FALLBACK` | `docs/GPU_ACCELERATION_ROADMAP.md:64` | - CPU fallback for unsupported nodes |
| P3 | `docs_claims` | `FALLBACK` | `docs/GPU_ACCELERATION_ROADMAP.md:122` | - automatic CPU fallback |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURAL_TRANSPORT_TRAINING.md:112` | - fallback to analytic critical-gradient model on any contract failure |
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
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/gpu_roadmap.rst:18` | 1. ``wgpu`` SOR kernel (red-black stencil + deterministic CPU fallback) |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/gpu_roadmap.rst:19` | 2. GPU-backed GMRES preconditioning (CUDA/ROCm adapters with CPU fallback) |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/gpu_roadmap.rst:39` | - Operations: runtime capability detection + automatic CPU fallback |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/index.rst:54` | speedups with pure-Python fallback |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/installation.rst:50` | INFO: scpn_fusion_rs not found -- using NumPy fallback. |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/userguide/control.rst:89` | - **Checkpoint fallback** ensuring graceful degradation if the model |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/userguide/diagnostics.rst:97` | with automatic fallback for environments where SciPy is not optimised. |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/userguide/hpc.rst:102` | WebGPU) with deterministic CPU fallback. |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/userguide/hpc.rst:110` | CUDA/ROCm adapters for the GMRES linear solver with CPU fallback |
| P3 | `docs_claims` | `FALLBACK` | `docs/sphinx/userguide/hpc.rst:129` | - Operations: runtime capability detection + automatic CPU fallback |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:24` | \| `SIMPLIFIED` \| 27 \| |
| P3 | `docs_claims` | `NOT_VALIDATED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:59` | | `src/scpn_fusion/core/fno_turbulence_suppressor.py` | 3 (`DEPRECATED`, `NOT_VALIDATED`, `SIMPLIFIED`) | Highest-credibility physics lan... |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:59` | | `src/scpn_fusion/core/fno_turbulence_suppressor.py` | 3 (`DEPRECATED`, `NOT_VALIDATED`, `SIMPLIFIED`) | Highest-credibility physics lan... |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:60` | \| `src/scpn_fusion/core/integrated_transport_solver.py` \| 1 (`SIMPLIFIED`) \| Largest physics file, central to transport claims \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:61` | \| `src/scpn_fusion/core/eped_pedestal.py` \| 1 (`SIMPLIFIED`) \| Pedestal closure fidelity bottleneck \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:62` | \| `src/scpn_fusion/core/stability_mhd.py` \| 1 (`SIMPLIFIED`) \| NTM/MHD reliability for disruption-oriented claims \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:63` | \| `src/scpn_fusion/control/fokker_planck_re.py` \| 1 (`SIMPLIFIED`) \| Runaway electron fidelity limits hard-safety narratives \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:64` | \| `src/scpn_fusion/control/spi_ablation.py` \| 1 (`SIMPLIFIED`) \| SPI mitigation realism in disruption campaigns \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:65` | \| `src/scpn_fusion/nuclear/blanket_neutronics.py` \| 1 (`SIMPLIFIED`) \| TBR confidence boundary \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:66` | \| `src/scpn_fusion/nuclear/nuclear_wall_interaction.py` \| 1 (`SIMPLIFIED`) \| PWI claims bounded by reduced model assumptions \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:72` | - `SIMPLIFIED`: 19 mentions |
| P3 | `docs_claims` | `NOT_VALIDATED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:1731` | 3. **Schema validation.** The JSON Schema (`scpnctl.schema.json`) is not validated against the artifact files in the test suite. Schema v... |
| P3 | `docs_claims` | `NOT_VALIDATED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:3084` | 5. **No schema validation.** The JSON Schema exists but is not validated at load time. This means a malformed artifact that satisfies the... |
| P3 | `docs_claims` | `FALLBACK` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:22` | \| `FALLBACK` \| 172 \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/DEEP_AUDIT_AND_SOTA_PLAN_2026-03-01.md:71` | - `FALLBACK`: 67 mentions |
| P3 | `docs_claims` | `FALLBACK` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:103` | - Transparent fallback to full physics solve when surrogate confidence is below threshold |
| P3 | `docs_claims` | `FALLBACK` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:276` | - f32 precision with tolerance-guarded fallback to f64 CPU path |
| P3 | `docs_claims` | `FALLBACK` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:508` | **2. Rust Performance with Python Accessibility:** The dual-language architecture provides 10-50x performance over pure-Python alternativ... |
| P3 | `docs_claims` | `PLANNED` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:690` | **Packet C — `scpn/runtime.py` — PetriNetEngine** (planned): |
| P3 | `docs_claims` | `PLANNED` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:704` | **Packet D — `examples/01_traffic_light.py` — Hello World Demo** (planned): |
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
| P3 | `docs_claims` | `PLANNED` | `docs/VALIDATION_AGAINST_ITER.md:238` | Planned follow-up after WP-E1: |
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
