# Underdeveloped Register

- Generated at: `2026-03-14T18:52:51.682552+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: production code + docs claims markers (tests/reports/html excluded)

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 96 |
| P0 + P1 entries | 29 |
| Source-domain entries | 41 |
| Source-domain P0 + P1 entries | 29 |
| Docs-claims entries | 55 |
| Domains affected | 5 |

## Marker Distribution

| Key | Count |
|---|---:|
| `FALLBACK` | 36 |
| `SIMPLIFIED` | 26 |
| `PLANNED` | 12 |
| `EXPERIMENTAL` | 8 |
| `MONOLITH` | 7 |
| `DEPRECATED` | 5 |
| `NOT_VALIDATED` | 2 |

## Domain Distribution

| Key | Count |
|---|---:|
| `docs_claims` | 55 |
| `core_physics` | 23 |
| `control` | 9 |
| `validation` | 8 |
| `diagnostics_io` | 1 |

## Source-Centric Priority Backlog (Top 29)

_Filtered to implementation domains to reduce docs/claims noise during hardening triage._

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 110 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=1419 exceeds monolith threshold (500+). |
| P0 | 110 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_tracking.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=1566 exceeds monolith threshold (500+). |
| P0 | 109 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver_model.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=884 exceeds monolith threshold (500+). |
| P0 | 109 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=849 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=526 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=669 exceeds monolith threshold (500+). |
| P0 | 99 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/tokamak_archive.py:1` | Diagnostics/IO WG | Split module into focused subcomponents and lock interface contracts. | module LOC=501 exceeds monolith threshold (500+). |
| P0 | 96 | `validation` | `EXPERIMENTAL` | `tools/train_neural_equilibrium_gpu.py:742` | Validation WG | Gate behind explicit flag and define validation exit criteria. | print("\n ACCEPTANCE CRITERIA NOT MET — weights saved as experimental") |
| P1 | 86 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/density_controller.py:213` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified prediction: static + noise |
| P1 | 86 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/mu_synthesis.py:146` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified: u = -K * x |
| P1 | 86 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/realtime_efit.py:99` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | """Simplified real-time equilibrium reconstruction (EFIT).""" |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/alfven_eigenmodes.py:251` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | omega_BAE = 0.1 * v_A / R0 # Highly simplified |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/disruption_sequence.py:111` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Extremely simplified mapping: drops to few eV via line radiation of impurities. |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/disruption_sequence.py:208` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | """F_z [MN] = I_halo * B_tor * 2pi R0 * TPF (simplified)""" |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:25` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | Critical edge current density limit (simplified scaling). |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:53` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified PB boundary: j_norm^2 + a_norm^2 > 1 |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:145` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Very simplified representation of overlap across outer rational surfaces |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/impurity_transport.py:170` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | rhs[-1] = s.source_rate * dt / dr # Very simplified edge source mapping |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/lh_transition.py:48` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Drive scales with pressure gradient, simplified here to scale with p |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/lh_transition.py:149` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Highly simplified heuristic logic |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/locked_mode.py:32` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified linear correction |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/locked_mode.py:72` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Very simplified proxy formula retaining the B_res^2 scaling |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/momentum_transport.py:110` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified criterion: omega_phi * tau_wall > 1% of something? |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neural_turbulence.py:177` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # highly simplified nu_star |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neural_turbulence.py:224` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | Compute flux targets from simplified analytical quasilinear model. |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/plasma_wall_interaction.py:28` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified threshold energy |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/plasma_wall_interaction.py:44` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Using a simplified parameterized curve from Eckstein |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/plasma_wall_interaction.py:178` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Highly simplified fatigue life mapping |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/vmec_lite.py:49` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | Highly simplified spectral 3D equilibrium solver mimicking VMEC principles. |

## Top Priority Backlog (Top 80)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 110 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=1419 exceeds monolith threshold (500+). |
| P0 | 110 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_tracking.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=1566 exceeds monolith threshold (500+). |
| P0 | 109 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver_model.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=884 exceeds monolith threshold (500+). |
| P0 | 109 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=849 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=526 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=669 exceeds monolith threshold (500+). |
| P0 | 99 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/tokamak_archive.py:1` | Diagnostics/IO WG | Split module into focused subcomponents and lock interface contracts. | module LOC=501 exceeds monolith threshold (500+). |
| P0 | 96 | `validation` | `EXPERIMENTAL` | `tools/train_neural_equilibrium_gpu.py:742` | Validation WG | Gate behind explicit flag and define validation exit criteria. | print("\n ACCEPTANCE CRITERIA NOT MET — weights saved as experimental") |
| P1 | 86 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/density_controller.py:213` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified prediction: static + noise |
| P1 | 86 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/mu_synthesis.py:146` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified: u = -K * x |
| P1 | 86 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/realtime_efit.py:99` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | """Simplified real-time equilibrium reconstruction (EFIT).""" |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/alfven_eigenmodes.py:251` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | omega_BAE = 0.1 * v_A / R0 # Highly simplified |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/disruption_sequence.py:111` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Extremely simplified mapping: drops to few eV via line radiation of impurities. |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/disruption_sequence.py:208` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | """F_z [MN] = I_halo * B_tor * 2pi R0 * TPF (simplified)""" |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:25` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | Critical edge current density limit (simplified scaling). |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:53` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified PB boundary: j_norm^2 + a_norm^2 > 1 |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:145` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Very simplified representation of overlap across outer rational surfaces |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/impurity_transport.py:170` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | rhs[-1] = s.source_rate * dt / dr # Very simplified edge source mapping |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/lh_transition.py:48` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Drive scales with pressure gradient, simplified here to scale with p |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/lh_transition.py:149` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Highly simplified heuristic logic |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/locked_mode.py:32` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified linear correction |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/locked_mode.py:72` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Very simplified proxy formula retaining the B_res^2 scaling |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/momentum_transport.py:110` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified criterion: omega_phi * tau_wall > 1% of something? |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neural_turbulence.py:177` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # highly simplified nu_star |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neural_turbulence.py:224` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | Compute flux targets from simplified analytical quasilinear model. |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/plasma_wall_interaction.py:28` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified threshold energy |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/plasma_wall_interaction.py:44` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Using a simplified parameterized curve from Eckstein |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/plasma_wall_interaction.py:178` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Highly simplified fatigue life mapping |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/vmec_lite.py:49` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | Highly simplified spectral 3D equilibrium solver mimicking VMEC principles. |
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:35` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | SUPERVISORY_ALERT_LEVEL_NAMES = ("nominal", "warning", "guarded", "fallback") |
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_tracking.py:13` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | rejection can ramp the coil set toward explicit safe fallback currents instead |
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_tracking.py:998` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | raise ValueError("fallback currents are not configured.") |
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_tracking.py:1430` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | f"lag={max_abs_actuator_lag:.3e} \| fallback={fallback_active} \| " |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/disruption_sequence.py:266` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | seed = 1e10 # Fallback |
| P2 | 73 | `validation` | `FALLBACK` | `tools/download_diiid_data.py:235` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | logger.debug("tokamak_archive fallback failed: %s", exc) |
| P2 | 73 | `validation` | `FALLBACK` | `tools/generate_fno_qlknn_spatial.py:157` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | print(" Will use critical-gradient fallback (less accurate).") |
| P2 | 73 | `validation` | `FALLBACK` | `validation/free_boundary_tracking_acceptance.py:1882` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | f"- Fallback active steps: `{summary['fallback_active_steps']}`", |
| P2 | 73 | `validation` | `FALLBACK` | `validation/task11_free_boundary_constraint_safety.py:238` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | f"- Fallback mode count: `{c['fallback_mode_count']}` (threshold `>= {th['min_fallback_mode_count']}`)", |
| P2 | 73 | `validation` | `FALLBACK` | `validation/task12_free_boundary_physics_margin_safety.py:247` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | f"- Fallback mode count: `{c['fallback_mode_count']}` (threshold `>= {th['min_fallback_mode_count']}`)", |
| P2 | 73 | `validation` | `FALLBACK` | `validation/task13_free_boundary_disruption_policy_recovery.py:279` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | f"- fallback mode count: `{c['fallback_mode_count']}` (threshold `>= {th['min_fallback_mode_count']}`)", |
| P2 | 73 | `validation` | `FALLBACK` | `validation/task14_free_boundary_failsafe_dropout_replay.py:286` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | f"- Fallback mode count: `{c['fallback_mode_count']}` (threshold `>= {th['min_fallback_mode_count']}`)", |
| P3 | 65 | `docs_claims` | `DEPRECATED` | `docs/PHYSICS_VALIDATION_STATUS.md:278` | Docs WG | Replace default path or remove lane before next major release. | \| `fno_eurofusion_jet` \| 2026-02-16 \| rel_L2=0.79 \| No (DEPRECATED) \| |
| P3 | 65 | `docs_claims` | `DEPRECATED` | `docs/PHYSICS_VALIDATION_STATUS.md:283` | Docs WG | Replace default path or remove lane before next major release. | The FNO is already DEPRECATED (v4.0 decision: retrain on real data or remove). |
| P3 | 65 | `docs_claims` | `DEPRECATED` | `docs/PHYSICS_VALIDATION_STATUS.md:326` | Docs WG | Replace default path or remove lane before next major release. | \| FNO turbulence \| Synthetic Hasegawa-Wakatani \| **No** (DEPRECATED) \| |
| P3 | 56 | `docs_claims` | `SIMPLIFIED` | `docs/assets/generate_header.py:27` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified Grad-Shafranov flux function with Shafranov shift + triangularity |
| P3 | 56 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/tutorials/current_profile_evolution.rst:131` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Bootstrap current (simplified model) |
| P3 | 56 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/tutorials/realtime_reconstruction.rst:45` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Define diagnostic positions (simplified: 12 flux loops, 8 B_p probes) |
| P3 | 56 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/control.rst:197` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | simplified equilibrium and transport problems, used primarily for: |
| P3 | 56 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/transport.rst:82` | Docs WG | Upgrade with higher-fidelity closure or tighten domain contract. | using simplified ray-tracing: |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md:47` | Docs WG | Gate behind explicit flag and define validation exit criteria. | --experimental \ |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md:48` | Docs WG | Gate behind explicit flag and define validation exit criteria. | --experimental-ack I_UNDERSTAND_EXPERIMENTAL \ |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/PHYSICS_VALIDATION_STATUS.md:224` | Docs WG | Gate behind explicit flag and define validation exit criteria. | \| Multi-Regime FNO \| `src/.../fno_training_multi_regime.py` \| NumPy \| Synthetic H-W \| 30-90 min \| Experimental \| |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/PHYSICS_VALIDATION_STATUS.md:290` | Docs WG | Gate behind explicit flag and define validation exit criteria. | 4. **Multi-Regime FNO** — experimental, synthetic data only. |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/api/core.rst:225` | Docs WG | Gate behind explicit flag and define validation exit criteria. | Experimental Bridges |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/learning/first_simulation.rst:191` | Docs WG | Gate behind explicit flag and define validation exit criteria. | Step 7: Read Real Experimental Data |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/learning/tokamak_physics_textbook.rst:55` | Docs WG | Gate behind explicit flag and define validation exit criteria. | :math:`F(\psi)` are constrained by experimental data (pressure profile and |
| P3 | 47 | `docs_claims` | `FALLBACK` | `docs/VALIDATION_GATE_MATRIX.md:19` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | | `freegs-strict` | FreeGS-only strict backend parity lane with runtime-fallback disallowed and artifact contract checks (`mode=freegs`, ... |
| P3 | 47 | `docs_claims` | `FALLBACK` | `docs/VALIDATION_GATE_MATRIX.md:29` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | | `freegs-strict` (`freegs-strict.yml`, manual dispatch) | Strict FreeGS backend parity lane; fails on any fallback or non-FreeGS referen... |
| P3 | 47 | `docs_claims` | `FALLBACK` | `docs/VALIDATION_GATE_MATRIX.md:36` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | enforces a no-fallback contract when invoked. |
| P3 | 37 | `docs_claims` | `DEPRECATED` | `docs/HARDENING_30_DAY_EXECUTION_PLAN.md:16` | Docs WG | Replace default path or remove lane before next major release. | - Lock default turbulence path to non-deprecated lanes only. |
| P3 | 37 | `docs_claims` | `DEPRECATED` | `docs/HARDENING_30_DAY_EXECUTION_PLAN.md:17` | Docs WG | Replace default path or remove lane before next major release. | - Keep deprecated FNO paths non-default and explicitly gated. |
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

## Full Register (Top 96)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:1` | module LOC=1419 exceeds monolith threshold (500+). |
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_tracking.py:1` | module LOC=1566 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver_model.py:1` | module LOC=884 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | module LOC=849 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | module LOC=526 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | module LOC=669 exceeds monolith threshold (500+). |
| P0 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/tokamak_archive.py:1` | module LOC=501 exceeds monolith threshold (500+). |
| P0 | `validation` | `EXPERIMENTAL` | `tools/train_neural_equilibrium_gpu.py:742` | print("\n ACCEPTANCE CRITERIA NOT MET — weights saved as experimental") |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/density_controller.py:213` | # Simplified prediction: static + noise |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/mu_synthesis.py:146` | # Simplified: u = -K * x |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/realtime_efit.py:99` | """Simplified real-time equilibrium reconstruction (EFIT).""" |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/alfven_eigenmodes.py:251` | omega_BAE = 0.1 * v_A / R0 # Highly simplified |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/disruption_sequence.py:111` | # Extremely simplified mapping: drops to few eV via line radiation of impurities. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/disruption_sequence.py:208` | """F_z [MN] = I_halo * B_tor * 2pi R0 * TPF (simplified)""" |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:25` | Critical edge current density limit (simplified scaling). |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:53` | # Simplified PB boundary: j_norm^2 + a_norm^2 > 1 |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:145` | # Very simplified representation of overlap across outer rational surfaces |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/impurity_transport.py:170` | rhs[-1] = s.source_rate * dt / dr # Very simplified edge source mapping |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/lh_transition.py:48` | # Drive scales with pressure gradient, simplified here to scale with p |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/lh_transition.py:149` | # Highly simplified heuristic logic |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/locked_mode.py:32` | # Simplified linear correction |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/locked_mode.py:72` | # Very simplified proxy formula retaining the B_res^2 scaling |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/momentum_transport.py:110` | # Simplified criterion: omega_phi * tau_wall > 1% of something? |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neural_turbulence.py:177` | # highly simplified nu_star |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neural_turbulence.py:224` | Compute flux targets from simplified analytical quasilinear model. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/plasma_wall_interaction.py:28` | # Simplified threshold energy |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/plasma_wall_interaction.py:44` | # Using a simplified parameterized curve from Eckstein |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/plasma_wall_interaction.py:178` | # Highly simplified fatigue life mapping |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/vmec_lite.py:49` | Highly simplified spectral 3D equilibrium solver mimicking VMEC principles. |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:35` | SUPERVISORY_ALERT_LEVEL_NAMES = ("nominal", "warning", "guarded", "fallback") |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_tracking.py:13` | rejection can ramp the coil set toward explicit safe fallback currents instead |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_tracking.py:998` | raise ValueError("fallback currents are not configured.") |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_tracking.py:1430` | f"lag={max_abs_actuator_lag:.3e} \| fallback={fallback_active} \| " |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/disruption_sequence.py:266` | seed = 1e10 # Fallback |
| P2 | `validation` | `FALLBACK` | `tools/download_diiid_data.py:235` | logger.debug("tokamak_archive fallback failed: %s", exc) |
| P2 | `validation` | `FALLBACK` | `tools/generate_fno_qlknn_spatial.py:157` | print(" Will use critical-gradient fallback (less accurate).") |
| P2 | `validation` | `FALLBACK` | `validation/free_boundary_tracking_acceptance.py:1882` | f"- Fallback active steps: `{summary['fallback_active_steps']}`", |
| P2 | `validation` | `FALLBACK` | `validation/task11_free_boundary_constraint_safety.py:238` | f"- Fallback mode count: `{c['fallback_mode_count']}` (threshold `>= {th['min_fallback_mode_count']}`)", |
| P2 | `validation` | `FALLBACK` | `validation/task12_free_boundary_physics_margin_safety.py:247` | f"- Fallback mode count: `{c['fallback_mode_count']}` (threshold `>= {th['min_fallback_mode_count']}`)", |
| P2 | `validation` | `FALLBACK` | `validation/task13_free_boundary_disruption_policy_recovery.py:279` | f"- fallback mode count: `{c['fallback_mode_count']}` (threshold `>= {th['min_fallback_mode_count']}`)", |
| P2 | `validation` | `FALLBACK` | `validation/task14_free_boundary_failsafe_dropout_replay.py:286` | f"- Fallback mode count: `{c['fallback_mode_count']}` (threshold `>= {th['min_fallback_mode_count']}`)", |
| P3 | `docs_claims` | `DEPRECATED` | `docs/PHYSICS_VALIDATION_STATUS.md:278` | \| `fno_eurofusion_jet` \| 2026-02-16 \| rel_L2=0.79 \| No (DEPRECATED) \| |
| P3 | `docs_claims` | `DEPRECATED` | `docs/PHYSICS_VALIDATION_STATUS.md:283` | The FNO is already DEPRECATED (v4.0 decision: retrain on real data or remove). |
| P3 | `docs_claims` | `DEPRECATED` | `docs/PHYSICS_VALIDATION_STATUS.md:326` | \| FNO turbulence \| Synthetic Hasegawa-Wakatani \| **No** (DEPRECATED) \| |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/assets/generate_header.py:27` | # Simplified Grad-Shafranov flux function with Shafranov shift + triangularity |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/tutorials/current_profile_evolution.rst:131` | # Bootstrap current (simplified model) |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/tutorials/realtime_reconstruction.rst:45` | # Define diagnostic positions (simplified: 12 flux loops, 8 B_p probes) |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/control.rst:197` | simplified equilibrium and transport problems, used primarily for: |
| P3 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/transport.rst:82` | using simplified ray-tracing: |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md:47` | --experimental \ |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md:48` | --experimental-ack I_UNDERSTAND_EXPERIMENTAL \ |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/PHYSICS_VALIDATION_STATUS.md:224` | \| Multi-Regime FNO \| `src/.../fno_training_multi_regime.py` \| NumPy \| Synthetic H-W \| 30-90 min \| Experimental \| |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/PHYSICS_VALIDATION_STATUS.md:290` | 4. **Multi-Regime FNO** — experimental, synthetic data only. |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/api/core.rst:225` | Experimental Bridges |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/learning/first_simulation.rst:191` | Step 7: Read Real Experimental Data |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/learning/tokamak_physics_textbook.rst:55` | :math:`F(\psi)` are constrained by experimental data (pressure profile and |
| P3 | `docs_claims` | `FALLBACK` | `docs/VALIDATION_GATE_MATRIX.md:19` | | `freegs-strict` | FreeGS-only strict backend parity lane with runtime-fallback disallowed and artifact contract checks (`mode=freegs`, ... |
| P3 | `docs_claims` | `FALLBACK` | `docs/VALIDATION_GATE_MATRIX.md:29` | | `freegs-strict` (`freegs-strict.yml`, manual dispatch) | Strict FreeGS backend parity lane; fails on any fallback or non-FreeGS referen... |
| P3 | `docs_claims` | `FALLBACK` | `docs/VALIDATION_GATE_MATRIX.md:36` | enforces a no-fallback contract when invoked. |
| P3 | `docs_claims` | `DEPRECATED` | `docs/HARDENING_30_DAY_EXECUTION_PLAN.md:16` | - Lock default turbulence path to non-deprecated lanes only. |
| P3 | `docs_claims` | `DEPRECATED` | `docs/HARDENING_30_DAY_EXECUTION_PLAN.md:17` | - Keep deprecated FNO paths non-default and explicitly gated. |
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
