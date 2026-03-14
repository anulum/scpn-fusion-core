# Underdeveloped Register

- Generated at: `2026-03-14T22:12:45.596671+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 50 |
| P0 + P1 entries | 43 |
| Source-domain entries | 50 |
| Source-domain P0 + P1 entries | 43 |
| Docs-claims entries | 0 |
| Domains affected | 4 |

## Marker Distribution

| Key | Count |
|---|---:|
| `SIMPLIFIED` | 35 |
| `FALLBACK` | 7 |
| `MONOLITH` | 7 |
| `TEST_GAP` | 1 |

## Domain Distribution

| Key | Count |
|---|---:|
| `core_physics` | 37 |
| `control` | 11 |
| `compiler_runtime` | 1 |
| `diagnostics_io` | 1 |

## Top Priority Backlog (Top 50)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 110 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=1419 exceeds monolith threshold (500+). |
| P0 | 110 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_tracking.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=1566 exceeds monolith threshold (500+). |
| P0 | 109 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver_model.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=884 exceeds monolith threshold (500+). |
| P0 | 109 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=849 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=526 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=669 exceeds monolith threshold (500+). |
| P0 | 99 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/tokamak_archive.py:1` | Diagnostics/IO WG | Split module into focused subcomponents and lock interface contracts. | module LOC=501 exceeds monolith threshold (500+). |
| P0 | 98 | `compiler_runtime` | `TEST_GAP` | `src/scpn_fusion/hpc/hpc_bridge.py:1` | Runtime WG | Add direct module tests and eliminate allowlist-only linkage. | large source module without direct test import/stem linkage. |
| P1 | 86 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/burn_controller.py:136` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified: rely on integral to find it |
| P1 | 86 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/controller_tuning.py:61` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | action = kp * error # Simplified |
| P1 | 86 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/density_controller.py:213` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified prediction: static + noise |
| P1 | 86 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/mu_synthesis.py:146` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified: u = -K * x |
| P1 | 86 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/realtime_efit.py:99` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | """Simplified real-time equilibrium reconstruction (EFIT).""" |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/alfven_eigenmodes.py:251` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | omega_BAE = 0.1 * v_A / R0 # Highly simplified |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/disruption_sequence.py:111` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Extremely simplified mapping: drops to few eV via line radiation of impurities. |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/disruption_sequence.py:208` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | """F_z [MN] = I_halo * B_tor * 2pi R0 * TPF (simplified)""" |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:25` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | Critical edge current density limit (simplified scaling). |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:53` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified PB boundary: j_norm^2 + a_norm^2 > 1 |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:145` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Very simplified representation of overlap across outer rational surfaces |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_eigenvalue.py:205` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Collision operator (simplified: nu_D * pitch-angle scattering applied to theta structure) |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_geometry.py:121` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Poloidal field: B_p = r / (q * R_s * \|J/r\|) — simplified |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_geometry.py:127` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # b . grad(theta) = B_p / (R_s * \|grad theta\|) ≈ 1 / (q * R_s) simplified |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_online_learner.py:138` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Backprop (simplified, output layer only for stability) |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_species.py:13` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | simplified Sugama model (pitch-angle scattering + energy diffusion). |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_species.py:157` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | Simplified Sugama model: |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_species.py:166` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified to order-of-magnitude for the linearised operator |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_species.py:179` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | nu_E = nu_D # simplified: energy diffusion ~ pitch-angle for like-species |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/impurity_transport.py:170` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | rhs[-1] = s.source_rate * dt / dr # Very simplified edge source mapping |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_scenario.py:35` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # CD parameters (simplified) |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_scenario.py:245` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | j_bs = np.zeros(self.nr) # simplified |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_scenario.py:255` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | q_peak = sol_res.q_parallel_MW_m2 # simplified |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/lh_transition.py:48` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Drive scales with pressure gradient, simplified here to scale with p |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/lh_transition.py:149` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Highly simplified heuristic logic |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/locked_mode.py:32` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified linear correction |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/locked_mode.py:72` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Very simplified proxy formula retaining the B_res^2 scaling |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/momentum_transport.py:110` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified criterion: omega_phi * tau_wall > 1% of something? |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neoclassical.py:208` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified for this module |
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
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/gk_tglf.py:204` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | _logger.warning("TGLF binary not found, returning fallback") |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/jax_gs_solver.py:24` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | NumPy fallback provided when JAX is unavailable. |

## Full Register (Top 50)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:1` | module LOC=1419 exceeds monolith threshold (500+). |
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_tracking.py:1` | module LOC=1566 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver_model.py:1` | module LOC=884 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | module LOC=849 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | module LOC=526 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | module LOC=669 exceeds monolith threshold (500+). |
| P0 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/tokamak_archive.py:1` | module LOC=501 exceeds monolith threshold (500+). |
| P0 | `compiler_runtime` | `TEST_GAP` | `src/scpn_fusion/hpc/hpc_bridge.py:1` | large source module without direct test import/stem linkage. |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/burn_controller.py:136` | # Simplified: rely on integral to find it |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/controller_tuning.py:61` | action = kp * error # Simplified |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/density_controller.py:213` | # Simplified prediction: static + noise |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/mu_synthesis.py:146` | # Simplified: u = -K * x |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/realtime_efit.py:99` | """Simplified real-time equilibrium reconstruction (EFIT).""" |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/alfven_eigenmodes.py:251` | omega_BAE = 0.1 * v_A / R0 # Highly simplified |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/disruption_sequence.py:111` | # Extremely simplified mapping: drops to few eV via line radiation of impurities. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/disruption_sequence.py:208` | """F_z [MN] = I_halo * B_tor * 2pi R0 * TPF (simplified)""" |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:25` | Critical edge current density limit (simplified scaling). |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:53` | # Simplified PB boundary: j_norm^2 + a_norm^2 > 1 |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:145` | # Very simplified representation of overlap across outer rational surfaces |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_eigenvalue.py:205` | # Collision operator (simplified: nu_D * pitch-angle scattering applied to theta structure) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_geometry.py:121` | # Poloidal field: B_p = r / (q * R_s * \|J/r\|) — simplified |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_geometry.py:127` | # b . grad(theta) = B_p / (R_s * \|grad theta\|) ≈ 1 / (q * R_s) simplified |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_online_learner.py:138` | # Backprop (simplified, output layer only for stability) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_species.py:13` | simplified Sugama model (pitch-angle scattering + energy diffusion). |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_species.py:157` | Simplified Sugama model: |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_species.py:166` | # Simplified to order-of-magnitude for the linearised operator |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_species.py:179` | nu_E = nu_D # simplified: energy diffusion ~ pitch-angle for like-species |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/impurity_transport.py:170` | rhs[-1] = s.source_rate * dt / dr # Very simplified edge source mapping |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_scenario.py:35` | # CD parameters (simplified) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_scenario.py:245` | j_bs = np.zeros(self.nr) # simplified |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_scenario.py:255` | q_peak = sol_res.q_parallel_MW_m2 # simplified |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/lh_transition.py:48` | # Drive scales with pressure gradient, simplified here to scale with p |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/lh_transition.py:149` | # Highly simplified heuristic logic |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/locked_mode.py:32` | # Simplified linear correction |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/locked_mode.py:72` | # Very simplified proxy formula retaining the B_res^2 scaling |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/momentum_transport.py:110` | # Simplified criterion: omega_phi * tau_wall > 1% of something? |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neoclassical.py:208` | # Simplified for this module |
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
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/gk_tglf.py:204` | _logger.warning("TGLF binary not found, returning fallback") |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/jax_gs_solver.py:24` | NumPy fallback provided when JAX is unavailable. |
