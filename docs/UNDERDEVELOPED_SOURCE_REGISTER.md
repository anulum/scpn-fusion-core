# Underdeveloped Register

- Generated at: `2026-03-22T21:27:12.775572+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 54 |
| P0 + P1 entries | 47 |
| Source-domain entries | 54 |
| Source-domain P0 + P1 entries | 47 |
| Docs-claims entries | 0 |
| Domains affected | 4 |

## Marker Distribution

| Key | Count |
|---|---:|
| `SIMPLIFIED` | 37 |
| `MONOLITH` | 8 |
| `FALLBACK` | 7 |
| `DEPRECATED` | 1 |
| `TEST_GAP` | 1 |

## Domain Distribution

| Key | Count |
|---|---:|
| `core_physics` | 40 |
| `control` | 12 |
| `compiler_runtime` | 1 |
| `diagnostics_io` | 1 |

## Top Priority Backlog (Top 54)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 110 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=1419 exceeds monolith threshold (500+). |
| P0 | 110 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_tracking.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=1566 exceeds monolith threshold (500+). |
| P0 | 109 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver_model.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=884 exceeds monolith threshold (500+). |
| P0 | 109 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=849 exceeds monolith threshold (500+). |
| P0 | 107 | `control` | `DEPRECATED` | `src/scpn_fusion/control/nengo_snn_wrapper.py:374` | Control WG | Replace default path or remove lane before next major release. | "NengoSNNControllerStub is deprecated. " |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/gk_nonlinear.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=581 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=526 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=669 exceeds monolith threshold (500+). |
| P0 | 99 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/tokamak_archive.py:1` | Diagnostics/IO WG | Split module into focused subcomponents and lock interface contracts. | module LOC=501 exceeds monolith threshold (500+). |
| P0 | 98 | `compiler_runtime` | `TEST_GAP` | `src/scpn_fusion/hpc/hpc_bridge.py:1` | Runtime WG | Add direct module tests and eliminate allowlist-only linkage. | large source module without direct test import/stem linkage. |
| P1 | 86 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/burn_controller.py:139` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified: rely on integral to find it |
| P1 | 86 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/controller_tuning.py:62` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | action = kp * error # Simplified |
| P1 | 86 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/density_controller.py:216` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified prediction: static + noise |
| P1 | 86 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/mu_synthesis.py:149` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified: u = -K * x |
| P1 | 86 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/realtime_efit.py:98` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | """Simplified real-time equilibrium reconstruction (EFIT).""" |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/alfven_eigenmodes.py:254` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | omega_BAE = 0.1 * v_A / R0 # Highly simplified |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/disruption_sequence.py:114` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Extremely simplified mapping: drops to few eV via line radiation of impurities. |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/disruption_sequence.py:211` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | """F_z [MN] = I_halo * B_tor * 2pi R0 * TPF (simplified)""" |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:28` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | Critical edge current density limit (simplified scaling). |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:56` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified PB boundary: j_norm^2 + a_norm^2 > 1 |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:148` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Very simplified representation of overlap across outer rational surfaces |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_eigenvalue.py:204` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Collision operator (simplified: nu_D * pitch-angle scattering applied to theta structure) |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_geometry.py:120` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Poloidal field: B_p = r / (q * R_s * \|J/r\|) — simplified |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_geometry.py:126` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # b . grad(theta) = B_p / (R_s * \|grad theta\|) ≈ 1 / (q * R_s) simplified |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_nonlinear.py:19` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | - Simplified Sugama collision operator |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_online_learner.py:137` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Backprop (simplified, output layer only for stability) |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_species.py:12` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | simplified Sugama model (pitch-angle scattering + energy diffusion). |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_species.py:156` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | Simplified Sugama model: |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_species.py:165` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified to order-of-magnitude for the linearised operator |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_species.py:178` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | nu_E = nu_D # simplified: energy diffusion ~ pitch-angle for like-species |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/impurity_transport.py:173` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | rhs[-1] = s.source_rate * dt / dr # Very simplified edge source mapping |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_scenario.py:38` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # CD parameters (simplified) |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_scenario.py:248` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | j_bs = np.zeros(self.nr) # simplified |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_scenario.py:258` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | q_peak = sol_res.q_parallel_MW_m2 # simplified |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/lh_transition.py:51` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Drive scales with pressure gradient, simplified here to scale with p |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/lh_transition.py:152` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Highly simplified heuristic logic |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/locked_mode.py:35` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified linear correction |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/locked_mode.py:75` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Very simplified proxy formula retaining the B_res^2 scaling |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/momentum_transport.py:113` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified criterion: omega_phi * tau_wall > 1% of something? |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neoclassical.py:213` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified for this module |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neural_turbulence.py:180` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # highly simplified nu_star |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neural_turbulence.py:227` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | Compute flux targets from simplified analytical quasilinear model. |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/plasma_wall_interaction.py:31` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified threshold energy |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/plasma_wall_interaction.py:47` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Using a simplified parameterized curve from Eckstein |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/plasma_wall_interaction.py:181` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Highly simplified fatigue life mapping |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/stellarator_geometry.py:154` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | Simplified analytic proxy: epsilon_eff ~ epsilon_h^(3/2) / sqrt(N_fp). |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/vmec_lite.py:52` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | Highly simplified spectral 3D equilibrium solver mimicking VMEC principles. |
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:34` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | SUPERVISORY_ALERT_LEVEL_NAMES = ("nominal", "warning", "guarded", "fallback") |
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_tracking.py:12` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | rejection can ramp the coil set toward explicit safe fallback currents instead |
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_tracking.py:997` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | raise ValueError("fallback currents are not configured.") |
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_tracking.py:1429` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | f"lag={max_abs_actuator_lag:.3e} \| fallback={fallback_active} \| " |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/disruption_sequence.py:269` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | seed = 1e10 # Fallback |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/gk_tglf.py:203` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | _logger.warning("TGLF binary not found, returning fallback") |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/jax_gs_solver.py:23` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | NumPy fallback provided when JAX is unavailable. |

## Full Register (Top 54)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:1` | module LOC=1419 exceeds monolith threshold (500+). |
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_tracking.py:1` | module LOC=1566 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver_model.py:1` | module LOC=884 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | module LOC=849 exceeds monolith threshold (500+). |
| P0 | `control` | `DEPRECATED` | `src/scpn_fusion/control/nengo_snn_wrapper.py:374` | "NengoSNNControllerStub is deprecated. " |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/gk_nonlinear.py:1` | module LOC=581 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | module LOC=526 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | module LOC=669 exceeds monolith threshold (500+). |
| P0 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/tokamak_archive.py:1` | module LOC=501 exceeds monolith threshold (500+). |
| P0 | `compiler_runtime` | `TEST_GAP` | `src/scpn_fusion/hpc/hpc_bridge.py:1` | large source module without direct test import/stem linkage. |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/burn_controller.py:139` | # Simplified: rely on integral to find it |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/controller_tuning.py:62` | action = kp * error # Simplified |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/density_controller.py:216` | # Simplified prediction: static + noise |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/mu_synthesis.py:149` | # Simplified: u = -K * x |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/realtime_efit.py:98` | """Simplified real-time equilibrium reconstruction (EFIT).""" |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/alfven_eigenmodes.py:254` | omega_BAE = 0.1 * v_A / R0 # Highly simplified |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/disruption_sequence.py:114` | # Extremely simplified mapping: drops to few eV via line radiation of impurities. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/disruption_sequence.py:211` | """F_z [MN] = I_halo * B_tor * 2pi R0 * TPF (simplified)""" |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:28` | Critical edge current density limit (simplified scaling). |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:56` | # Simplified PB boundary: j_norm^2 + a_norm^2 > 1 |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/elm_model.py:148` | # Very simplified representation of overlap across outer rational surfaces |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_eigenvalue.py:204` | # Collision operator (simplified: nu_D * pitch-angle scattering applied to theta structure) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_geometry.py:120` | # Poloidal field: B_p = r / (q * R_s * \|J/r\|) — simplified |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_geometry.py:126` | # b . grad(theta) = B_p / (R_s * \|grad theta\|) ≈ 1 / (q * R_s) simplified |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_nonlinear.py:19` | - Simplified Sugama collision operator |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_online_learner.py:137` | # Backprop (simplified, output layer only for stability) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_species.py:12` | simplified Sugama model (pitch-angle scattering + energy diffusion). |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_species.py:156` | Simplified Sugama model: |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_species.py:165` | # Simplified to order-of-magnitude for the linearised operator |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/gk_species.py:178` | nu_E = nu_D # simplified: energy diffusion ~ pitch-angle for like-species |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/impurity_transport.py:173` | rhs[-1] = s.source_rate * dt / dr # Very simplified edge source mapping |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_scenario.py:38` | # CD parameters (simplified) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_scenario.py:248` | j_bs = np.zeros(self.nr) # simplified |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_scenario.py:258` | q_peak = sol_res.q_parallel_MW_m2 # simplified |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/lh_transition.py:51` | # Drive scales with pressure gradient, simplified here to scale with p |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/lh_transition.py:152` | # Highly simplified heuristic logic |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/locked_mode.py:35` | # Simplified linear correction |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/locked_mode.py:75` | # Very simplified proxy formula retaining the B_res^2 scaling |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/momentum_transport.py:113` | # Simplified criterion: omega_phi * tau_wall > 1% of something? |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neoclassical.py:213` | # Simplified for this module |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neural_turbulence.py:180` | # highly simplified nu_star |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neural_turbulence.py:227` | Compute flux targets from simplified analytical quasilinear model. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/plasma_wall_interaction.py:31` | # Simplified threshold energy |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/plasma_wall_interaction.py:47` | # Using a simplified parameterized curve from Eckstein |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/plasma_wall_interaction.py:181` | # Highly simplified fatigue life mapping |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/stellarator_geometry.py:154` | Simplified analytic proxy: epsilon_eff ~ epsilon_h^(3/2) / sqrt(N_fp). |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/vmec_lite.py:52` | Highly simplified spectral 3D equilibrium solver mimicking VMEC principles. |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:34` | SUPERVISORY_ALERT_LEVEL_NAMES = ("nominal", "warning", "guarded", "fallback") |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_tracking.py:12` | rejection can ramp the coil set toward explicit safe fallback currents instead |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_tracking.py:997` | raise ValueError("fallback currents are not configured.") |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_tracking.py:1429` | f"lag={max_abs_actuator_lag:.3e} \| fallback={fallback_active} \| " |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/disruption_sequence.py:269` | seed = 1e10 # Fallback |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/gk_tglf.py:203` | _logger.warning("TGLF binary not found, returning fallback") |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/jax_gs_solver.py:23` | NumPy fallback provided when JAX is unavailable. |
