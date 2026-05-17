# Underdeveloped Register

- Generated at: `2026-05-17T20:14:06.241883+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 29 |
| P0 + P1 entries | 22 |
| Source-domain entries | 29 |
| Source-domain P0 + P1 entries | 22 |
| Docs-claims entries | 0 |
| Domains affected | 2 |

## Marker Distribution

| Key | Count |
|---|---:|
| `SIMPLIFIED` | 22 |
| `FALLBACK` | 7 |

## Domain Distribution

| Key | Count |
|---|---:|
| `core_physics` | 25 |
| `control` | 4 |

## Top Priority Backlog (Top 29)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
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
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/_free_boundary_supervisory_types.py:22` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | SUPERVISORY_ALERT_LEVEL_NAMES = ("nominal", "warning", "guarded", "fallback") |
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/_free_boundary_tracking_control.py:171` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | raise ValueError("fallback currents are not configured.") |
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/_free_boundary_tracking_shot.py:276` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | f"lag={max_abs_actuator_lag:.3e} \| fallback={fallback_active} \| " |
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_tracking.py:12` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | rejection can ramp the coil set toward explicit safe fallback currents instead |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/disruption_sequence.py:285` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | seed = 1e10 # Fallback |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/gk_tglf.py:203` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | _logger.warning("TGLF binary not found, returning fallback") |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/jax_gs_solver.py:23` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | NumPy fallback provided when JAX is unavailable. |

## Full Register (Top 29)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
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
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/_free_boundary_supervisory_types.py:22` | SUPERVISORY_ALERT_LEVEL_NAMES = ("nominal", "warning", "guarded", "fallback") |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/_free_boundary_tracking_control.py:171` | raise ValueError("fallback currents are not configured.") |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/_free_boundary_tracking_shot.py:276` | f"lag={max_abs_actuator_lag:.3e} \| fallback={fallback_active} \| " |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_tracking.py:12` | rejection can ramp the coil set toward explicit safe fallback currents instead |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/disruption_sequence.py:285` | seed = 1e10 # Fallback |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/gk_tglf.py:203` | _logger.warning("TGLF binary not found, returning fallback") |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/jax_gs_solver.py:23` | NumPy fallback provided when JAX is unavailable. |
