# Underdeveloped Register

- Generated at: `2026-03-14T20:19:35.336727+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 33 |
| P0 + P1 entries | 28 |
| Source-domain entries | 33 |
| Source-domain P0 + P1 entries | 28 |
| Docs-claims entries | 0 |
| Domains affected | 3 |

## Marker Distribution

| Key | Count |
|---|---:|
| `SIMPLIFIED` | 21 |
| `MONOLITH` | 7 |
| `FALLBACK` | 5 |

## Domain Distribution

| Key | Count |
|---|---:|
| `core_physics` | 23 |
| `control` | 9 |
| `diagnostics_io` | 1 |

## Top Priority Backlog (Top 33)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 110 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=1419 exceeds monolith threshold (500+). |
| P0 | 110 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_tracking.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=1566 exceeds monolith threshold (500+). |
| P0 | 109 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver_model.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=884 exceeds monolith threshold (500+). |
| P0 | 109 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=849 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=526 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=669 exceeds monolith threshold (500+). |
| P0 | 99 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/tokamak_archive.py:1` | Diagnostics/IO WG | Split module into focused subcomponents and lock interface contracts. | module LOC=501 exceeds monolith threshold (500+). |
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

## Full Register (Top 33)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:1` | module LOC=1419 exceeds monolith threshold (500+). |
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_tracking.py:1` | module LOC=1566 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver_model.py:1` | module LOC=884 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | module LOC=849 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | module LOC=526 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | module LOC=669 exceeds monolith threshold (500+). |
| P0 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/tokamak_archive.py:1` | module LOC=501 exceeds monolith threshold (500+). |
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
