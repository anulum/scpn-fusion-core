# Underdeveloped Register

- Generated at: `2026-05-27T17:56:11.357689+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 16 |
| P0 + P1 entries | 8 |
| Source-domain entries | 15 |
| Source-domain P0 + P1 entries | 8 |
| Docs-claims entries | 0 |
| Domains affected | 4 |

## Marker Distribution

| Key | Count |
|---|---:|
| `FALLBACK` | 8 |
| `MONOLITH` | 5 |
| `SIMPLIFIED` | 2 |
| `NOT_VALIDATED` | 1 |

## Domain Distribution

| Key | Count |
|---|---:|
| `core_physics` | 13 |
| `compiler_runtime` | 1 |
| `control` | 1 |
| `other` | 1 |

## Top Priority Backlog (Top 16)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel_free_boundary.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=614 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_scenario.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=556 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/jax_gs_solver.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=514 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_equilibrium.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=507 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/pretrained_surrogates.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=584 exceeds monolith threshold (500+). |
| P0 | 96 | `compiler_runtime` | `NOT_VALIDATED` | `src/scpn_fusion/scpn/structure.py:379` | Runtime WG | Add real-data validation campaign and publish error bars. | """Return the latest validation report, or ``None`` if not validated yet.""" |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/orbit_following.py:234` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | """Simplified alpha slowing-down analytic helpers.""" |
| P1 | 85 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/rf_heating.py:8` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | """RF heating workflows with simplified ICRH/ECRH ray-tracing examples.""" |
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/runaway_electron_model.py:218` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | The integration advances a coupled Dreicer + avalanche + fallback-loss |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_multi_compat.py:201` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | "Fallback telemetry skipped for kernel=%s selected=%s: %s", |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:11` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | ``IntegratedTransportSolver`` interface. It also exposes conservative fallback |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver_model.py:31` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | surrogate vs. deterministic fallback), and updates auxiliary transport state |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver_runtime_physics.py:41` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | adaptation with explicit fallback metadata for downstream observability. |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver_runtime_utils.py:12` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | (tridiagonal solve, diffusion assembly, fallback sanitisation) testable in |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/quasi_3d_contracts.py:218` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | Asymmetry fallback coefficient used if Hall-MHD backend is unavailable. |
| P3 | 65 | `other` | `FALLBACK` | `src/scpn_fusion/ui/app.py:29` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | except ImportError: # pragma: no cover - platform fallback |

## Full Register (Top 16)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel_free_boundary.py:1` | module LOC=614 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_scenario.py:1` | module LOC=556 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/jax_gs_solver.py:1` | module LOC=514 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_equilibrium.py:1` | module LOC=507 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/pretrained_surrogates.py:1` | module LOC=584 exceeds monolith threshold (500+). |
| P0 | `compiler_runtime` | `NOT_VALIDATED` | `src/scpn_fusion/scpn/structure.py:379` | """Return the latest validation report, or ``None`` if not validated yet.""" |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/orbit_following.py:234` | """Simplified alpha slowing-down analytic helpers.""" |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/rf_heating.py:8` | """RF heating workflows with simplified ICRH/ECRH ray-tracing examples.""" |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/runaway_electron_model.py:218` | The integration advances a coupled Dreicer + avalanche + fallback-loss |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_multi_compat.py:201` | "Fallback telemetry skipped for kernel=%s selected=%s: %s", |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:11` | ``IntegratedTransportSolver`` interface. It also exposes conservative fallback |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver_model.py:31` | surrogate vs. deterministic fallback), and updates auxiliary transport state |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver_runtime_physics.py:41` | adaptation with explicit fallback metadata for downstream observability. |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver_runtime_utils.py:12` | (tridiagonal solve, diffusion assembly, fallback sanitisation) testable in |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/quasi_3d_contracts.py:218` | Asymmetry fallback coefficient used if Hall-MHD backend is unavailable. |
| P3 | `other` | `FALLBACK` | `src/scpn_fusion/ui/app.py:29` | except ImportError: # pragma: no cover - platform fallback |
