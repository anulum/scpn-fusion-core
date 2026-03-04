# Underdeveloped Register

- Generated at: `2026-03-03T23:57:51.463292+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 15 |
| P0 + P1 entries | 13 |
| Source-domain entries | 13 |
| Source-domain P0 + P1 entries | 13 |
| Docs-claims entries | 0 |
| Domains affected | 4 |

## Marker Distribution

| Key | Count |
|---|---:|
| `MONOLITH` | 12 |
| `FALLBACK` | 2 |
| `FALLBACK_DENSITY` | 1 |

## Domain Distribution

| Key | Count |
|---|---:|
| `core_physics` | 7 |
| `control` | 4 |
| `compiler_runtime` | 2 |
| `other` | 2 |

## Top Priority Backlog (Top 15)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 102 | `control` | `MONOLITH` | `src/scpn_fusion/control/disruption_contracts.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=649 exceeds monolith threshold (500+). |
| P0 | 102 | `control` | `MONOLITH` | `src/scpn_fusion/control/halo_re_physics.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=651 exceeds monolith threshold (500+). |
| P0 | 102 | `control` | `MONOLITH` | `src/scpn_fusion/control/hil_harness.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=547 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fno_training.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=572 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=590 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel_newton_solver.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=632 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver_runtime.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=518 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_equilibrium.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=595 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=509 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=517 exceeds monolith threshold (500+). |
| P0 | 100 | `compiler_runtime` | `MONOLITH` | `src/scpn_fusion/scpn/artifact.py:1` | Runtime WG | Split module into focused subcomponents and lock interface contracts. | module LOC=583 exceeds monolith threshold (500+). |
| P0 | 100 | `compiler_runtime` | `MONOLITH` | `src/scpn_fusion/scpn/controller.py:1` | Runtime WG | Split module into focused subcomponents and lock interface contracts. | module LOC=599 exceeds monolith threshold (500+). |
| P0 | 96 | `control` | `FALLBACK_DENSITY` | `src/scpn_fusion/control/disruption_predictor.py:1` | Control WG | Reduce fallback concentration and enforce strict-backend parity checks. | fallback risk signals=6 across LOC=437; high fallback concentration in runtime code paths. |
| P3 | 65 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:32` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | raise ValueError("fallback domain must be non-empty") |
| P3 | 65 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:114` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | "Fallback budget exceeded: " |

## Full Register (Top 15)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/disruption_contracts.py:1` | module LOC=649 exceeds monolith threshold (500+). |
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/halo_re_physics.py:1` | module LOC=651 exceeds monolith threshold (500+). |
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/hil_harness.py:1` | module LOC=547 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fno_training.py:1` | module LOC=572 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel.py:1` | module LOC=590 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel_newton_solver.py:1` | module LOC=632 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver_runtime.py:1` | module LOC=518 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_equilibrium.py:1` | module LOC=595 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | module LOC=509 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | module LOC=517 exceeds monolith threshold (500+). |
| P0 | `compiler_runtime` | `MONOLITH` | `src/scpn_fusion/scpn/artifact.py:1` | module LOC=583 exceeds monolith threshold (500+). |
| P0 | `compiler_runtime` | `MONOLITH` | `src/scpn_fusion/scpn/controller.py:1` | module LOC=599 exceeds monolith threshold (500+). |
| P0 | `control` | `FALLBACK_DENSITY` | `src/scpn_fusion/control/disruption_predictor.py:1` | fallback risk signals=6 across LOC=437; high fallback concentration in runtime code paths. |
| P3 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:32` | raise ValueError("fallback domain must be non-empty") |
| P3 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:114` | "Fallback budget exceeded: " |
