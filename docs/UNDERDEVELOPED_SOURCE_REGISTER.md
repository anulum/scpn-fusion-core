# Underdeveloped Register

- Generated at: `2026-03-03T01:38:25.207022+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 8 |
| P0 + P1 entries | 8 |
| Source-domain entries | 8 |
| Source-domain P0 + P1 entries | 8 |
| Docs-claims entries | 0 |
| Domains affected | 4 |

## Marker Distribution

| Key | Count |
|---|---:|
| `MONOLITH` | 6 |
| `FALLBACK_DENSITY` | 2 |

## Domain Distribution

| Key | Count |
|---|---:|
| `control` | 3 |
| `core_physics` | 2 |
| `diagnostics_io` | 2 |
| `compiler_runtime` | 1 |

## Top Priority Backlog (Top 8)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 100 | `control` | `MONOLITH` | `src/scpn_fusion/control/disruption_predictor.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=827 exceeds monolith threshold (700+). |
| P0 | 99 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=830 exceeds monolith threshold (700+). |
| P0 | 99 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver_runtime.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=839 exceeds monolith threshold (700+). |
| P0 | 98 | `compiler_runtime` | `MONOLITH` | `src/scpn_fusion/scpn/controller.py:1` | Runtime WG | Split module into focused subcomponents and lock interface contracts. | module LOC=707 exceeds monolith threshold (700+). |
| P0 | 97 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/imas_connector.py:1` | Diagnostics/IO WG | Split module into focused subcomponents and lock interface contracts. | module LOC=819 exceeds monolith threshold (700+). |
| P0 | 97 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/tokamak_archive.py:1` | Diagnostics/IO WG | Split module into focused subcomponents and lock interface contracts. | module LOC=727 exceeds monolith threshold (700+). |
| P1 | 94 | `control` | `FALLBACK_DENSITY` | `src/scpn_fusion/control/analytic_solver.py:1` | Control WG | Reduce fallback concentration and enforce strict-backend parity checks. | fallback mentions=8 across LOC=247; high fallback concentration in primary module. |
| P1 | 94 | `control` | `FALLBACK_DENSITY` | `src/scpn_fusion/control/disruption_predictor.py:1` | Control WG | Reduce fallback concentration and enforce strict-backend parity checks. | fallback mentions=10 across LOC=827; high fallback concentration in primary module. |

## Full Register (Top 8)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/disruption_predictor.py:1` | module LOC=827 exceeds monolith threshold (700+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | module LOC=830 exceeds monolith threshold (700+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver_runtime.py:1` | module LOC=839 exceeds monolith threshold (700+). |
| P0 | `compiler_runtime` | `MONOLITH` | `src/scpn_fusion/scpn/controller.py:1` | module LOC=707 exceeds monolith threshold (700+). |
| P0 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/imas_connector.py:1` | module LOC=819 exceeds monolith threshold (700+). |
| P0 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/tokamak_archive.py:1` | module LOC=727 exceeds monolith threshold (700+). |
| P1 | `control` | `FALLBACK_DENSITY` | `src/scpn_fusion/control/analytic_solver.py:1` | fallback mentions=8 across LOC=247; high fallback concentration in primary module. |
| P1 | `control` | `FALLBACK_DENSITY` | `src/scpn_fusion/control/disruption_predictor.py:1` | fallback mentions=10 across LOC=827; high fallback concentration in primary module. |
