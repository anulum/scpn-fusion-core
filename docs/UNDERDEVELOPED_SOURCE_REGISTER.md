# Underdeveloped Register

- Generated at: `2026-03-02T15:27:17.430532+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 3 |
| P0 + P1 entries | 3 |
| Source-domain entries | 3 |
| Source-domain P0 + P1 entries | 3 |
| Docs-claims entries | 0 |
| Domains affected | 2 |

## Marker Distribution

| Key | Count |
|---|---:|
| `MONOLITH` | 3 |

## Domain Distribution

| Key | Count |
|---|---:|
| `core_physics` | 2 |
| `diagnostics_io` | 1 |

## Top Priority Backlog (Top 3)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 107 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=1435 exceeds monolith threshold (900+). |
| P0 | 107 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=1738 exceeds monolith threshold (900+). |
| P0 | 97 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/imas_connector.py:1` | Diagnostics/IO WG | Split module into focused subcomponents and lock interface contracts. | module LOC=990 exceeds monolith threshold (900+). |

## Full Register (Top 3)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel.py:1` | module LOC=1435 exceeds monolith threshold (900+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | module LOC=1738 exceeds monolith threshold (900+). |
| P0 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/imas_connector.py:1` | module LOC=990 exceeds monolith threshold (900+). |
