# Underdeveloped Register

- Generated at: `2026-03-02T15:49:40.351323+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 2 |
| P0 + P1 entries | 2 |
| Source-domain entries | 2 |
| Source-domain P0 + P1 entries | 2 |
| Docs-claims entries | 0 |
| Domains affected | 1 |

## Marker Distribution

| Key | Count |
|---|---:|
| `MONOLITH` | 2 |

## Domain Distribution

| Key | Count |
|---|---:|
| `core_physics` | 2 |

## Top Priority Backlog (Top 2)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 107 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=1435 exceeds monolith threshold (900+). |
| P0 | 107 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=1738 exceeds monolith threshold (900+). |

## Full Register (Top 2)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel.py:1` | module LOC=1435 exceeds monolith threshold (900+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | module LOC=1738 exceeds monolith threshold (900+). |
