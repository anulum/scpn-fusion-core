# Underdeveloped Register

- Generated at: `2026-03-02T17:12:46.769385+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 1 |
| P0 + P1 entries | 1 |
| Source-domain entries | 1 |
| Source-domain P0 + P1 entries | 1 |
| Docs-claims entries | 0 |
| Domains affected | 1 |

## Marker Distribution

| Key | Count |
|---|---:|
| `MONOLITH` | 1 |

## Domain Distribution

| Key | Count |
|---|---:|
| `core_physics` | 1 |

## Top Priority Backlog (Top 1)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 107 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=1738 exceeds monolith threshold (900+). |

## Full Register (Top 1)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | module LOC=1738 exceeds monolith threshold (900+). |
