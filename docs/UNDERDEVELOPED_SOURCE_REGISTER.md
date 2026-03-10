# Underdeveloped Register

- Generated at: `2026-03-10T23:45:12.344544+00:00`
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
| Domains affected | 2 |

## Marker Distribution

| Key | Count |
|---|---:|
| `MONOLITH` | 2 |

## Domain Distribution

| Key | Count |
|---|---:|
| `control` | 1 |
| `core_physics` | 1 |

## Top Priority Backlog (Top 2)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 102 | `control` | `MONOLITH` | `src/scpn_fusion/control/h_infinity_controller.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=545 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/stability_mhd.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=548 exceeds monolith threshold (500+). |

## Full Register (Top 2)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/h_infinity_controller.py:1` | module LOC=545 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/stability_mhd.py:1` | module LOC=548 exceeds monolith threshold (500+). |
