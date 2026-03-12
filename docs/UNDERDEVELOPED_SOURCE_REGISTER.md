# Underdeveloped Register

- Generated at: `2026-03-11T23:44:51.781006+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 5 |
| P0 + P1 entries | 4 |
| Source-domain entries | 5 |
| Source-domain P0 + P1 entries | 4 |
| Docs-claims entries | 0 |
| Domains affected | 2 |

## Marker Distribution

| Key | Count |
|---|---:|
| `MONOLITH` | 4 |
| `FALLBACK` | 1 |

## Domain Distribution

| Key | Count |
|---|---:|
| `core_physics` | 3 |
| `control` | 2 |

## Top Priority Backlog (Top 5)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 110 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=1367 exceeds monolith threshold (500+). |
| P0 | 109 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver_model.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=868 exceeds monolith threshold (500+). |
| P0 | 109 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=825 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=667 exceeds monolith threshold (500+). |
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:35` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | SUPERVISORY_ALERT_LEVEL_NAMES = ("nominal", "warning", "guarded", "fallback") |

## Full Register (Top 5)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:1` | module LOC=1367 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver_model.py:1` | module LOC=868 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | module LOC=825 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | module LOC=667 exceeds monolith threshold (500+). |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:35` | SUPERVISORY_ALERT_LEVEL_NAMES = ("nominal", "warning", "guarded", "fallback") |
