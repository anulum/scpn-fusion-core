# Underdeveloped Register

- Generated at: `2026-03-13T16:38:45.170317+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 8 |
| P0 + P1 entries | 7 |
| Source-domain entries | 8 |
| Source-domain P0 + P1 entries | 7 |
| Docs-claims entries | 0 |
| Domains affected | 3 |

## Marker Distribution

| Key | Count |
|---|---:|
| `MONOLITH` | 6 |
| `FALLBACK` | 1 |
| `SIMPLIFIED` | 1 |

## Domain Distribution

| Key | Count |
|---|---:|
| `core_physics` | 4 |
| `control` | 3 |
| `diagnostics_io` | 1 |

## Top Priority Backlog (Top 8)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 110 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=1419 exceeds monolith threshold (500+). |
| P0 | 109 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver_model.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=884 exceeds monolith threshold (500+). |
| P0 | 109 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=849 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=526 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=669 exceeds monolith threshold (500+). |
| P0 | 99 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/tokamak_archive.py:1` | Diagnostics/IO WG | Split module into focused subcomponents and lock interface contracts. | module LOC=501 exceeds monolith threshold (500+). |
| P1 | 86 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/realtime_efit.py:95` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | """Simplified real-time equilibrium reconstruction (EFIT).""" |
| P2 | 77 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:35` | Control WG | Measure fallback hit-rate and retire fallback from default lane. | SUPERVISORY_ALERT_LEVEL_NAMES = ("nominal", "warning", "guarded", "fallback") |

## Full Register (Top 8)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:1` | module LOC=1419 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver_model.py:1` | module LOC=884 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_transport.py:1` | module LOC=849 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/integrated_transport_solver.py:1` | module LOC=526 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/tglf_interface.py:1` | module LOC=669 exceeds monolith threshold (500+). |
| P0 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/tokamak_archive.py:1` | module LOC=501 exceeds monolith threshold (500+). |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/realtime_efit.py:95` | """Simplified real-time equilibrium reconstruction (EFIT).""" |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/free_boundary_supervisory_control.py:35` | SUPERVISORY_ALERT_LEVEL_NAMES = ("nominal", "warning", "guarded", "fallback") |
