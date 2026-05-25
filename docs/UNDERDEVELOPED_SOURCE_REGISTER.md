# Underdeveloped Register

- Generated at: `2026-05-25T22:36:35.303057+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 3 |
| P0 + P1 entries | 2 |
| Source-domain entries | 3 |
| Source-domain P0 + P1 entries | 2 |
| Docs-claims entries | 0 |
| Domains affected | 1 |

## Marker Distribution

| Key | Count |
|---|---:|
| `MONOLITH` | 2 |
| `FALLBACK` | 1 |

## Domain Distribution

| Key | Count |
|---|---:|
| `core_physics` | 3 |

## Top Priority Backlog (Top 3)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel_free_boundary.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=614 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/jax_gs_solver.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=510 exceeds monolith threshold (500+). |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_multi_compat.py:201` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | "Fallback telemetry skipped for kernel=%s selected=%s: %s", |

## Full Register (Top 3)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel_free_boundary.py:1` | module LOC=614 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/jax_gs_solver.py:1` | module LOC=510 exceeds monolith threshold (500+). |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_multi_compat.py:201` | "Fallback telemetry skipped for kernel=%s selected=%s: %s", |
