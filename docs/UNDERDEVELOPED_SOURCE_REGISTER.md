# Underdeveloped Register

- Generated at: `2026-03-04T10:50:39.221651+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 5 |
| P0 + P1 entries | 3 |
| Source-domain entries | 3 |
| Source-domain P0 + P1 entries | 3 |
| Docs-claims entries | 0 |
| Domains affected | 3 |

## Marker Distribution

| Key | Count |
|---|---:|
| `FALLBACK` | 2 |
| `MONOLITH` | 2 |
| `FALLBACK_DENSITY` | 1 |

## Domain Distribution

| Key | Count |
|---|---:|
| `core_physics` | 2 |
| `other` | 2 |
| `control` | 1 |

## Top Priority Backlog (Top 5)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fno_training.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=572 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_equilibrium.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=595 exceeds monolith threshold (500+). |
| P0 | 96 | `control` | `FALLBACK_DENSITY` | `src/scpn_fusion/control/disruption_predictor.py:1` | Control WG | Reduce fallback concentration and enforce strict-backend parity checks. | fallback risk signals=6 across LOC=437; high fallback concentration in runtime code paths. |
| P3 | 65 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:32` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | raise ValueError("fallback domain must be non-empty") |
| P3 | 65 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:114` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | "Fallback budget exceeded: " |

## Full Register (Top 5)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fno_training.py:1` | module LOC=572 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/neural_equilibrium.py:1` | module LOC=595 exceeds monolith threshold (500+). |
| P0 | `control` | `FALLBACK_DENSITY` | `src/scpn_fusion/control/disruption_predictor.py:1` | fallback risk signals=6 across LOC=437; high fallback concentration in runtime code paths. |
| P3 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:32` | raise ValueError("fallback domain must be non-empty") |
| P3 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:114` | "Fallback budget exceeded: " |
