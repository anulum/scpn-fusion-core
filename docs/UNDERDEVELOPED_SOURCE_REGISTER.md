# Underdeveloped Register

- Generated at: `2026-03-03T05:13:38.888962+00:00`
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
| `FALLBACK_DENSITY` | 2 |
| `MONOLITH` | 1 |

## Domain Distribution

| Key | Count |
|---|---:|
| `control` | 2 |
| `other` | 2 |
| `diagnostics_io` | 1 |

## Top Priority Backlog (Top 5)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 97 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/imas_connector.py:1` | Diagnostics/IO WG | Split module into focused subcomponents and lock interface contracts. | module LOC=819 exceeds monolith threshold (700+). |
| P1 | 94 | `control` | `FALLBACK_DENSITY` | `src/scpn_fusion/control/analytic_solver.py:1` | Control WG | Reduce fallback concentration and enforce strict-backend parity checks. | fallback mentions=8 across LOC=247; high fallback concentration in primary module. |
| P1 | 94 | `control` | `FALLBACK_DENSITY` | `src/scpn_fusion/control/disruption_predictor.py:1` | Control WG | Reduce fallback concentration and enforce strict-backend parity checks. | fallback mentions=10 across LOC=467; high fallback concentration in primary module. |
| P3 | 65 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:32` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | raise ValueError("fallback domain must be non-empty") |
| P3 | 65 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:114` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | "Fallback budget exceeded: " |

## Full Register (Top 5)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `diagnostics_io` | `MONOLITH` | `src/scpn_fusion/io/imas_connector.py:1` | module LOC=819 exceeds monolith threshold (700+). |
| P1 | `control` | `FALLBACK_DENSITY` | `src/scpn_fusion/control/analytic_solver.py:1` | fallback mentions=8 across LOC=247; high fallback concentration in primary module. |
| P1 | `control` | `FALLBACK_DENSITY` | `src/scpn_fusion/control/disruption_predictor.py:1` | fallback mentions=10 across LOC=467; high fallback concentration in primary module. |
| P3 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:32` | raise ValueError("fallback domain must be non-empty") |
| P3 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:114` | "Fallback budget exceeded: " |
