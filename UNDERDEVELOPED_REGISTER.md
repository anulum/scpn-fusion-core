# Underdeveloped Register

- Generated at: `2026-05-21T19:43:46.889057+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: production code + docs claims markers (tests/reports/html excluded)

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 1 |
| P0 + P1 entries | 0 |
| Source-domain entries | 1 |
| Source-domain P0 + P1 entries | 0 |
| Docs-claims entries | 0 |
| Domains affected | 1 |

## Marker Distribution

| Key | Count |
|---|---:|
| `FALLBACK` | 1 |

## Domain Distribution

| Key | Count |
|---|---:|
| `core_physics` | 1 |

## Source-Centric Priority Backlog (Top 0)

_Filtered to implementation domains to reduce docs/claims noise during hardening triage._

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|

## Top Priority Backlog (Top 1)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_multi_compat.py:201` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | "Fallback telemetry skipped for kernel=%s selected=%s: %s", |

## Full Register (Top 1)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_multi_compat.py:201` | "Fallback telemetry skipped for kernel=%s selected=%s: %s", |
