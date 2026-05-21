# Underdeveloped Register

- Generated at: `2026-05-21T19:44:15.902734+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

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

## Top Priority Backlog (Top 1)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_multi_compat.py:201` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | "Fallback telemetry skipped for kernel=%s selected=%s: %s", |

## Full Register (Top 1)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_multi_compat.py:201` | "Fallback telemetry skipped for kernel=%s selected=%s: %s", |
