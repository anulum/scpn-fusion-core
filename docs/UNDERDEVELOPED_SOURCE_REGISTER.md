# Underdeveloped Register

- Generated at: `2026-03-05T19:27:50.541831+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: source-only (`src/scpn_fusion/**`) markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 2 |
| P0 + P1 entries | 0 |
| Source-domain entries | 0 |
| Source-domain P0 + P1 entries | 0 |
| Docs-claims entries | 0 |
| Domains affected | 1 |

## Marker Distribution

| Key | Count |
|---|---:|
| `FALLBACK` | 2 |

## Domain Distribution

| Key | Count |
|---|---:|
| `other` | 2 |

## Top Priority Backlog (Top 2)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P3 | 65 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:32` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | raise ValueError("fallback domain must be non-empty") |
| P3 | 65 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:114` | Architecture WG | Measure fallback hit-rate and retire fallback from default lane. | "Fallback budget exceeded: " |

## Full Register (Top 2)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P3 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:32` | raise ValueError("fallback domain must be non-empty") |
| P3 | `other` | `FALLBACK` | `src/scpn_fusion/fallback_telemetry.py:114` | "Fallback budget exceeded: " |
