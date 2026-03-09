# Underdeveloped Register

- Generated at: `2026-03-09T13:20:28.401197+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: production code + docs claims markers (tests/reports/html excluded)

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 4 |
| P0 + P1 entries | 0 |
| Source-domain entries | 2 |
| Source-domain P0 + P1 entries | 0 |
| Docs-claims entries | 2 |
| Domains affected | 2 |

## Marker Distribution

| Key | Count |
|---|---:|
| `EXPERIMENTAL` | 2 |
| `FALLBACK` | 2 |

## Domain Distribution

| Key | Count |
|---|---:|
| `docs_claims` | 2 |
| `validation` | 2 |

## Source-Centric Priority Backlog (Top 0)

_Filtered to implementation domains to reduce docs/claims noise during hardening triage._

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|

## Top Priority Backlog (Top 4)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P2 | 73 | `validation` | `FALLBACK` | `tools/download_diiid_data.py:230` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | logger.debug("tokamak_archive fallback failed: %s", exc) |
| P2 | 73 | `validation` | `FALLBACK` | `tools/generate_fno_qlknn_spatial.py:143` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | print(" Will use critical-gradient fallback (less accurate).") |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md:47` | Docs WG | Gate behind explicit flag and define validation exit criteria. | --experimental \ |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md:48` | Docs WG | Gate behind explicit flag and define validation exit criteria. | --experimental-ack I_UNDERSTAND_EXPERIMENTAL \ |

## Full Register (Top 4)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P2 | `validation` | `FALLBACK` | `tools/download_diiid_data.py:230` | logger.debug("tokamak_archive fallback failed: %s", exc) |
| P2 | `validation` | `FALLBACK` | `tools/generate_fno_qlknn_spatial.py:143` | print(" Will use critical-gradient fallback (less accurate).") |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md:47` | --experimental \ |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md:48` | --experimental-ack I_UNDERSTAND_EXPERIMENTAL \ |
