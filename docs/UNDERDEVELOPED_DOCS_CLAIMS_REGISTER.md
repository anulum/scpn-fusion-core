# Underdeveloped Register

- Generated at: `2026-05-27T17:56:11.357749+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: docs-claims-only markers

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 2 |
| P0 + P1 entries | 0 |
| Source-domain entries | 0 |
| Source-domain P0 + P1 entries | 0 |
| Docs-claims entries | 2 |
| Domains affected | 1 |

## Marker Distribution

| Key | Count |
|---|---:|
| `FALLBACK` | 2 |

## Domain Distribution

| Key | Count |
|---|---:|
| `docs_claims` | 2 |

## Top Priority Backlog (Top 2)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P3 | 47 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:148` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | \| Solov'ev manufactured-source FreeGS fallback \| PASS \| `artifacts/freegs_benchmark.json` on the GPU host \| |
| P3 | 47 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:156` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | They are not CI or harness failures and must not be hidden by fallback rows. |

## Full Register (Top 2)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:148` | \| Solov'ev manufactured-source FreeGS fallback \| PASS \| `artifacts/freegs_benchmark.json` on the GPU host \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:156` | They are not CI or harness failures and must not be hidden by fallback rows. |
