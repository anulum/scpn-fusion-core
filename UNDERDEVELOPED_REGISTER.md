# Underdeveloped Register

- Generated at: `2026-05-25T22:32:21.601128+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: production code + docs claims markers (tests/reports/html excluded)

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 5 |
| P0 + P1 entries | 2 |
| Source-domain entries | 3 |
| Source-domain P0 + P1 entries | 2 |
| Docs-claims entries | 2 |
| Domains affected | 2 |

## Marker Distribution

| Key | Count |
|---|---:|
| `FALLBACK` | 3 |
| `MONOLITH` | 2 |

## Domain Distribution

| Key | Count |
|---|---:|
| `core_physics` | 3 |
| `docs_claims` | 2 |

## Source-Centric Priority Backlog (Top 2)

_Filtered to implementation domains to reduce docs/claims noise during hardening triage._

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel_free_boundary.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=614 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/jax_gs_solver.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=510 exceeds monolith threshold (500+). |

## Top Priority Backlog (Top 5)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel_free_boundary.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=614 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/jax_gs_solver.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=510 exceeds monolith threshold (500+). |
| P2 | 76 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_multi_compat.py:201` | Core Physics WG | Measure fallback hit-rate and retire fallback from default lane. | "Fallback telemetry skipped for kernel=%s selected=%s: %s", |
| P3 | 47 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:148` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | \| Solov'ev manufactured-source FreeGS fallback \| PASS \| `artifacts/freegs_benchmark.json` on the GPU host \| |
| P3 | 47 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:156` | Docs WG | Measure fallback hit-rate and retire fallback from default lane. | They are not CI or harness failures and must not be hidden by fallback rows. |

## Full Register (Top 5)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/fusion_kernel_free_boundary.py:1` | module LOC=614 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/jax_gs_solver.py:1` | module LOC=510 exceeds monolith threshold (500+). |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_multi_compat.py:201` | "Fallback telemetry skipped for kernel=%s selected=%s: %s", |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:148` | \| Solov'ev manufactured-source FreeGS fallback \| PASS \| `artifacts/freegs_benchmark.json` on the GPU host \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:156` | They are not CI or harness failures and must not be hidden by fallback rows. |
