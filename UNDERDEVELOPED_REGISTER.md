# Underdeveloped Register

- Generated at: `2026-03-11T02:38:27.352244+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: production code + docs claims markers (tests/reports/html excluded)

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 7 |
| P0 + P1 entries | 3 |
| Source-domain entries | 5 |
| Source-domain P0 + P1 entries | 3 |
| Docs-claims entries | 2 |
| Domains affected | 4 |

## Marker Distribution

| Key | Count |
|---|---:|
| `EXPERIMENTAL` | 3 |
| `FALLBACK` | 2 |
| `MONOLITH` | 2 |

## Domain Distribution

| Key | Count |
|---|---:|
| `validation` | 3 |
| `docs_claims` | 2 |
| `control` | 1 |
| `core_physics` | 1 |

## Source-Centric Priority Backlog (Top 3)

_Filtered to implementation domains to reduce docs/claims noise during hardening triage._

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 102 | `control` | `MONOLITH` | `src/scpn_fusion/control/h_infinity_controller.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=545 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/stability_mhd.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=548 exceeds monolith threshold (500+). |
| P0 | 96 | `validation` | `EXPERIMENTAL` | `tools/train_neural_equilibrium_gpu.py:714` | Validation WG | Gate behind explicit flag and define validation exit criteria. | print("\n ACCEPTANCE CRITERIA NOT MET — weights saved as experimental") |

## Top Priority Backlog (Top 7)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 102 | `control` | `MONOLITH` | `src/scpn_fusion/control/h_infinity_controller.py:1` | Control WG | Split module into focused subcomponents and lock interface contracts. | module LOC=545 exceeds monolith threshold (500+). |
| P0 | 101 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/stability_mhd.py:1` | Core Physics WG | Split module into focused subcomponents and lock interface contracts. | module LOC=548 exceeds monolith threshold (500+). |
| P0 | 96 | `validation` | `EXPERIMENTAL` | `tools/train_neural_equilibrium_gpu.py:714` | Validation WG | Gate behind explicit flag and define validation exit criteria. | print("\n ACCEPTANCE CRITERIA NOT MET — weights saved as experimental") |
| P2 | 73 | `validation` | `FALLBACK` | `tools/download_diiid_data.py:230` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | logger.debug("tokamak_archive fallback failed: %s", exc) |
| P2 | 73 | `validation` | `FALLBACK` | `tools/generate_fno_qlknn_spatial.py:143` | Validation WG | Measure fallback hit-rate and retire fallback from default lane. | print(" Will use critical-gradient fallback (less accurate).") |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md:47` | Docs WG | Gate behind explicit flag and define validation exit criteria. | --experimental \ |
| P3 | 52 | `docs_claims` | `EXPERIMENTAL` | `docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md:48` | Docs WG | Gate behind explicit flag and define validation exit criteria. | --experimental-ack I_UNDERSTAND_EXPERIMENTAL \ |

## Full Register (Top 7)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `control` | `MONOLITH` | `src/scpn_fusion/control/h_infinity_controller.py:1` | module LOC=545 exceeds monolith threshold (500+). |
| P0 | `core_physics` | `MONOLITH` | `src/scpn_fusion/core/stability_mhd.py:1` | module LOC=548 exceeds monolith threshold (500+). |
| P0 | `validation` | `EXPERIMENTAL` | `tools/train_neural_equilibrium_gpu.py:714` | print("\n ACCEPTANCE CRITERIA NOT MET — weights saved as experimental") |
| P2 | `validation` | `FALLBACK` | `tools/download_diiid_data.py:230` | logger.debug("tokamak_archive fallback failed: %s", exc) |
| P2 | `validation` | `FALLBACK` | `tools/generate_fno_qlknn_spatial.py:143` | print(" Will use critical-gradient fallback (less accurate).") |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md:47` | --experimental \ |
| P3 | `docs_claims` | `EXPERIMENTAL` | `docs/FNO_EXTERNAL_RETRAIN_RUNBOOK.md:48` | --experimental-ack I_UNDERSTAND_EXPERIMENTAL \ |
