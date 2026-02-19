# v3.6 Milestone Board (Phase-A Kickoff)

This board tracks the first 20 high-impact hardening tasks to move SCPN Fusion Core from mixed validated/surrogate state toward release-grade physics and control credibility.

## Scope

- Version train: `v3.6.0-rc` -> `v3.6.0`
- Program objective: remove highest-risk simplifications from default lanes and enforce claim-evidence traceability
- Governance: every task must ship with tests + artifact evidence

## Backlog (Top 20)

| ID | Lane | Priority | Task | Deliverable | Exit Gate | Owner | Status |
|---|---|---|---|---|---|---|---|
| A01 | Governance | P0 | Maintain machine-generated underdeveloped register | `UNDERDEVELOPED_REGISTER.md` + generator script | Register regenerated in CI with no script drift | Architecture WG | In progress |
| A02 | Governance | P0 | Enforce claim-evidence linkage for public metrics | `tools/claims_audit.py` + manifest | CI fails on missing evidence for tracked claims | Docs WG | In progress |
| A03 | Governance | P0 | Split release-gate vs research-gate validations | New gate matrix in docs + CI workflow updates | Release gate excludes experimental-only lanes | Validation WG | Completed |
| A04 | Data | P0 | Expand disruption real-shot manifest and provenance fields | Versioned shot manifest with source/license/proc hash | Provenance audit passes for all shots | Validation WG | Completed |
| A05 | Data | P1 | Add leakage checks for train/val/test shot splits | Deterministic split checker in `validation/` | CI fails on shot overlap across splits | Validation WG | Backlog |
| A06 | Data | P1 | Normalize real-shot replay ingest contracts | Stricter schema + failure diagnostics | Replay pipeline rejects malformed payloads with explicit errors | Diagnostics/IO WG | Backlog |
| A07 | Control | P0 | Replace simplified disruption-risk weighting with calibrated lane | Calibrated predictor config + reproducible report | Holdout recall/FPR gates pass | Control WG | Backlog |
| A08 | Control | P0 | Upgrade SNN objective so it is not dominated by baselines | New reward/constraint shaping + benchmark table | SNN objective gap <= 5% vs strongest baseline | Control WG | Backlog |
| A09 | Control | P1 | Add full end-to-end latency accounting | End-to-end latency artifact + notebook plots | p95 loop latency published for surrogate/full modes | Control WG | Backlog |
| A10 | Control | P1 | Add actuator lag + sensor preprocessing as default replay lane | Integrated replay path with toggles and contracts | Contract tests pass on lag/preprocess invariants | Control WG | Backlog |
| A11 | Physics-Transport | P0 | Advance from single-fluid assumptions to multi-ion D/T/He-ash default | Multi-ion solver path in integrated transport | Particle+energy conservation gates hold | Core Physics WG | Backlog |
| A12 | Physics-Transport | P1 | Tighten EPED-like domain validity contracts | Domain metadata + bounded extrapolation penalties | Out-of-domain usage flagged in reports | Core Physics WG | Backlog |
| A13 | Physics-Transport | P1 | Publish uncertainty envelopes for transport metrics | p95 bands in transport reports | Uncertainty contracts validated in tests | Core Physics WG | Backlog |
| A14 | Turbulence | P0 | Remove deprecated FNO from default execution paths | Default lane switch + explicit opt-in for deprecated path | No deprecated lane in release default run | Core Physics WG | Backlog |
| A15 | Neutronics | P1 | Calibrate reduced neutronics surrogate against reference lane | Calibration artifact + error breakdown | TBR bias/error gate documented and enforced | Nuclear WG | Backlog |
| A16 | Runtime | P1 | Reduce fallback usage in critical runtime lanes | Fallback hit-rate telemetry | Fallback usage trend published and bounded | Runtime WG | Backlog |
| A17 | Runtime | P1 | Add deterministic parity checks across Python/Rust kernels | Expanded parity tests with stricter tolerances | Parity suite passes on release branch | Runtime WG | Backlog |
| A18 | Documentation | P0 | Reconcile README/RESULTS/VALIDATION version consistency | Versioned claim tables with artifact pointers | Zero stale version tags in release docs | Docs WG | Completed |
| A19 | Documentation | P1 | Add claim-to-artifact map page for all headline metrics | `docs/CLAIMS_EVIDENCE_MAP.md` | Every tracked claim resolves to artifact path | Docs WG | Completed |
| A20 | Release | P0 | Define v3.6 release acceptance checklist | Release checklist doc + CI release job | Checklist green required before tag publish | Architecture WG | Completed |

## Weekly Cadence

| Week | Focus | Mandatory Output |
|---|---|---|
| W1 | Governance + register + claim audit + gate split | Automated register, claim audit gate, board updates |
| W2 | Data/provenance + split hygiene | Shot manifest v2, split checker, ingestion contracts |
| W3 | Control calibration and SNN uplift | Updated control benchmark + latency artifact |
| W4 | Transport/neutronics fidelity + docs reconciliation | Updated physics reports + release candidate checklist |
