<!--
SCPN Fusion Core — Phase 3 Execution Registry
© 1998–2026 Miroslav Šotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
License: GNU AGPL v3 | Commercial licensing available
-->

# Phase 3 Execution Registry and Sprint S2 Queue

Date: 2026-02-13

This registry tracks the high-impact enhancement backlog imported after Phase 2 closure.

## Registry Size

| Source Pack | Imported Tasks |
|---|---|
| Control and simulation enhancement plan | 20 |
| HPC solver enhancement plan | 21 |
| Nuclear simulation enhancement plan | 19 |
| SCPN compiler enhancement plan | 25 |
| Total imported Phase 3 backlog | 85 |

Current tracker baseline (`docs/PHASE2_ADVANCED_RFC_TRACKER.md`): 20/20 tasks complete (`Done`).

## Prioritization Guardrails

- Keep strict green gates before each merge:
  - `cargo clippy --all-targets --all-features -- -D warnings`
  - `cargo test --all-features`
  - `python -m pytest -v`
- Maximum active work-in-progress: 2 tasks.
- Prefer changes that improve determinism, testability, and runtime safety before model scope expansion.
- External-data-dependent work stays blocked until data/license readiness is explicit in RFC.

## Sprint S2 Execution Queue (Ordered)

| ID | Priority | Track | Task | Target Files | Definition of Done | Validation |
|---|---|---|---|---|---|---|
| S2-001 | P0 | HPC | Add in-place zero-allocation bridge solve paths (`solve_into`, converged variant) | `src/scpn_fusion/hpc/hpc_bridge.py`, `tests/test_hpc_bridge.py` | Reusable output buffer paths exist with strict shape/layout checks and tests | `python -m pytest tests/test_hpc_bridge.py -v` |
| S2-002 | P0 | HPC | Harden C++ SOR convergence API with explicit bad-input guards | `src/scpn_fusion/hpc/solver.cpp`, `tests/test_hpc_bridge.py` | Null/size/iteration guards verified by tests and no clippy/format regressions | `python -m pytest tests/test_hpc_bridge.py -v` |
| S2-003 | P0 | SCPN | Add stochastic-vs-float equivalence benchmark gate (`<=5%` error) | `src/scpn_fusion/scpn/controller.py`, `validation/gneu_01_benchmark.py`, `tests/` | Deterministic campaign metric exposed and thresholded in tests | `python -m pytest tests/test_gneu_01_benchmark.py -v` |
| S2-004 | P1 | Control | Add robust model-loading fallback path in disruption predictor | `src/scpn_fusion/control/disruption_predictor.py`, `tests/` | Missing-model path no longer crashes and emits deterministic fallback score | `python -m pytest tests/test_gneu_02_anomaly.py -v` |
| S2-005 | P1 | Nuclear | Increase PWI coverage for angle-energy invariants + redeposition bounds | `src/scpn_fusion/nuclear/pwi_erosion.py`, `tests/test_pwi_erosion.py` | Invariants captured in tests with deterministic tolerances | `python -m pytest tests/test_pwi_erosion.py -v` |
| S2-006 | P1 | Nuclear | Extend TEMHD solver regression with pathological edge cases | `src/scpn_fusion/nuclear/temhd_peltier.py`, `tests/test_temhd_peltier.py` | Singular/near-singular stability behavior documented and tested | `python -m pytest tests/test_temhd_peltier.py -v` |
| S2-007 | P2 | Docs | Normalize path mappings for all imported tasks to repository modules | `docs/3d_gaps.md`, `docs/PHASE3_EXECUTION_REGISTRY.md` | Every queued task maps to real paths only | Docs review |
| S2-008 | P2 | Release | Add queue-level release gate summary for S2 | `validation/gdep_05_release_readiness.py`, `tests/test_gdep_05_release_readiness.py` | Report includes S2 queue health section without breaking current checks | `python -m pytest tests/test_gdep_05_release_readiness.py -v` |

## Sprint S3 Execution Queue (Ordered)

| ID | Priority | Track | Task | Target Files | Definition of Done | Validation |
|---|---|---|---|---|---|---|
| S3-001 | P0 | SCPN | Add topology diagnostics for dead nodes and unseeded place cycles | `src/scpn_fusion/scpn/structure.py`, `tests/test_scpn_compiler.py` | `validate_topology()` + optional strict compile validation implemented and tested | `python -m pytest tests/test_scpn_compiler.py tests/test_hypothesis_properties.py -v` |
| S3-002 | P0 | SCPN | Add inhibitor-arc support with explicit opt-in semantics | `src/scpn_fusion/scpn/structure.py`, `tests/test_scpn_compiler.py` | Inhibitor arc definition supported without breaking default positive-weight semantics | `python -m pytest tests/test_scpn_compiler.py -v` |
| S3-003 | P1 | SCPN | Add compact artifact serialization mode for packed bitstreams | `src/scpn_fusion/scpn/artifact.py`, `tests/test_controller.py` | Optional compact export/import path round-trips deterministically | `python -m pytest tests/test_controller.py -v` |
| S3-004 | P1 | Control | Normalize control simulation imports and deterministic fallback entry points | `src/scpn_fusion/control/disruption_predictor.py`, `src/scpn_fusion/control/tokamak_digital_twin.py`, `tests/test_gneu_02_anomaly.py` | No hard crash on missing model/dependency paths in core control entry points | `python -m pytest tests/test_gneu_02_anomaly.py -v` |
| S3-005 | P1 | HPC | Extend bridge/solver validation for invalid stride/shape edge paths | `src/scpn_fusion/hpc/hpc_bridge.py`, `src/scpn_fusion/hpc/solver.cpp`, `tests/test_hpc_bridge.py` | Additional boundary/fuzz-style guards and tests for invalid buffer geometry | `python -m pytest tests/test_hpc_bridge.py -v` |
| S3-006 | P2 | Release | Add S3 queue health visibility in release-readiness markdown | `validation/gdep_05_release_readiness.py`, `tests/test_gdep_05_release_readiness.py` | Release report captures S3 progress without weakening existing hard gates | `python -m pytest tests/test_gdep_05_release_readiness.py -v` |

## Sprint S4 Execution Queue (Ordered)

| ID | Priority | Track | Task | Target Files | Definition of Done | Validation |
|---|---|---|---|---|---|---|
| S4-001 | P0 | SCPN | Wire topology/inhibitor compile controls through `FusionCompiler` | `src/scpn_fusion/scpn/compiler.py`, `tests/test_scpn_compiler.py` | Compiler can enforce topology checks and inhibitor opt-in at compile entry point | `python -m pytest tests/test_scpn_compiler.py tests/test_hypothesis_properties.py -v` |
| S4-002 | P1 | SCPN | Add deterministic compact-packed artifact codec smoke checks independent of runtime packed availability | `src/scpn_fusion/scpn/artifact.py`, `tests/test_controller.py` | Compact codec validated directly for deterministic roundtrip of u64 payloads | `python -m pytest tests/test_controller.py -v` |
| S4-003 | P1 | Control | Add deterministic safe runtime summary path for `tokamak_flight_sim` | `src/scpn_fusion/control/tokamak_flight_sim.py`, `tests/` | Control script callable in CI without interactive plotting dependency | `python -m pytest -v` |
| S4-004 | P2 | Release | Extend release-readiness queue health to include S4 lane | `validation/gdep_05_release_readiness.py`, `tests/test_gdep_05_release_readiness.py` | Release report includes S2/S3/S4 queue snapshots | `python -m pytest tests/test_gdep_05_release_readiness.py -v` |

## Task Accounting

- Total imported tasks: 85
- Tasks currently queued for Sprint S2: 8
- Tasks currently queued for Sprint S3: 6
- Tasks currently queued for Sprint S4: 4
- Remaining in deferred pool after queue selection: 67

## Active Task

- Completed: `S2-001`
- Completed: `S2-002`
- Completed: `S2-003`
- Completed: `S2-004`
- Completed: `S2-005`
- Completed: `S2-006`
- Completed: `S2-007`
- Completed: `S2-008`
- Completed: `S3-001`
- Completed: `S3-002`
- Completed: `S3-003`
- Completed: `S3-004`
- Completed: `S3-005`
- Completed: `S3-006`
- Completed: `S4-001`
- Completed: `S4-002`
- Completed: `S4-003`
- Completed: `S4-004`
- Next active task: none (Sprint S4 queue baseline closed; deferred pool unchanged).
