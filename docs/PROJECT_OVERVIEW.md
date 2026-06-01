# Project Overview

SCPN Fusion Core is a neuro-symbolic tokamak-control and plasma-solver research
stack. It exists to answer a practical engineering question: how can fusion
control software be developed, benchmarked, and constrained before it is placed
near real plant hardware?

## What the repository contains

- Controller surfaces: PID, H-infinity, LQR, NMPC, replay, fault tolerance,
  stochastic Petri nets, SNN compilation, and hardware-in-the-loop contracts.
- Physics surfaces: Grad-Shafranov equilibrium, GEQDSK ingestion, 1.5D
  transport, gyrokinetic research operators, electromagnetic diagnostics,
  runaway-electron contracts, impurity contracts, free-boundary validation,
  neutronics, and plant-model utilities.
- Native acceleration: Python first, with Rust crates and bindings for selected
  kernels plus polyglot solver surfaces where equivalent logic is maintained.
- Evidence surfaces: tracked validation reports, JSON schemas, benchmark
  bundles, notebooks, Sphinx docs, and fail-closed acceptance gates.

## What it is for

- Fast iteration on control algorithms against physics-informed models.
- Reproducible validation campaigns that separate accepted evidence from
  blocked or diagnostic-only rows.
- Benchmarking native kernels and accelerator paths before larger cloud or
  cluster runs.
- Teaching the software architecture of control-first fusion simulation.
- Preparing reference-code parity campaigns against established solvers.

## What it is not

- Not a completed replacement for GENE, CGYRO, GS2, DREAM, Aurora, STRAHL,
  FreeGS, EFIT, TRANSP, or JINTRAC.
- Not a certified plant-control system.
- Not a source of accepted full-fidelity parity claims unless the linked report
  marks the relevant gate as accepted and provides provenance, thresholds,
  commands, and artifact checksums.

## Evidence model

The project uses a fail-closed evidence model. A benchmark can be useful while
still blocked. Missing external same-case outputs, missing license/provenance,
missing thresholds, missing checksums, missing native comparison rows, or
missing grid/scaling evidence keep a row blocked.

## Where to verify claims

- `docs/BENCHMARKS.md` for benchmark taxonomy and commands.
- `RESULTS.md` for summarized measured results.
- `validation/reports/full_fidelity_end_to_end_campaign.md` for full-fidelity
  campaign status.
- `validation/reports/full_fidelity_acceptance_benchmark.md` for acceptance
  blockers.
- `validation/reports/production_decomposition_contract.md` for decomposition
  evidence.
- `validation/reports/gk_electromagnetic_fidelity.md` for EM fidelity gates.
