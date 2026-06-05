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

## What a reviewer should understand first

SCPN Fusion Core is valuable because it turns fusion-control research into an
auditable evidence pipeline. A contributor can write a controller, run it
against local physics contracts, capture deterministic replay metadata, compare
native kernels across Python and Rust, and then keep the claim blocked until
external same-case reference output exists.

This distinction matters commercially. It lets a laboratory, startup, or
investor see which money buys software hardening, which money buys reference
solver execution, and which money buys production-scale hardware evidence. The
repository is therefore both a codebase and a validation operating model.

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


## How to read the maturity status

The repository uses explicit maturity labels so readers can tell what is ready
for local development and what still needs external parity evidence.

| Label | Meaning |
|---|---|
| Validated local contract | The implementation has repository-local tests, reports, or benchmark evidence for the stated scope. |
| Diagnostic only | The output is useful for debugging or planning, but it is not an acceptance gate. |
| Blocked parity gate | Required public same-case data, reference-solver output, thresholds, checksums, or hardware evidence are missing. |
| Accepted parity evidence | The row has provenance, license, command, artifact checksum, thresholds, native comparison, and pass/fail status. |

## Practical value chain

1. Define the control or solver contract.
2. Run a local deterministic or physics-informed benchmark.
3. Publish the command, inputs, outputs, checksums, hardware metadata, and thresholds.
4. Compare against external same-case reference outputs where redistribution is permitted.
5. Keep the row blocked until the evidence package is complete.

This workflow is the market value of the project: it reduces ambiguity between a
prototype, a benchmark result, and a defensible validation claim.

## Fifteen-minute orientation path

1. Read `README.md` for the product boundary and current headline reports.
2. Read this overview to understand the evidence model and maturity labels.
3. Follow `docs/ONBOARDING.md` to install and run the first local commands.
4. Use `docs/API_OVERVIEW.md` to find the Python, Rust, and validation entry points.
5. Use `docs/BENCHMARKS.md` and `validation/reports/` before citing any number.
