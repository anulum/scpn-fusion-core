<!-- SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Project: SCPN Fusion Core -->
<!-- Description: Architecture decision record for solver, validation, API, and repository boundaries. -->

# SCPN Fusion Core — Architecture Decisions

This document records the durable design choices behind `scpn-fusion-core`.
It complements [`docs/ARCHITECTURE.md`](ARCHITECTURE.md), which maps the live
modules and wiring. The decisions below explain why the repository is shaped
this way, which alternatives were rejected, what trade-offs remain, and where a
reader can verify the boundary.

The document is intentionally conservative. It records design constraints and
evidence routes; it does not promote blocked validation rows into accepted
physics claims.

## Decision Index

| ID | Decision | Primary surfaces |
|---|---|---|
| ADR-001 | Keep module ownership split by responsibility | `core/`, `control/`, `io/`, `validation/`, `scpn/`, `phase/`, `hpc/`, `ui/` |
| ADR-002 | Use fastest-first dispatch with a NumPy floor | `core/_multi_compat.py`, `scpn-fusion-rs/`, dispatcher benchmarks |
| ADR-003 | Match solver algorithms to published physics contracts | `core/`, `validation/`, `docs/PHYSICS_METHODS_COMPLETE.md` |
| ADR-004 | Treat validation reports as claim gates | `validation/`, `validation/reports/`, preflight tools |
| ADR-005 | Keep public APIs thin and internal planning private | `scpn_fusion`, `docs/`, `docs/internal/`, `.coordination/` |
| ADR-006 | Keep optional and external backends fail-closed | optional extras, external solver adapters, reference runners |

## ADR-001: Module Ownership Split By Responsibility

**Decision.** The repository keeps separate package lanes for physics kernels
(`core/`), controller and replay surfaces (`control/`), data adapters (`io/`),
compiled-control artifacts (`scpn/`), phase dynamics (`phase/`), diagnostics,
nuclear/engineering utilities, optional native integration (`hpc/`), UI, and
validation/report generation.

**Rationale.** Fusion software crosses several kinds of contracts: numerical
physics, control logic, external data interchange, public evidence, and user
interfaces. Keeping those contracts in separate lanes prevents one surface from
silently changing another. A physics solver can evolve without changing the
Studio federation schema; an IMAS adapter can harden data validation without
rewriting equilibrium algorithms; a validation report can block a claim without
removing a useful local model.

**Alternatives considered.**

- One broad `fusion.py` or `engine/` module. Rejected because it would mix data
  adapters, numerical kernels, evidence rules, and UI orchestration.
- A micro-package per method. Rejected because cross-module validation reports
  need shared config paths, report conventions, and dispatch telemetry.
- Mirroring downstream repositories inside FUSION. Rejected because
  `scpn-control`, `scpn-quantum-control`, and `scpn-mif-core` have distinct
  ownership boundaries.

**Trade-offs.** The split creates more files and more explicit wiring. That is
acceptable because direct-test linkage, capability manifests, and Sphinx API
pages make missing tests or unwired modules visible.

**Constraints.**

- New production modules need direct tests and capability-manifest refresh.
- New compute kernels must state whether a Rust or other backend is equivalent,
  future work, or intentionally absent.
- Cross-repository contracts are recorded in `docs/ARCHITECTURE.md`, not copied
  as duplicate code into sibling repositories.

## ADR-002: Fastest-First Dispatch With A NumPy Floor

**Decision.** Production compute dispatch uses a priority order in
`core/_multi_compat.py`:

```text
RUST -> GPU -> MOJO -> JULIA -> GO -> JAX -> NUMPY
```

The selected provider must expose the same production contract as the NumPy
floor. When no accelerator is available, NumPy remains the compatibility
baseline.

**Rationale.** Optional native speedups are useful only when they preserve the
same contract. The dispatcher records fallback telemetry so a benchmark or
runtime can distinguish "fastest tier selected" from "NumPy floor selected".
This keeps acceleration from becoming an implicit claim that a backend is
available, equivalent, or faster on every host.

**Alternatives considered.**

- Direct `import scpn_fusion_rs` at call sites. Rejected because it hides
  fallback decisions and makes optional-extension absence harder to test.
- Always prefer JAX when installed. Rejected because JAX is an optional research
  tier here; its differentiability value is separate from production dispatch
  equivalence.
- Register every PyO3 export as a production tier. Rejected because several
  native exports are library-only or not yet reconciled against a Python twin.

**Trade-offs.** Each registered kernel needs parity tests, fallback tests, and
benchmark evidence. That slows registration, but it prevents accelerator
availability from being confused with algorithmic equivalence.

**Constraints.**

- Every dispatched kernel keeps a NumPy floor unless the surface is explicitly
  documented as Rust-only.
- A compiled backend that changes algorithmic meaning is not a dispatch tier.
- GPU timing reports must distinguish functional local evidence from isolated
  production benchmark evidence.

## ADR-003: Solver Algorithms Match Published Physics Contracts

**Decision.** Solver implementations are tied to named physics or numerical
contracts, and validation reports state the exact boundary that was tested. The
main anchors are:

| Solver family | Design choice | Reference/evidence anchor |
|---|---|---|
| Axisymmetric equilibrium | Grad-Shafranov operator with Picard/Newton/SOR/multigrid paths | Grad and Rubin (1958); Shafranov (1966); Solov'ev validation reports |
| 1.5D transport | Crank-Nicolson-style implicit diffusion with explicit claim boundaries against TORAX artifacts | `validation/reports/torax_*`; transport conservation tests |
| Confinement scaling | IPB98(y,2) as a non-surrogate confinement baseline | ITER Physics Basis, Nucl. Fusion 39 (1999) |
| Bootstrap and pedestal terms | Sauter bootstrap terms, Snyder EPED-style pedestal scaling, Connor-Hastie-Taylor ballooning boundary where used | `docs/PHYSICS_METHODS_COMPLETE.md`; EPED reports |
| Runaway-electron rates | Source-checked Connor-Hastie / Rosenbluth-Putvinski-style reduced formulas with DREAM comparison boundaries | DREAM reference reports |
| FRC rigid rotor | Source-verified Rostoker and Qerushi one-dimensional rigid-rotor closure for the rotating lane | FRC acceptance reports |
| Petri-net control compiler | Stochastic Petri-net structure compiled into SNN control artifacts with safety contracts | `scpn/`, Lean proof surfaces, safety traceability |
| Kuramoto/UPDE phase lane | Kuramoto-Sakaguchi and UPDE kernels with Rust/NumPy parity where registered | phase tests and replay certificates |

**Rationale.** The repository cannot rely on naming alone. A solver is credible
only when its algorithm, inputs, units, acceptance threshold, and evidence
surface are traceable. This is why the same codebase can contain a useful
local model while still keeping external parity blocked.

**Alternatives considered.**

- Treat surrogate outputs as replacements for published physics formulas.
  Rejected; surrogate cards keep fallback behavior and uncertainty/OOD gaps
  explicit.
- Accept local agreement without external same-case output. Rejected for
  external parity claims; local agreement remains local evidence.
- Collapse solver docs into README prose. Rejected because readers need method
  references, report links, and blocked rows close to the technical surface.

**Trade-offs.** Published-method anchoring increases documentation work and can
leave rows blocked when source data is not redistributable. That is preferable
to overstating the maturity of a local implementation.

**Constraints.**

- If a solver contract changes, update tests, validation reports, public docs,
  and any equivalent backend in the same commit.
- If an external reference is unavailable or not redistributable, keep the row
  blocked and name the missing artifact.
- If a native/backend implementation uses a different model, do not register it
  as equivalent until the contract is reconciled.

## ADR-004: Validation Reports Are Claim Gates

**Decision.** Public claims point to tracked JSON/Markdown reports under
`validation/reports/`. Report generators own schemas, status fields, thresholds,
checksums, provenance, and blocked-row explanations. The public documentation
links to reports instead of converting local outputs directly into claims.

**Rationale.** A benchmark result can be useful and still blocked. The report
layer lets the repository publish local evidence while making missing external
outputs, license metadata, hardware timing, or grid/scaling evidence visible.

**Alternatives considered.**

- Put benchmark numbers directly in README only. Rejected because README text
  drifts and lacks machine-readable status.
- Keep validation evidence internal until every row passes. Rejected because
  blocked reports are useful for planning and review when their boundary is
  explicit.
- Make every validation script a release blocker. Rejected because some reports
  are diagnostic, while release preflight should run the selected stable gates.

**Trade-offs.** The report layer adds generator code and drift checks. The
benefit is that public docs can be audited against machine-readable evidence
instead of hand-maintained prose.

**Constraints.**

- A report that lacks required external evidence must remain fail-closed.
- Generated report changes require corresponding docs and checks when the
  public surface changes.
- Benchmark timing claims need recorded command, hardware/runtime context, and
  isolated-core evidence when presented as production timing.

## ADR-005: Thin Public APIs And Private Planning Surfaces

**Decision.** The package root stays thin. Public users enter through domain
subpackages (`scpn_fusion.core`, `control`, `io`, `scpn`, `phase`, `studio`,
`ui`) or CLIs. Internal planning, TODOs, reviews, and coordination records stay in
ignored `docs/internal/` or monorepo `.coordination/` paths.

**Rationale.** A broad root API makes compatibility promises too early. Domain
subpackages can carry their own tests, docs, and validation contracts. Private
planning documents often include unfinished strategy, blocked evidence, or
coordination details; publishing them as public docs would confuse readers and
weaken claim hygiene.

**Alternatives considered.**

- Export most classes from `scpn_fusion.__init__`. Rejected because it would
  enlarge the stable API and import optional dependencies too early.
- Track internal TODOs in public docs. Rejected because active planning is not
  public evidence.
- Hide all architecture status internally. Rejected because public users need a
  map of stable surfaces and limitations.

**Trade-offs.** Users need to import from subpackages rather than one root
namespace. The benefit is a smaller compatibility surface and clearer ownership.

**Constraints.**

- Public docs describe implemented and evidence-linked surfaces only.
- Internal coordination files are never committed as public docs.
- Public API changes need direct tests and documentation updates.

## ADR-006: Optional And External Backends Fail Closed

**Decision.** Optional dependencies and external solvers are soft integrations.
When a dependency, binary, license, or same-case output is missing, adapters
return blocked status, raise explicit import/runtime errors, or skip optional
branches in tests. They do not silently produce acceptance evidence.

**Rationale.** Fusion validation depends on tools and datasets that may be
licensed, platform-specific, or too large to bundle. A fail-closed adapter keeps
local development usable while preserving the boundary between "adapter exists"
and "external comparison accepted".

**Alternatives considered.**

- Vendor external solver outputs into the repository. Rejected unless license,
  provenance, size, and redistribution rules are satisfied.
- Treat missing optional dependencies as hard install failures. Rejected because
  the base package must run without UI, ML, GPU, MPI, OMAS, and external solver
  extras.
- Generate synthetic references for blocked external rows. Rejected as evidence
  for parity; synthetic data can be diagnostic only.

**Trade-offs.** Some tests are skipped when optional dependencies are absent,
and some reports remain blocked. That is acceptable because the blocked state is
visible and reproducible.

**Constraints.**

- Optional dependency paths need real tests with local test doubles or installed
  extras where practical.
- External-solver rows need same-case outputs, provenance, thresholds,
  checksums, license posture, and native comparison before acceptance.
- Hardware/HIL rows need device identity, clock source, trace checksums,
  latency/jitter evidence, and replay metadata before hardware claims.

## How To Use This Record

- When adding a solver, identify its ADR-003 method anchor and its validation
  report before changing public docs.
- When adding an accelerator, register it only after ADR-002 parity and
  fallback evidence exists.
- When adding an external adapter, keep ADR-006 fail-closed behavior until the
  external evidence package is present.
- When adding docs, keep ADR-005 public/internal boundaries intact.
- When changing validation claims, update ADR-004-linked reports and public
  references together.
