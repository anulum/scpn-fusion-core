# Applications and Market Context


## Positioning and readers

This document translates solver and control capabilities into practical deployment contexts. It is intended to help operators, technical buyers, and collaborators interpret what is demonstrably ready versus what still requires parity evidence.

## Scope and factual use

This page is the application-side orientation for external review. It connects each deployment area to the minimum evidence package needed before a readiness claim can be treated as production-facing.

For every listed use case, the repository boundary is: local reproducibility first, reference comparison second, and safety-assurance or control certification claims only after the supporting evidence set is complete. If a downstream user expects a vendor-style readiness signal from this page, they should confirm that the linked validation and benchmark contracts are published and currently passing for that use case.

SCPN Fusion Core targets the gap between fast control-algorithm development and
high-fidelity fusion-code validation. The commercial value is in reducing the
time and ambiguity between a control idea, a reproducible simulation campaign,
and a reference-code-backed evidence package.

## Primary application areas

| Area | Practical use | Evidence needed before production use |
|---|---|---|
| Tokamak control R&D | Prototype controllers, replay campaigns, deterministic stress tests | Facility-specific HIL, actuator contracts, safety review |
| Solver validation | Compare native kernels against public or partner reference outputs | Same-case decks, thresholds, checksums, grid convergence |
| Accelerator benchmarking | Measure CPU, Rust, MPI, CUDA, and cloud GPU paths | Reproducible hardware metadata and archived artifacts |
| Education and onboarding | Teach control-first fusion simulation architecture | Clear examples and bounded claims |
| Formal-methods research | Expand machine-checked safety boundaries | More Lean/TLA-style proofs tied to implementation contracts |

## What buyers can value before plant deployment

The near-term product is not a certified controller. It is a validation and
control-development platform that can reduce engineering ambiguity before
expensive facility time is used. The strongest commercial value propositions
are:

| Value proposition | Why it matters |
|---|---|
| Faster controller iteration | Teams can test logic, replay faults, and compare runtime behavior before committing to plant-specific integration. |
| Evidence-backed funding decisions | Blocked rows show exactly which reference runs, licenses, hardware, or datasets are missing. |
| Accelerator budget clarity | CPU, Rust, CUDA, MPI, and hardware metadata can be separated from unsupported performance prose. |
| Safer collaboration boundary | Public artifacts can be audited while proprietary facility data remains outside the repository. |
| Training and onboarding | New contributors can learn fusion control software through reproducible examples instead of informal demos. |

## Differentiation

- Control-first architecture rather than physics-only architecture.
- Evidence gates that preserve blocked states instead of hiding them.
- Native solver contracts with polyglot parity where equivalent logic exists.
- Public documentation that separates current local evidence from required
  full-fidelity reference-code parity.
- A route from notebook/demo workflows to tracked benchmark artifacts.

## Buyer and collaborator profiles

- Fusion startups evaluating control software architecture.
- National laboratories and universities building validation pipelines.
- HPC and accelerator teams evaluating solver kernels and scaling bottlenecks.
- Safety and assurance teams interested in deterministic replay and formal
  safety contracts.
- Education and training programs that need reproducible, inspectable examples.

## Evaluation packages

The repository becomes easier to evaluate when the commercial question is
converted into a bounded evidence package. These packages are possible today;
none is a plant-deployment certificate.

| Evaluation package | Partner input | Repository output | Explicitly not included |
|---|---|---|---|
| Controller replay | Scenario, actuator limits, telemetry schema | Deterministic trace, fault injections, timing taxonomy, pass/fail criteria | Facility authority or certified interlock |
| Same-case solver comparison | Redistributable deck and trusted reference output | Native result, convention audit, error metrics, checksums, blocked/accepted verdict | A claim that the native solver replaces the reference code |
| Accelerator study | Hardware target and representative workload | Reproducible CPU/Rust/CUDA/MPI measurements with environment metadata | Extrapolated cluster or plant throughput |
| Studio federation review | Required verbs, schemas, and exactness class | Machine-readable capability/evidence manifest and reproduction comparator | Automatic promotion of simulated evidence |
| Training workshop | Audience level and learning outcome | Curated quickstart, notebooks, evidence-reading exercises | Operator qualification or safety certification |

## What creates defensible value

Market value here is the reduction of technical uncertainty. A useful engagement
should leave behind artifacts that another reviewer can inspect: a replay trace,
same-case comparison, hardware report, schema, checksum, or explicit blocked row.
That makes the result portable across engineering, investment, and assurance
reviews without turning exploratory output into a readiness claim.

## Current investment case

The near-term financing need is evidence generation: same-deck reference-solver
runs, GPU/cluster scaling campaigns, DREAM/Aurora/STRAHL parity artifacts, and
FreeGS/free-boundary sidecar data. Funding should buy reproducibility and
independent comparability, not inflated claims.

The highest-leverage inputs are therefore concrete: legally usable external
outputs, facility-specific replay data, isolated accelerator time, independent
replication, and safety-assurance expertise. Each closes a named evidence gap;
none is represented as buying scientific acceptance by itself.

## Why this matters to external readers

This page is not a sales-only document; it is a practical evidence index. Readers
are expected to distinguish:

- technical readiness from deployment readiness,
- benchmark scripts from accepted parity gates,
- local proof-of-concept value from certified safety status.

The immediate commercial value is the ability to move from a controller idea to
an auditable evidence set with explicit blockers and acceptance criteria.

## Why the repository has market value before full parity

The current value is not a claim that the native solver replaces production
fusion codes. The value is the infrastructure that makes those claims testable:
schemas, provenance fields, blocked states, reproducible commands, native
kernel paths, Rust parity surfaces, notebooks, and public evidence reports.

That gives partners a lower-risk path to collaboration. They can bring a
reference deck, a shot replay, a hardware target, or a validation requirement
and see exactly what must pass before a claim changes from local contract to
accepted parity evidence.

## Claim boundary

The repository is not yet full-fidelity end-to-end. Its current public value is
a broad, inspectable control-and-validation platform with explicit blockers and
a path to production-grade parity evidence.


## Market value by evidence package

| Evidence package | Why it matters commercially |
|---|---|
| Deterministic replay | Makes controller regressions auditable across software and hardware targets. |
| Reference-solver parity manifests | Lets partners compare native kernels against trusted tools without vague claims. |
| Formal proof artifacts | Creates a path toward safety-assurance arguments for high-consequence control software. |
| GPU/MPI scaling reports | Converts performance claims into budgetable infrastructure plans. |
| Open validation datasets | Reduces onboarding cost for collaborators and makes third-party replication possible. |

## Near-term audience-specific value

- Fusion startups can evaluate controller architecture and validation workflow before committing plant-specific data.
- Laboratories can use the fail-closed report pattern for reproducible benchmark campaigns.
- Hardware teams can plug in GPU, FPGA, neuromorphic, or HIL backends behind explicit replay contracts.
- Formal-methods teams can extend proof coverage toward compiler and runtime guarantees.
- Investors can see which money buys external data, reference runs, hardware time, or safety evidence.
