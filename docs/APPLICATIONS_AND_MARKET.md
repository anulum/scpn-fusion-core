# Applications and Market Context


## Positioning and readers

This document translates solver and control capabilities into practical deployment contexts. It is intended to help operators, technical buyers, and collaborators interpret what is demonstrably ready versus what still requires parity evidence.

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

## Current investment case

The near-term financing need is evidence generation: same-deck reference-solver
runs, GPU/cluster scaling campaigns, DREAM/Aurora/STRAHL parity artifacts, and
FreeGS/free-boundary sidecar data. Funding should buy reproducibility and
independent comparability, not inflated claims.

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
