# Applications and Market Context

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

## Claim boundary

The repository is not yet full-fidelity end-to-end. Its current public value is
a broad, inspectable control-and-validation platform with explicit blockers and
a path to production-grade parity evidence.
