# Future Applications


## Vision and boundary

This document collects future application vectors. It is a planning artifact that ties candidate directions to expected technical dependencies and evidence requirements.

## Position for roadmap readers

This page is used as a public filter between concept exploration and claim-ready
work. Every lane below is linked to the minimum public evidence needed before it can be treated as production-facing.

## How this document is used

This document distinguishes strategic direction from validated production claims.
Each future lane must map to explicit public evidence before it is treated as
deployment-ready.

SCPN Fusion Core is a control-and-validation platform under active development.
This page describes plausible application areas and the evidence needed before
any of them should be treated as production claims.

## 1. Real-time tokamak control research

Current value:

- Encode controller logic as stochastic Petri nets and compile it into SNN
  execution paths.
- Compare classical PID, H-infinity, LQR, NMPC, and replay controllers under
  reproducible simulation campaigns.
- Publish latency and disruption-risk diagnostics with explicit hardware and
  benchmark provenance.

Evidence still needed for plant deployment:

- Facility-specific actuator, diagnostic, timing, and safety-system contracts.
- Hardware-in-the-loop campaigns with plant-representative I/O latency.
- Independent review of safety interlocks, fallback behavior, and operator
  override paths.

## Lane entry criteria

A future application enters active execution only when the needed validation
artifacts are published and linked from `validation/reports/`. Until then it
remains a roadmap item, not a claimed production lane.

## 2. Fusion solver validation infrastructure

Current value:

- Fail-closed benchmark contracts for Grad-Shafranov, GEQDSK, FreeGS,
  nonlinear gyrokinetics, electromagnetic GK, runaway electrons, impurities,
  and production decomposition.
- Public reports that keep missing same-case reference outputs blocked instead
  of substituting synthetic or reduced-order evidence.
- Polyglot parity surfaces where equivalent solver logic exists.

Evidence still needed:

- Redistribution-permitted same-deck nonlinear GENE/CGYRO/GS2 outputs.
- DREAM and Aurora/STRAHL same-case output artifacts.
- Public coil-current and machine-sidecar data for strict free-boundary parity.
- Cluster and multi-GPU scaling evidence with reproducible hardware metadata.

## 3. Industrial control software assurance

Current value:

- Machine-readable benchmark reports and acceptance gates.
- Lean 4 proofs for narrow fail-closed controller/solver safety boundaries.
- Deterministic replay contracts for selected stochastic-controller surfaces.

Evidence still needed:

- A broader formal proof surface covering actuator bounds, controller fallback,
  and compiler preservation properties.
- Independent security and safety reviews.
- Facility-specific validation matrices.

## 4. Accelerator and cloud benchmarking

Current value:

- Provider-neutral benchmark bundles for control diagnostics, native solvers,
  and full-fidelity campaign reporting.
- Local CPU, Rust, MPI, and optional CUDA/MPI evidence lanes.

Evidence still needed:

- Production-scale GPU and cluster runs across declared large-grid workloads.
- Published cost, throughput, and efficiency thresholds per hardware class.
- Artifact archives with checksums, commands, logs, and environment metadata.

## 5. Market relevance

Potential buyers or collaborators include fusion startups, national labs,
university plasma-control groups, industrial simulation teams, and control
software assurance groups. The practical value proposition is not that this
repository replaces established physics codes today. The value is a transparent
control-and-validation workbench that can connect fast control experiments to
strict reference-code evidence as those reference artifacts become available.

## Current status summary

The platform is suitable for research, benchmarking, education, controller
architecture work, and validation-pipeline development. It is not yet a
production-certified plant controller or a full-fidelity replacement for
GENE/CGYRO/GS2, DREAM, Aurora/STRAHL, FreeGS, EFIT, TRANSP, or JINTRAC.
