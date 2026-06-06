<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Copyright 1998-2026 Miroslav Sotek. All rights reserved.
Contact: www.anulum.li | protoscience@anulum.li
ORCID: https://orcid.org/0009-0009-3560-0851
Repository: SCPN Fusion Core
Commercial licensing available upon request.
-->

# IEC 61508 Functional-Safety Roadmap

SCPN Fusion Core is not an IEC 61508 certified product today. This roadmap
defines the evidence path required to move selected control surfaces from
research software toward an independently assessable functional-safety case.

The immediate target is an IEC 61508-aligned safety-assurance programme for
control-loop software that supports later SIL-2 assessment. Any actual SIL
claim requires independent assessment, qualified tools, validated requirements,
controlled configuration management, and evidence from representative hardware
and plant interfaces.

## Scope boundary

The roadmap applies to safety-relevant control and runtime surfaces:

- Petri-net controller contracts and inhibitor-arc interlocks.
- SNN compilation, deterministic replay, and runtime guard checks.
- PID, H-infinity, NMPC, and fallback-controller orchestration.
- WebSocket, gRPC, and hardware-in-the-loop telemetry boundaries.
- Diagnostic ingestion, normalisation, rate limiting, and replay metadata.
- Fail-closed benchmark gates and validation-report publication.

The roadmap does not certify the native physics solvers, reference-code parity
campaigns, GPU kernels, cloud benchmark runs, or reduced-order surrogate lanes.
Those surfaces provide evidence for controller testing, but they are not safety
functions unless a future safety case assigns them that role.

## Practical use in release management

This roadmap is the gatekeeper for safety-aligned claims. It is published so
partners and reviewers can distinguish implemented evidence from future work.
It also defines what must be present before control surfaces are presented as
assessment-ready in any release or deployment section.

## Current status

| Area | Current repository state | IEC 61508 gap |
|---|---|---|
| Requirements | Public docs describe control, replay, validation, and evidence boundaries. | Need controlled safety requirements with traceability to hazards and safety functions. |
| Hazard analysis | Safety interlocks and fail-closed gates exist as engineering contracts. | Need formal hazard analysis, failure-mode analysis, and allocation of safety integrity requirements. |
| Architecture | Control compiler, runtime guards, deterministic replay, and validation reports are separated. | Need safety architecture diagrams, independence arguments, and defensive partitioning evidence. |
| Verification | Unit, integration, fuzz, benchmark, Lean-proof, and replay checks exist for selected surfaces. | Need requirement-linked verification, coverage justification, tool qualification, and independent review. |
| Configuration management | Git history, CI, release artefacts, and tracked reports exist. | Need safety-baseline control, change-impact analysis, and release evidence packs. |
| Runtime evidence | Replay metadata, benchmark artefacts, and blocked/accepted evidence rows are tracked. | Need plant-interface timing evidence, fault-injection coverage, watchdog behaviour, and field diagnostics. |
| Assessment | No independent IEC 61508 assessor engagement is claimed. | Need assessor selection, assessment plan, and staged audit gates. |

## Safety lifecycle mapping

| IEC 61508 lifecycle concern | Repository deliverable | Required hardening |
|---|---|---|
| Concept and scope | Public project overview, roadmap, safety boundary docs. | Define exact safety functions and excluded surfaces. |
| Hazard and risk analysis | Interlock contracts, disruption/fallback reports, fail-closed gates. | Produce a maintained hazard log and SIL allocation rationale. |
| Safety requirements | Controller contracts, replay contracts, API docs. | Convert contracts into controlled safety requirements with unique IDs. |
| Design and implementation | Python/Rust control/runtime surfaces, formal proof slices. | Add traceability from requirements to implementation and tests. |
| Verification and validation | CI, fuzzing, deterministic replay, Lean proofs, benchmark gates. | Add independence review, test adequacy rationale, and tool qualification. |
| Operation and maintenance | Release notes, docs, deployment hardening, incident response. | Add operational procedures, anomaly reporting, and change-control evidence. |

## Gate interpretation

This roadmap is the boundary document for safety claims: it separates published
workshop capabilities from safety-assessment readiness evidence. The file does not
claim assessed SIL status and is updated as hardening milestones are completed.

## Evidence packages to build

Each package must be reproducible and auditable. A package is not accepted until
it includes requirements, commands, artefacts, checksums, pass/fail thresholds,
hardware metadata when relevant, and reviewer sign-off.

| Package | Purpose | Acceptance condition |
|---|---|---|
| Safety requirements matrix | Links hazards, safety functions, implementation surfaces, tests, and evidence. | Every safety function has traceable requirements and verification rows. |
| Deterministic replay dossier | Shows bit-for-bit replay across compiler and runtime lanes. | Same input artefact, seed, and telemetry stream reproduce identical outputs. |
| Interlock proof dossier | Captures machine-checkable proof slices for safety guards. | Lean proofs build in CI and map to named runtime contracts. |
| Fault-injection dossier | Exercises sensor noise, timing jitter, dropout, malformed messages, and unavailable actuators. | The controller enters documented fallback or fail-closed states within bounded time. |
| Runtime timing dossier | Measures control-loop timing on representative hardware. | Timing evidence records hardware, load, isolation method, commands, and thresholds. |
| Toolchain qualification dossier | Documents compiler, code generator, test tools, fuzz tools, and CI assumptions. | Tool confidence argument exists for every tool used as safety evidence. |
| Configuration baseline dossier | Freezes safety-relevant source, dependencies, containers, and documentation. | Release artefact includes SBOM, lockfiles, hashes, and change-impact notes. |

## SIL-2 readiness milestones

### Milestone 1: Safety function definition

- Identify the exact controller behaviours proposed as safety functions.
- Mark all non-safety functions and diagnostic-only surfaces.
- Define fail-safe states for controller, telemetry, and actuator boundaries.
- Publish the first safety requirements matrix.

### Milestone 2: Traceable verification

- Attach each safety requirement to a module, test, proof, or benchmark row.
- Enforce deterministic replay in CI for every safety-relevant runtime path.
- Add fuzzing and malformed-input gates for every safety-relevant parser or
  streaming interface.
- Record change-impact analysis for every safety-surface change.

### Milestone 3: Independent evidence review

- Split author, reviewer, and approver roles for safety evidence.
- Add review checklists for requirements, code, proofs, tests, and artefacts.
- Document unresolved assumptions and blocked evidence rows.
- Prepare an assessor-facing evidence index.

### Milestone 4: Representative hardware evidence

- Run timing and fault-injection bundles on representative edge, FPGA, or
  hardware-in-the-loop systems.
- Record CPU/GPU/FPGA model, firmware, kernel, runtime, load, isolation method,
  commands, artefacts, and checksums.
- Demonstrate fallback behaviour under sensor loss, network delay, malformed
  telemetry, actuator saturation, and replay mismatch.

### Milestone 5: Assessment preparation

- Select an IEC 61508 assessor or safety consultant.
- Freeze the candidate safety baseline.
- Prepare safety manual, requirements matrix, hazard log, verification dossier,
  tool qualification dossier, SBOM, dependency attestations, and release
  evidence.
- Avoid any SIL claim until the independent assessment is complete.

## Immediate repository actions

| Action | Repository surface | Status |
|---|---|---|
| Add public IEC 61508 roadmap | `docs/IEC_61508_ROADMAP.md` | Complete in this documentation slice. |
| Link roadmap from public entry points | `README.md`, `ROADMAP.md`, Sphinx index, project overview | Complete in this documentation slice. |
| Add safety requirements matrix | `docs/safety/` or `validation/reports/` | Planned. |
| Add deterministic replay evidence pack | Tests, reports, CI artefacts | Planned. |
| Add fault-injection evidence pack | Parser, streaming, telemetry, actuator tests | Planned. |
| Add tool qualification notes | CI, compiler, fuzzing, proof, benchmark tools | Planned. |
| Add assessor-facing evidence index | Public docs plus internal review checklist | Planned. |

## Claim discipline

Allowed public claim:

> SCPN Fusion Core publishes an IEC 61508-aligned roadmap for selected
> safety-relevant control surfaces and tracks the evidence required for future
> SIL-2 assessment.

Forbidden public claim until independent assessment completes:

> SCPN Fusion Core is IEC 61508 certified or SIL-2 certified.

Every release note, benchmark report, investor page, and documentation update
must preserve this distinction.
