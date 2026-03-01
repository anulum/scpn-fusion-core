# SCPN Fusion Core: 30-Day Hardening Execution Plan

Baseline snapshot (generated on 2026-02-28):
- Underdeveloped flags: 297 total, 92 P0/P1 (`UNDERDEVELOPED_REGISTER.md`)
- Target by day 30: <= 70 P0/P1 with no regression in CI release lanes

## Scope Guardrails

- Prioritize physics credibility, validation rigor, and runtime hardening.
- Defer non-core expansion unless it directly improves validation or adoption.
- Keep release claims tied to `docs/CLAIMS_EVIDENCE_MAP.md` and `RESULTS.md`.

## Workstream A: Physics and Validation Credibility (P0)

Week 1
- Lock default turbulence path to non-deprecated lanes only.
- Keep deprecated FNO paths non-default and explicitly gated.
- Add claim/evidence check for every benchmark table row touched.

Week 2
- Expand real-data validation ingestion path (SPARC/ITPA first).
- Add stricter validation gate checks for data provenance and holdouts.
- Publish updated benchmark delta table vs previous release artifacts.

Week 3
- Add FreeGS parity benchmark lane and CI artifact export.
- Define pass/fail threshold for free-boundary parity against reference shots.

Week 4
- Regenerate claims maps + underdeveloped register and close stale claims.
- Freeze v4.0 physics credibility checklist and acceptance criteria.

Exit criteria
- P0 physics/validation flags reduced materially from baseline.
- No unresolved claim without evidence mapping.
- Free-boundary benchmark lane running in CI and archived in artifacts.

## Workstream B: Packaging and Adoption (P1)

Week 1
- Ensure prebuilt wheel coverage for primary platforms in release workflow.
- Add one-command demo path contract to CLI (`scpn-fusion demo`).

Week 2
- Add short demo artifact (GIF/video) and top-fold README linkage.
- Add architecture visual near quick-start.

Week 3
- Publish reproducibility stamp definition (release benchmark gate).
- Add release checklist step that verifies benchmark reproducibility artifacts.

Exit criteria
- New users can run a demo without local Rust toolchain setup.
- README path-to-value is <= 60 seconds with visible architecture context.

## Workstream C: Sustainability and Community (P1)

Week 1
- Add/verify issue templates: bug, feature, validation-data, paper-review.
- Seed Discussions with roadmap and validation-collaboration threads.

Week 2
- Convert underdeveloped P0/P1 backlog into a tracked project board.
- Add labels for owner domains (`core_physics`, `validation`, `control`, `docs`).

Week 3
- Publish contribution path for validation data with schema/provenance rules.
- Announce focused collaboration asks (validation data + physics review).

Exit criteria
- External contributors can identify a data/physics contribution path quickly.
- P0/P1 backlog has owner, milestone, and evidence-linked closure criteria.

## Weekly KPI Pack

- Underdeveloped totals and P0/P1 counts
- CI release lane pass rate and mean duration
- Number of claims with explicit evidence mapping
- Number of real-data validation entries onboarded
- New external contributors/issues/discussions in physics-validation lanes
