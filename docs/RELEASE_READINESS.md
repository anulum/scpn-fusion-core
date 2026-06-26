# v3.10.0 Release Acceptance


## Evidence and governance meaning

This checklist is the public release-control gate for repository-wide changes.
Its role is to make scope changes traceable before any release tag and to keep
claim transitions explicit.

The checklist is not a compliance certificate: it records that required
preflight and documentation conditions are checked before tagging and that no
evidence requirement was silently dropped.

This checklist is the operational gate for release publication. It tracks required checks and references for traceable, repeatable release decisions.

Release Version: `v3.10.0`
Readiness State: `ready`

This file is a tracked, non-sensitive release gate artifact. It records the
public release checklist required by the tag-publish workflow. Private provider
credentials, internal handovers, and operational scratch notes remain in ignored
internal locations.

## What is verified before a release commit

Before a tag is considered ready, the release gate ties public-facing behavior
changes, benchmark contracts, and documentation to current artifacts and
preflight checks so claim state changes are evidence-backed.

## Checklist

- [x] Release preflight (`python tools/run_python_preflight.py --gate release`)
- [x] Research preflight (`python tools/run_python_preflight.py --gate research`)
- [x] Claims audit, claim-range guard, and claims evidence map are up to date
- [x] Internal readiness register regenerated in current branch
- [x] Version metadata and release docs are consistent
- [x] Changelog contains the release section and date
- [x] CI workflow on `main` is green for the release commit
- [x] Tag/release notes reviewed and approved
- [x] TestPyPI publish is manual-only unless its trusted publisher is configured

## Release evidence links

- `docs/releases/v3.10.0.md`
- `CHANGELOG.md`
- `README.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/ONBOARDING.md`
- `docs/API_OVERVIEW.md`
- `docs/APPLICATIONS_AND_MARKET.md`
- `docs/BENCHMARKS.md`

## Evidence boundary

The release covers runtime hardening, Studio federation surfaces, documentation
alignment, dependency maintenance, and repository-polish work. It does not
promote blocked full-fidelity solver parity gates to accepted status.

## Public release boundary

All tags reference concrete evidence files in this repository. A release should
not shift a blocker state from "blocked" to "accepted" by language changes in
documentation; only passing validation artifacts and updated benchmark contracts are
allowed to change a claim state.
