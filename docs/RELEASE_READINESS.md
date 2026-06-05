# v3.9.10 Release Acceptance

Release Version: `v3.9.10`
Readiness State: `ready`

This file is a tracked, non-sensitive release gate artifact. It records the
public release checklist required by the tag-publish workflow. Private provider
credentials, internal handovers, and operational scratch notes remain in ignored
internal locations.

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

- `docs/releases/v3.9.10.md`
- `CHANGELOG.md`
- `README.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/ONBOARDING.md`
- `docs/API_OVERVIEW.md`
- `docs/APPLICATIONS_AND_MARKET.md`
- `docs/BENCHMARKS.md`

## Evidence boundary

The release is documentation and repository-polish focused. It does not promote
blocked full-fidelity solver parity gates to accepted status.
