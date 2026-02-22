# Release Acceptance Checklist

Release Version: `v3.9.1`  
Checklist State: `ready`

This checklist must be fully green before publishing a release tag.

## Required Checks

- [x] Release preflight (`python tools/run_python_preflight.py --gate release`)
- [x] Research preflight (`python tools/run_python_preflight.py --gate research`)
- [x] Claims audit and claims evidence map are up to date
- [x] Underdeveloped register regenerated in current branch
- [x] Version metadata and release docs are consistent
- [x] Changelog contains the release section and date
- [x] CI workflow on `main` is green for the release commit
- [x] Tag/release notes reviewed and approved
