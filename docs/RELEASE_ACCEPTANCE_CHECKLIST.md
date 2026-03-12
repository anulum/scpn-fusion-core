# Release Acceptance Checklist

Release Version: `v3.9.3`
Checklist State: `ready`

This checklist must be fully green before publishing a release tag.

## Required Checks

- [x] Release preflight (`python tools/run_python_preflight.py --gate release`)
- [x] Research preflight (`python tools/run_python_preflight.py --gate research`)
- [x] Claims audit, claim-range guard, and claims evidence map are up to date
- [x] Underdeveloped register regenerated in current branch
- [x] Version metadata and release docs are consistent
- [x] Packaging contract guard passes (`python tools/check_packaging_contract.py`)
- [x] Release artifacts include `dist/SHA256SUMS` and checksum verification passes
- [x] Changelog contains the release section and date
- [x] CI workflow on `main` is green for the release commit
- [x] Tag/release notes reviewed and approved

## Post-Publish Verification

- Verify PyPI parity (same version as release tag):
  `python tools/check_pypi_sync.py --package scpn-fusion --mode equal --strip-v-prefix --retries 30 --retry-delay 10`
