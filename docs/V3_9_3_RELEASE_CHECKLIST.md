# v3.9.3 Release Checklist (Hardening Wave)

Release objective:
- Ship hardening and credibility improvements without scope creep.
- Reduce underdeveloped P0/P1 flags from the previous baseline.

## 1) Hardening and Register Accuracy

- [ ] Regenerate `UNDERDEVELOPED_REGISTER.md` from `tools/generate_underdeveloped_register.py`.
- [ ] Confirm Executive Summary counts are current and reflected in docs by reference (not hard-coded).
- [ ] Verify no stale count literals remain in `README.md`, `ROADMAP.md`, `CONTRIBUTING.md`, `docs/HONEST_SCOPE.md`.

## 2) Validation and Claims Integrity

- [ ] Re-run release validation lane (`python tools/run_python_preflight.py --gate release`).
- [ ] Regenerate `RESULTS.md` via `validation/collect_results.py`.
- [ ] Verify updated claims are mapped in `docs/CLAIMS_EVIDENCE_MAP.md`.
- [ ] Confirm deprecated/experimental paths remain non-default in release-facing commands.

## 3) Community and Contribution Infrastructure

- [ ] Ensure issue templates cover bug, feature, validation-data, and paper-review flows.
- [ ] Ensure Discussions link is discoverable in issue config and README community section.
- [ ] Ensure contribution path for validation data references provenance + license constraints.

## 4) Packaging and CI Health

- [ ] Confirm CI release lanes pass on `main`.
- [ ] Confirm coverage artifact upload lane remains green.
- [ ] Confirm dependency update automation config is present and valid.

## 5) Versioning and Release Notes

- [ ] Finalize `CHANGELOG.md` entries under `Unreleased`.
- [ ] Tag release as `v3.9.3` with summary of hardening deltas and remaining known gaps.
- [ ] Attach benchmark/regression artifacts to the release notes.

## 6) Exit Gates

- [ ] Underdeveloped P0/P1 count reduced relative to previous published baseline.
- [ ] No unsupported headline claims in README/RESULTS/HONEST_SCOPE.
- [ ] CI status green for release lanes.
- [ ] Reproducibility commands execute cleanly in documented environment.
