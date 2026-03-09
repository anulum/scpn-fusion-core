# CI Failure Ledger

This ledger records recurring CI failure patterns seen during the 2026-03-03 hardening waves.

## Usage
- Add one row per **new failure pattern** (do not duplicate identical root causes).
- Link at least one representative run ID and the fixing commit.
- Mark `Transient` vs `Deterministic` so triage is fast.

## Baseline Prevention Policy (2026-03-04)
- Enforce local git hooks via `git config core.hooksPath .githooks`.
- `pre-commit` blocks metadata drift (`tools/sync_metadata.py --check`).
- `pre-push` blocks pushes unless `python tools/run_python_preflight.py --gate release` passes.
- CI failures from deterministic drift/schema causes should be considered process violations and fixed locally before next push.
- Distinguish `cancelled` from `failure`: chain-guard cancellations are expected on superseded pushes and should not trigger code hotfixes.

## Incident Table

| ID | First Seen (UTC) | Representative Runs | Failure Signature | Category | Root Cause | Resolved By | Prevention Guard |
|---|---|---|---|---|---|---|---|
| CI-2026-03-03-001 | 2026-03-03 05:58 | 22610422427 | `FAILED at 'Untested module linkage guard'` | Deterministic | Split modules were missing explicit test linkage entries expected by preflight guard. | `84491bb`, `a2f1347` | Run `python tools/check_test_module_linkage.py --summary-json artifacts/untested_module_guard_summary.json` before push. |
| CI-2026-03-03-002 | 2026-03-03 06:12 | 22610745599, 22610975267 | `Reference-data provenance manifest is stale` and `FAILED at 'Reference-data provenance manifest check'` | Deterministic | Provenance manifest generation was unstable across file filtering/order/line-ending conditions during rapid hardening edits. | `66372f7` through `f133115` series | Regenerate before commit: `python tools/generate_reference_data_provenance_manifest.py` and re-run release preflight locally. |
| CI-2026-03-03-003 | 2026-03-03 07:13 | 22612403872 | `KeyError: 'mean_abs_risk_error'`, `KeyError: 'fault_count'`, `TypeError: 'float' object is not subscriptable` in disruption/anomaly tests | Deterministic | Contract mismatch between campaign report/anomaly return schema and tests expecting dict fields. | `f92521f` | Keep `tests/test_control_resilience_campaign.py`, `tests/test_disruption_toroidal_features.py`, `tests/test_gneu_02_anomaly.py` in release suite (non-optional). |
| CI-2026-03-03-004 | 2026-03-03 16:32 | 22632674358, 22640654698, 22640854683, 22640861297 | `Underdeveloped register drift detected: UNDERDEVELOPED_REGISTER.md` | Deterministic | Code/docs changed without regenerating tracked underdeveloped register artifacts. | `124b39c` | Run `python tools/generate_underdeveloped_register.py` before push and ensure clean diff. |
| CI-2026-03-03-005 | 2026-03-03 20:44 | 22641870792 | `Scope report drift detected: docs/UNDERDEVELOPED_DOCS_CLAIMS_REGISTER.md` | Deterministic | Scope reports were not regenerated after source/register updates. | `a9eb2d4` | Run `python tools/generate_underdeveloped_scope_reports.py` before push; keep preflight `--check` green locally. |
| CI-2026-03-03-006 | 2026-03-03 19:18 | 22638800884 (Python 3.9/3.10) | `ModuleNotFoundError: No module named 'tomllib'` from `tools/check_pypi_sync.py` | Deterministic | `tomllib` is Python 3.11+ only; CI matrix includes Python 3.9/3.10. | `967d23c` | Use compatibility import (`tomli` shim) in tooling scripts and keep 3.9/3.10 tests for script modules. |
| CI-2026-03-03-007 | 2026-03-03 19:17 | 22638800884 (Python 3.12 job) | `fatal: ... requested URL returned error: 500` in `actions/checkout` | Transient infra | GitHub remote fetch returned HTTP 500 repeatedly during checkout retry window. | Re-run succeeded in later run(s) | Treat as transient unless repeated across multiple runs; re-run workflow before code changes. |
| CI-2026-03-03-008 | 2026-03-03 19:21 | 22638800884 (Rust Security Audit) | `couldn't fetch advisory database ... Received HTTP status 500` | Transient infra | RustSec advisory DB fetch outage/IO error. | Re-run succeeded (example: 22642012052) | Keep audit job, allow operator rerun on network outages; do not hotfix code for infra-only failure. |
| CI-2026-03-04-001 | 2026-03-04 11:41 | 22667815814 | `KeyError: 'target_flux'` in `tests/test_coil_optimization.py` | Deterministic | Free-boundary runtime refactor invoked helper directly and bypassed `FusionKernel.optimize_coil_currents` method seam used by tests/contract hooks. | `99213c1` | Keep method-dispatch seam in runtime path and run `python -m pytest tests/test_coil_optimization.py -q` before push after free-boundary edits. |
| CI-2026-03-04-002 | 2026-03-04 11:25 | 22667290256 | CI run shown as `cancelled` (`#829`) | Expected chain behavior | Run was superseded by a newer push and terminated by CI chain guard; no failing job signature present. | N/A (no code fix) | Avoid push bursts during long-running suites; wait for current CI to complete before pushing non-urgent commits. |

## Quick Triage Rules
- If failure text includes `drift detected` or `manifest is stale`: regenerate artifacts, re-run preflight locally, then push.
- If failure is `HTTP 500` in checkout/audit fetch: classify as transient and re-run once before touching code.
- If failure is test contract `KeyError`/`TypeError`: treat as interface regression and fix code+tests together in one commit.
