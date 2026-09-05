# Free-boundary Strict Parity Benchmark

No positive scientific evidence contract is admitted by this classifier.
These diagnostics check stored summaries and source bytes; they do not
establish raw field/q-profile custody or independent predictive validation.

- Schema: `free-boundary-strict-parity-benchmark.v2`
- Status: `blocked_free_boundary_strict_parity`
- Evidence classification: `legacy_non_admitting`
- Accepted full fidelity: `False`
- Case count: `2`
- Failed threshold checks: `0`

## Checks

| Check | Ready |
| --- | ---: |
| `external_nonlinear_output_ready` | `False` |
| `native_same_case_profile_source_ready` | `False` |
| `strict_threshold_acceptance_ready` | `False` |
| `geometry_containment_ready` | `False` |
| `boundary_containment_metric_ready` | `False` |
| `grid_convergence_ready` | `False` |
| `coil_vacuum_sidecar_ready` | `False` |
| `machine_metadata_ready` | `False` |
| `same_case_public_reference_output_ready` | `False` |

## Acceptance matrix

| Requirement | Ready |
| --- | ---: |
| `same_case_reference_output` | `False` |
| `native_same_case_profile_source` | `False` |
| `strict_threshold_metrics` | `False` |
| `grid_convergence_ladder` | `False` |
| `coil_vacuum_sidecars` | `False` |
| `machine_metadata` | `False` |

## Blockers

- `legacy_evidence_has_no_verified_field_custody`
- `source_example_bytes_unverified`

## Threshold cases

| Case | External output | Native comparison | Thresholds ready | Failed checks |
| --- | ---: | ---: | ---: | ---: |
| freegs_01_test_tokamak_freeboundary | `False` | `False` | `False` | 0 |
| freegs_16_diiid_public_example | `False` | `False` | `False` | 0 |

## Grid-convergence cases

| Case | Machine | Observed | Required | Missing | Ready | Blocker |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| freegs_01_test_tokamak_freeboundary | TestTokamak | 3 | 3 | 0 | `False` | grid_summary_has_no_verified_field_custody |
| freegs_16_diiid_public_example | DIIID | 3 | 3 | 0 | `False` | grid_summary_has_no_verified_field_custody |

## Machine metadata

- Schema: `free-boundary-public-machine-metadata-inventory-report.v1`
- Status: `accepted_public_machine_metadata_with_same_case_free_boundary_reference`
- Machine config count: `23`
- Machines: `ITER, MAST-U, SPARC, example, test`

## Provenance and checksums

- Generator: `validation/benchmark_free_boundary_strict_parity.py`
- Source commit: `a84317b5315c4d2c4d8412b4fbfe16cf7ac6f650`
- Python version: `3.12.3`

| Input report | Payload SHA-256 | File SHA-256 |
| --- | --- | --- |
| validation/reports/freegs_public_example_reconstruction.json | `39b624fba02e92e0e9d5a569d7ceb425405582ea18e2527dbb5defb3690f6928` | `73edc6c0ebc1d0acc249dd8d13dc14a668e3ecd3419e336e49397b322af663ed` |
| validation/reports/free_boundary_public_machine_metadata_inventory.json | `89dc932da5b2d2a2e48cc6179d19a1fb3ef2c320abbdae09a89ffac7c07acb48` | `e0da36d9f1016abb0dbce77d313032d2710e325d3eb922d1908a9fb582c7e7e1` |

| Evidence section | SHA-256 |
| --- | --- |
| `acceptance_contract_sha256` | `7970bb92a78a3fe9cda1f0a7f057216f7081c7c10c5949dded7e1343601b484c` |
| `acceptance_matrix_sha256` | `a72958b95ddc47a7f8dfa7ad474b1fb860789e7f5998e77388b5c357a7385526` |
| `checks_sha256` | `ea15999ab61a8517f3e60d62e6629449deb01b926c9774be7789891b6e55904a` |
| `threshold_cases_sha256` | `64d0f6534eb753bc87b8a4946f8c7f5fd2a167deb4030a9bab3446b05461d61f` |
| `grid_convergence_cases_sha256` | `469eb0605bfa8ce8968d7af2a4cd2ffd04ac3b47aac3c8e58e3b7ba8825d779a` |
| `machine_metadata_sha256` | `d4fb919289c0c1e4ca885c597b34d488838a1f21967cf26bed561df4021b2d6d` |

## Input byte binding

- `validation/reports/freegs_public_example_reconstruction.json`: payload matches file `True`
- `validation/reports/free_boundary_public_machine_metadata_inventory.json`: payload matches file `True`

## Source example files

- `freegs_01_test_tokamak_freeboundary`: `data/external/full_fidelity_public_sources/repos/freegs/01-freeboundary.py`; SHA-256 `None`; declared hash matches `False`
- `freegs_16_diiid_public_example`: `data/external/full_fidelity_public_sources/repos/freegs/16-DIIID.py`; SHA-256 `None`; declared hash matches `False`

## Case identity errors

- No reported identity discrepancies; this is not field custody.

## Threshold diagnostics

- `freegs_01_test_tokamak_freeboundary`:
  - Unverified: `q_profile_sanity_status`.
- `freegs_16_diiid_public_example`:
  - Unverified: `q_profile_sanity_status`.

## Grid diagnostic discrepancies

- `freegs_01_test_tokamak_freeboundary`:
- `freegs_16_diiid_public_example`:
