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
| `wall_flux_agreement_metric_ready` | `False` |
| `lcfs_limiter_containment_measured` | `False` |
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
- `lcfs_limiter_containment_unmeasured`

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
- Source commit: `8685ac0a5013c4658ebe22a5b2493a002574dcb5`
- Python version: `3.12.3`

| Input report | Payload SHA-256 | File SHA-256 |
| --- | --- | --- |
| validation/reports/freegs_public_example_reconstruction.json | `94b1ab85c1171dd35cdd2a9cec93212a8e5e99eb3bdcc83b8aaace77357e2da6` | `9ef13c3eaf7b92d80fa6c915fd5ade208950ad43894063ddaf590801ec5743b5` |
| validation/reports/free_boundary_public_machine_metadata_inventory.json | `89dc932da5b2d2a2e48cc6179d19a1fb3ef2c320abbdae09a89ffac7c07acb48` | `e0da36d9f1016abb0dbce77d313032d2710e325d3eb922d1908a9fb582c7e7e1` |

| Evidence section | SHA-256 |
| --- | --- |
| `acceptance_contract_sha256` | `c26e39ae02eb475fccba0faf050bd7a1ff36d85d376df7dfb486065403846c0b` |
| `acceptance_matrix_sha256` | `a72958b95ddc47a7f8dfa7ad474b1fb860789e7f5998e77388b5c357a7385526` |
| `checks_sha256` | `1cd2bb491252b84b1a023d6760b01040e48890565c690fdc7c13406cada4782b` |
| `threshold_cases_sha256` | `64d0f6534eb753bc87b8a4946f8c7f5fd2a167deb4030a9bab3446b05461d61f` |
| `grid_convergence_cases_sha256` | `469eb0605bfa8ce8968d7af2a4cd2ffd04ac3b47aac3c8e58e3b7ba8825d779a` |
| `machine_metadata_sha256` | `d4fb919289c0c1e4ca885c597b34d488838a1f21967cf26bed561df4021b2d6d` |

## Input byte binding

- `validation/reports/freegs_public_example_reconstruction.json`: payload matches file `True`
- `validation/reports/free_boundary_public_machine_metadata_inventory.json`: payload matches file `True`

## Source example files

- `freegs_01_test_tokamak_freeboundary`: `data/external/full_fidelity_public_sources/repos/freegs/01-freeboundary.py`; SHA-256 `1d4805d9e0e757c6244e482ad8ec782a2e266aca1d280815103ab5c41e38f06b`; declared hash matches `True`
- `freegs_16_diiid_public_example`: `data/external/full_fidelity_public_sources/repos/freegs/16-DIIID.py`; SHA-256 `445a7415c0e24d5de2a6ed4563230070215ab49e88c061a990b4243987f463af`; declared hash matches `True`

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
