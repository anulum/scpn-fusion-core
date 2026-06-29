# Free-boundary Strict Parity Benchmark

This gate is fail-closed. It accepts full-fidelity free-boundary parity
only when same-case public FreeGS output, native profile-source
comparison, strict thresholds, grid convergence, and public external
coil/vacuum sidecars are all present.

- Schema: `free-boundary-strict-parity-benchmark.v1`
- Status: `accepted_full_fidelity_free_boundary_parity`
- Accepted full fidelity: `True`
- Case count: `2`
- Failed threshold checks: `0`

## Checks

| Check | Ready |
| --- | ---: |
| `external_nonlinear_output_ready` | `True` |
| `native_same_case_profile_source_ready` | `True` |
| `strict_threshold_acceptance_ready` | `True` |
| `geometry_containment_ready` | `True` |
| `boundary_containment_metric_ready` | `True` |
| `grid_convergence_ready` | `True` |
| `coil_vacuum_sidecar_ready` | `True` |
| `machine_metadata_ready` | `True` |
| `same_case_public_reference_output_ready` | `True` |

## Acceptance matrix

| Requirement | Ready |
| --- | ---: |
| `same_case_reference_output` | `True` |
| `native_same_case_profile_source` | `True` |
| `strict_threshold_metrics` | `True` |
| `grid_convergence_ladder` | `True` |
| `coil_vacuum_sidecars` | `True` |
| `machine_metadata` | `True` |

## Blockers

- None

## Threshold cases

| Case | External output | Native comparison | Thresholds ready | Failed checks |
| --- | ---: | ---: | ---: | ---: |
| freegs_01_test_tokamak_freeboundary | `True` | `True` | `True` | 0 |
| freegs_16_diiid_public_example | `True` | `True` | `True` | 0 |

## Grid-convergence cases

| Case | Machine | Observed | Required | Missing | Ready | Blocker |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| freegs_01_test_tokamak_freeboundary | TestTokamak | 3 | 3 | 0 | `True` |  |
| freegs_16_diiid_public_example | DIIID | 3 | 3 | 0 | `True` |  |

## Machine metadata

- Schema: `free-boundary-public-machine-metadata-inventory-report.v1`
- Status: `accepted_public_machine_metadata_with_same_case_free_boundary_reference`
- Machine config count: `23`
- Machines: `ITER, MAST-U, SPARC, example, test`

## Provenance and checksums

- Generator: `validation/benchmark_free_boundary_strict_parity.py`
- Source commit: `a950028dc366bfc9b28e12924c168523b1c17f7c`
- Python version: `3.12.3`

| Input report | Payload SHA-256 | File SHA-256 |
| --- | --- | --- |
| validation/reports/freegs_public_example_reconstruction.json | `39b624fba02e92e0e9d5a569d7ceb425405582ea18e2527dbb5defb3690f6928` | `73edc6c0ebc1d0acc249dd8d13dc14a668e3ecd3419e336e49397b322af663ed` |
| validation/reports/free_boundary_public_machine_metadata_inventory.json | `89dc932da5b2d2a2e48cc6179d19a1fb3ef2c320abbdae09a89ffac7c07acb48` | `e0da36d9f1016abb0dbce77d313032d2710e325d3eb922d1908a9fb582c7e7e1` |

| Evidence section | SHA-256 |
| --- | --- |
| `acceptance_contract_sha256` | `7970bb92a78a3fe9cda1f0a7f057216f7081c7c10c5949dded7e1343601b484c` |
| `acceptance_matrix_sha256` | `ab744c8a30650b2ecdfbfd4070f118e794cdcc46f7439dec27fc2abd4bbd6595` |
| `checks_sha256` | `962acf255628c4000a54b62edfaa5a0af39bab6c63896fee6e0cf143b62da61f` |
| `threshold_cases_sha256` | `3d195e8a55ee4b475cbfef550282ac3c3c7a49070d0382202377f01ada051fa1` |
| `grid_convergence_cases_sha256` | `a50410531c913ad059b95621bc1d402ffdb17c9f7e3a7e78c0bac84aa5735286` |
| `machine_metadata_sha256` | `d4fb919289c0c1e4ca885c597b34d488838a1f21967cf26bed561df4021b2d6d` |
