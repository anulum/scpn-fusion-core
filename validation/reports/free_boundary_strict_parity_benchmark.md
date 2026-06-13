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
