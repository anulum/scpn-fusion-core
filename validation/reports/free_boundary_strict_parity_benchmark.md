# Free-boundary Strict Parity Benchmark

This gate is fail-closed. It accepts full-fidelity free-boundary parity
only when same-case public FreeGS output, native profile-source
comparison, strict thresholds, grid convergence, and public external
coil/vacuum sidecars are all present.

- Schema: `free-boundary-strict-parity-benchmark.v1`
- Status: `blocked_free_boundary_strict_parity`
- Accepted full fidelity: `False`
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
| `grid_convergence_ready` | `False` |
| `coil_vacuum_sidecar_ready` | `False` |
| `machine_metadata_ready` | `True` |
| `same_case_public_reference_output_ready` | `False` |

## Acceptance matrix

| Requirement | Ready |
| --- | ---: |
| `same_case_reference_output` | `False` |
| `native_same_case_profile_source` | `True` |
| `strict_threshold_metrics` | `True` |
| `grid_convergence_ladder` | `False` |
| `coil_vacuum_sidecars` | `False` |
| `machine_metadata` | `True` |

## Blockers

- `grid_convergence_evidence_missing`
- `public_external_coil_vacuum_sidecars_missing`
- `same_case_public_reference_output_missing`

## Threshold cases

| Case | External output | Native comparison | Thresholds ready | Failed checks |
| --- | ---: | ---: | ---: | ---: |
| freegs_01_test_tokamak_freeboundary | `True` | `True` | `True` | 0 |
| freegs_16_diiid_public_example | `True` | `True` | `True` | 0 |

## Grid-convergence cases

| Case | Machine | Observed | Required | Missing | Ready | Blocker |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| freegs_01_test_tokamak_freeboundary | TestTokamak | 1 | 3 | 2 | `False` | public_example_has_single_resolution |
| freegs_16_diiid_public_example | DIIID | 1 | 3 | 2 | `False` | public_example_has_single_resolution |

## Machine metadata

- Schema: `free-boundary-public-machine-metadata-inventory-report.v1`
- Status: `blocked_machine_metadata_indexed_missing_same_case_free_boundary_reconstruction`
- Machine config count: `23`
- Machines: `ITER, MAST-U, SPARC, example, test`
