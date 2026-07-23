# IDA free-boundary same-case evidence

- Status: `blocked_same_case_evidence`
- Schema: `scpn-fusion.ida-same-case-evidence.v1`
- Payload SHA-256: `d62bedf91bde3d616f3ad8b210a84ad281dac163dcf494fd5b409c8284e04761`
- Facility/control/PCS/safety claims: `false`

## Cases

| Case | Role | Grid | ψ span NRMSE | current relative error | warm p95 (ms) |
|---|---|---:|---:|---:|---:|
| freegs_01_test_tokamak_freeboundary | development | 65×65 | 2.63462 | 0.590941 | 2252.17 |
| freegs_16_diiid_public_example | evaluation_candidate | 129×129 | 2.08755 | 0.520574 | 2175.64 |

## Blockers

- `collaborator_solver_reference_not_bound`
- `evaluation_threshold_failed:gradient_audit`
- `evaluation_threshold_failed:latency`
- `evaluation_threshold_failed:psi_span_nrmse`
- `evaluation_threshold_failed:relative_current_error`
- `execution_preceding_selection_lock_missing`
- `facility_validation_not_bound`
- `isolated_latency_evidence_missing`
- `pcs_and_safety_programmes_not_bound`
- `statistically_held_out_case_missing`
