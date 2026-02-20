# Claims Evidence Map

Auto-generated map from `validation/claims_manifest.json` linking headline claims
to concrete evidence files and patterns.

- Manifest: `validation/claims_manifest.json`
- Claims tracked: `5`

## Summary

| Claim ID | Source | Evidence Files | Pattern Checks |
|---|---|---:|---:|
| `readme_full_physics_transport_rmse` | `README.md` | 3 | 2 |
| `readme_neural_surrogate_rmse` | `README.md` | 2 | 2 |
| `readme_disruption_prevention_claim` | `README.md` | 2 | 2 |
| `readme_pretrained_coverage_claim` | `README.md` | 2 | 2 |
| `results_fno_validated_status` | `RESULTS.md` | 2 | 2 |

## Claim Details

### `readme_full_physics_transport_rmse`

- Source file: `README.md`
- Source pattern: `28\.6% full-physics relative RMSE`

Evidence files:
- `RESULTS.md`
- `validation/validate_transport_itpa.py`
- `validation/reference_data/itpa/hmode_confinement.csv`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `RESULTS.md` | `tau_E relative RMSE \\| 28\.6%` |
| `validation/validate_transport_itpa.py` | `"rmse_relative": round\(rmse_rel, 4\)` |

### `readme_neural_surrogate_rmse`

- Source file: `README.md`
- Source pattern: `\(13\.5% neural-surrogate fit lane\)`

Evidence files:
- `RESULTS.md`
- `validation/reports/task2_pretrained_surrogates_benchmark.json`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `RESULTS.md` | `Neural transport MLP surrogate \\| tau_E RMSE % \\| 13\.5%` |
| `validation/reports/task2_pretrained_surrogates_benchmark.json` | `"rmse_pct"\s*:\s*13\.489` |

### `readme_disruption_prevention_claim`

- Source file: `README.md`
- Source pattern: `>60% disruption prevention rate`

Evidence files:
- `RESULTS.md`
- `validation/reports/task5_disruption_mitigation_integration.json`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `RESULTS.md` | `Disruption prevention rate \(SNN\) \\| >60` |
| `validation/reports/task5_disruption_mitigation_integration.json` | `"disruption_prevention_rate"\s*:\s*1\.0` |

### `readme_pretrained_coverage_claim`

- Source file: `README.md`
- Source pattern: `3 of 7 shipped`

Evidence files:
- `validation/reports/task2_pretrained_surrogates_benchmark.json`
- `validation/reports/task2_pretrained_surrogates_benchmark.md`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `validation/reports/task2_pretrained_surrogates_benchmark.json` | `"coverage_percent"\s*:\s*28\.5714` |
| `validation/reports/task2_pretrained_surrogates_benchmark.md` | `Pretrained coverage: \`28\.6%\`` |

### `results_fno_validated_status`

- Source file: `RESULTS.md`
- Source pattern: `NEW — JAX FNO turbulence surrogate`

Evidence files:
- `src/scpn_fusion/core/fno_turbulence_suppressor.py`
- `src/scpn_fusion/core/fno_training.py`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `src/scpn_fusion/core/fno_turbulence_suppressor.py` | `JAX-FNO` |
| `src/scpn_fusion/core/fno_training.py` | `JAX-accelerated version` |
