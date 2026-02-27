# Claims Evidence Map

Auto-generated map from `validation/claims_manifest.json` linking headline claims
to concrete evidence files and patterns.

- Manifest: `validation/claims_manifest.json`
- Claims tracked: `2`

## Summary

| Claim ID | Source | Evidence Files | Pattern Checks |
|---|---|---:|---:|
| `readme_pretrained_coverage_claim` | `README.md` | 3 | 1 |
| `results_fno_validated_status` | `RESULTS.md` | 1 | 1 |

## Claim Details

### `readme_pretrained_coverage_claim`

- Source file: `README.md`
- Source pattern: `4 of 7 shipped`

Evidence files:
- `validation/reports/task2_pretrained_surrogates_benchmark.json`
- `validation/reports/task2_pretrained_surrogates_benchmark.md`
- `weights/neural_transport_qlknn.npz`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `validation/reports/task2_pretrained_surrogates_benchmark.json` | `"coverage_percent"\s*:\s*57\.1428` |

### `results_fno_validated_status`

- Source file: `RESULTS.md`
- Source pattern: `JAX FNO turbulence surrogate`

Evidence files:
- `src/scpn_fusion/core/fno_jax_training.py`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `src/scpn_fusion/core/fno_jax_training.py` | `JAX-FNO` |
