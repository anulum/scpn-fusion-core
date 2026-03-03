# Claims Evidence Map

Auto-generated map from `validation/claims_manifest.json` linking headline claims
to concrete evidence files and patterns.

- Manifest: `validation/claims_manifest.json`
- Claims tracked: `13`

## Summary

| Claim ID | Source | Evidence Files | Pattern Checks |
|---|---|---:|---:|
| `readme_rust_speedup_claim` | `README.md` | 1 | 2 |
| `readme_rust_control_loop_latency_claim` | `README.md` | 1 | 1 |
| `readme_qlknn_transport_claim` | `README.md` | 1 | 1 |
| `readme_zero_disruption_rust_pid_claim` | `README.md` | 1 | 1 |
| `readme_itpa_hmode_coverage_claim` | `README.md` | 1 | 1 |
| `results_qlknn_lane_pass` | `RESULTS.md` | 1 | 1 |
| `results_real_shot_overall_pass` | `RESULTS.md` | 1 | 1 |
| `results_real_shot_disruption_recall` | `RESULTS.md` | 1 | 1 |
| `results_real_shot_disruption_fpr` | `RESULTS.md` | 1 | 1 |
| `results_manufactured_source_equilibrium_pass` | `RESULTS.md` | 1 | 3 |
| `results_threshold_sweep_dataset_size` | `RESULTS.md` | 1 | 1 |
| `results_threshold_sweep_fpr` | `RESULTS.md` | 1 | 1 |
| `security_current_supported_release` | `SECURITY.md` | 2 | 2 |

## Claim Details

### `readme_rust_speedup_claim`

- Source file: `README.md`
- Source pattern: `6,600x speedup`

Evidence files:
- `validation/reports/stress_test_campaign.json`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `validation/reports/stress_test_campaign.json` | `"PID"[\s\S]*"p50_latency_us"\s*:\s*3431\.3758333333335` |
| `validation/reports/stress_test_campaign.json` | `"Rust-PID"[\s\S]*"p50_latency_us"\s*:\s*0\.522531` |

### `readme_rust_control_loop_latency_claim`

- Source file: `README.md`
- Source pattern: `Rust control-loop latency \| \*\*0\.52 us P50\*\*`

Evidence files:
- `validation/reports/stress_test_campaign.json`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `validation/reports/stress_test_campaign.json` | `"Rust-PID"[\s\S]*"p50_latency_us"\s*:\s*0\.522531` |

### `readme_qlknn_transport_claim`

- Source file: `README.md`
- Source pattern: `QLKNN-10D transport surrogate \| test rel_L2 = 0\.094`

Evidence files:
- `weights/neural_transport_qlknn.metrics.json`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `weights/neural_transport_qlknn.metrics.json` | `"test_relative_l2"\s*:\s*0\.09430918020180648` |

### `readme_zero_disruption_rust_pid_claim`

- Source file: `README.md`
- Source pattern: `Disruption rate \(1,000-shot campaign\) \| \*\*0%\*\* \(Rust-PID\)`

Evidence files:
- `validation/reports/stress_test_campaign.json`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `validation/reports/stress_test_campaign.json` | `"Rust-PID"[\s\S]*"disruption_rate"\s*:\s*0\.0` |

### `readme_itpa_hmode_coverage_claim`

- Source file: `README.md`
- Source pattern: `ITPA H-mode confinement`

Evidence files:
- `artifacts/real_shot_validation.json`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `artifacts/real_shot_validation.json` | `"transport"[\s\S]*"n_shots"\s*:\s*53` |

### `results_qlknn_lane_pass`

- Source file: `RESULTS.md`
- Source pattern: `QLKNN Transport \| PASS`

Evidence files:
- `weights/neural_transport_qlknn.metrics.json`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `weights/neural_transport_qlknn.metrics.json` | `"test_relative_l2"\s*:\s*0\.09430918020180648` |

### `results_real_shot_overall_pass`

- Source file: `RESULTS.md`
- Source pattern: `Overall real-shot pass \| Yes`

Evidence files:
- `artifacts/real_shot_validation.json`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `artifacts/real_shot_validation.json` | `"overall_pass"\s*:\s*true` |

### `results_real_shot_disruption_recall`

- Source file: `RESULTS.md`
- Source pattern: `Disruption recall \| 1\.00`

Evidence files:
- `artifacts/real_shot_validation.json`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `artifacts/real_shot_validation.json` | `"disruption"[\s\S]*"recall"\s*:\s*1\.0` |

### `results_real_shot_disruption_fpr`

- Source file: `RESULTS.md`
- Source pattern: `Disruption FPR \| 0\.00`

Evidence files:
- `artifacts/real_shot_validation.json`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `artifacts/real_shot_validation.json` | `"disruption"[\s\S]*"false_positive_rate"\s*:\s*0\.0` |

### `results_manufactured_source_equilibrium_pass`

- Source file: `RESULTS.md`
- Source pattern: `Manufactured-Source Equilibrium Parity \(Solov'ev lane\)`

Evidence files:
- `artifacts/freegs_benchmark.json`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `artifacts/freegs_benchmark.json` | `"mode"\s*:\s*"solovev_manufactured_source"` |
| `artifacts/freegs_benchmark.json` | `"overall_psi_nrmse"\s*:\s*0\.076363` |
| `artifacts/freegs_benchmark.json` | `"passes"\s*:\s*true` |

### `results_threshold_sweep_dataset_size`

- Source file: `RESULTS.md`
- Source pattern: `Shots evaluated \| 16`

Evidence files:
- `artifacts/disruption_threshold_sweep.json`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `artifacts/disruption_threshold_sweep.json` | `"n_shots"\s*:\s*16` |

### `results_threshold_sweep_fpr`

- Source file: `RESULTS.md`
- Source pattern: `\| FPR \| 0\.50`

Evidence files:
- `artifacts/disruption_threshold_sweep.json`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `artifacts/disruption_threshold_sweep.json` | `"fpr"\s*:\s*0\.5` |

### `security_current_supported_release`

- Source file: `SECURITY.md`
- Source pattern: `\| 3\.9\.3\s+\| :white_check_mark:`

Evidence files:
- `pyproject.toml`
- `src/scpn_fusion/__init__.py`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `pyproject.toml` | `version\s*=\s*"3\.9\.3"` |
| `src/scpn_fusion/__init__.py` | `__version__\s*=\s*"3\.9\.3"` |
