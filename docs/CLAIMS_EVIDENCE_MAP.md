# Claims Evidence Map

Auto-generated map from `validation/claims_manifest.json` linking headline claims
to concrete evidence files and patterns.

- Manifest: `validation/claims_manifest.json`
- Claims tracked: `18`

## Summary

| Claim ID | Source | Evidence Files | Pattern Checks |
|---|---|---:|---:|
| `readme_control_latency_scope_claim` | `README.md` | 2 | 2 |
| `readme_rust_control_loop_latency_claim` | `README.md` | 1 | 1 |
| `readme_qlknn_transport_claim` | `README.md` | 1 | 1 |
| `readme_zero_disruption_rust_pid_claim` | `README.md` | 1 | 1 |
| `readme_itpa_hmode_coverage_claim` | `README.md` | 1 | 1 |
| `results_qlknn_lane_pass` | `RESULTS.md` | 1 | 1 |
| `results_real_shot_overall_pass` | `RESULTS.md` | 1 | 1 |
| `results_real_shot_disruption_recall` | `RESULTS.md` | 1 | 1 |
| `results_real_shot_disruption_fpr` | `RESULTS.md` | 1 | 1 |
| `results_freegs_equilibrium_benchmark` | `RESULTS.md` | 1 | 3 |
| `results_threshold_sweep_dataset_size` | `RESULTS.md` | 1 | 1 |
| `results_threshold_sweep_fpr` | `RESULTS.md` | 1 | 1 |
| `security_current_supported_release` | `SECURITY.md` | 2 | 2 |
| `readme_free_boundary_scope_boundary` | `README.md` | 1 | 2 |
| `readme_native_gk_scope_boundary` | `README.md` | 1 | 2 |
| `readme_runaway_scope_boundary` | `README.md` | 1 | 2 |
| `readme_elm_proxy_scope_boundary` | `README.md` | 1 | 2 |
| `readme_impurity_scope_boundary` | `README.md` | 1 | 2 |

## Claim Details

### `readme_control_latency_scope_claim`

- Source file: `README.md`
- Source pattern: `metric-scoped and are not same-work Rust-versus-Python physics speedups`

Evidence files:
- `docs/PERFORMANCE_METRIC_TAXONOMY.md`
- `validation/reports/scpn_end_to_end_latency.md`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `docs/PERFORMANCE_METRIC_TAXONOMY.md` | `mixed-fidelity throughput ratio, not a\s+same-work Rust-versus-Python speedup` |
| `validation/reports/scpn_end_to_end_latency.md` | `Overall pass: \`YES\`` |

### `readme_rust_control_loop_latency_claim`

- Source file: `README.md`
- Source pattern: `Rust PID kernel latency \| \*\*0\.52 us P50\*\*`

Evidence files:
- `validation/reports/stress_test_campaign.json`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `validation/reports/stress_test_campaign.json` | `"Rust-PID"[\s\S]*"p50_latency_us"\s*:\s*[0-9eE+\-.]+` |

### `readme_qlknn_transport_claim`

- Source file: `README.md`
- Source pattern: `QLKNN-10D transport surrogate \| test rel_L2 = \*\*0\.094\*\*`

Evidence files:
- `weights/neural_transport_qlknn.metrics.json`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `weights/neural_transport_qlknn.metrics.json` | `"test_relative_l2"\s*:\s*[0-9eE+\-.]+` |

### `readme_zero_disruption_rust_pid_claim`

- Source file: `README.md`
- Source pattern: `Disruption rate \(1,000-shot sim campaign\) \| \*\*0%\*\* \(Rust-PID\)`

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
| `weights/neural_transport_qlknn.metrics.json` | `"test_relative_l2"\s*:\s*[0-9eE+\-.]+` |

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

### `results_freegs_equilibrium_benchmark`

- Source file: `RESULTS.md`
- Source pattern: `Equilibrium Parity`

Evidence files:
- `artifacts/freegs_benchmark.json`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `artifacts/freegs_benchmark.json` | `"mode"\s*:\s*"(freegs\|solovev_manufactured_source)"` |
| `artifacts/freegs_benchmark.json` | `"overall_psi_nrmse"\s*:\s*[0-9eE+\-.]+` |
| `artifacts/freegs_benchmark.json` | `"passes"\s*:\s*(true\|false)` |

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
- Source pattern: `\| 3\.9\.5\s+\| :white_check_mark:`

Evidence files:
- `pyproject.toml`
- `src/scpn_fusion/__init__.py`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `pyproject.toml` | `version\s*=\s*"3\.9\.5"` |
| `src/scpn_fusion/__init__.py` | `__version__\s*=\s*"3\.9\.5"` |

### `readme_free_boundary_scope_boundary`

- Source file: `README.md`
- Source pattern: `Free-boundary GS solve \| Public GEQDSK gate passes; FreeGS strict backend opt-in; not EFIT-grade inverse reconstruction`

Evidence files:
- `docs/HONEST_SCOPE.md`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `docs/HONEST_SCOPE.md` | `Public SPARC GEQDSK validation gates` |
| `docs/HONEST_SCOPE.md` | `Full EFIT/LiUQE-quality profile and boundary reconstruction` |

### `readme_native_gk_scope_boundary`

- Source file: `README.md`
- Source pattern: `Native GK solver \| Linear eigenvalue plus nonlinear 5D operator/invariant benchmarks; not GENE/CGYRO-class production turbulence`

Evidence files:
- `docs/HONEST_SCOPE.md`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `docs/HONEST_SCOPE.md` | `Linear eigenvalue solver plus bounded nonlinear 5D operator and invariant benchmarks` |
| `docs/HONEST_SCOPE.md` | `Production nonlinear turbulence replacement for full nonlinear gyrokinetic \(GENE, GS2, CGYRO solve 5D Vlasov-Maxwell\) engines` |

### `readme_runaway_scope_boundary`

- Source file: `README.md`
- Source pattern: `Runaway electron dynamics \| DREAM-style fluid balance and 1D momentum Fokker-Planck contracts; no multidimensional DREAM kinetic-distribution parity`

Evidence files:
- `docs/PHYSICS_METHODS_COMPLETE.md`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `docs/PHYSICS_METHODS_COMPLETE.md` | `DREAM-style fluid density balance` |
| `docs/PHYSICS_METHODS_COMPLETE.md` | `does not\s+claim parity with DREAM's kinetic momentum-space distribution solver` |

### `readme_elm_proxy_scope_boundary`

- Source file: `README.md`
- Source pattern: `ELM model \+ RMP suppression \| Peeling-ballooning proxy; no nonlinear MHD ELM simulation`

Evidence files:
- `docs/HONEST_SCOPE.md`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `docs/HONEST_SCOPE.md` | `Peeling-ballooning stability proxy with crash operator` |
| `docs/HONEST_SCOPE.md` | `Nonlinear MHD ELM simulation \(JOREK, BOUT\+\+\)` |

### `readme_impurity_scope_boundary`

- Source file: `README.md`
- Source pattern: `Impurity transport \(neoclassical\) \| Trace radial transport with source conservation and neoclassical pinch contracts; no STRAHL/JINTRAC collisional-operator parity`

Evidence files:
- `docs/PHYSICS_METHODS_COMPLETE.md`

Evidence pattern checks:

| File | Pattern |
|---|---|
| `docs/PHYSICS_METHODS_COMPLETE.md` | `Trace radial impurity transport` |
| `docs/PHYSICS_METHODS_COMPLETE.md` | `does not claim STRAHL/JINTRAC collisional-operator parity` |
