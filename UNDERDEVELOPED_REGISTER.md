# Underdeveloped Register

- Generated at: `2026-02-28T23:47:01.795196+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: production code + docs claims markers (tests/reports/html excluded)

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 297 |
| P0 + P1 entries | 92 |
| Domains affected | 8 |

## Marker Distribution

| Key | Count |
|---|---:|
| `FALLBACK` | 172 |
| `EXPERIMENTAL` | 55 |
| `SIMPLIFIED` | 27 |
| `PLANNED` | 22 |
| `DEPRECATED` | 17 |
| `NOT_VALIDATED` | 4 |

## Domain Distribution

| Key | Count |
|---|---:|
| `docs_claims` | 139 |
| `core_physics` | 44 |
| `other` | 36 |
| `validation` | 35 |
| `control` | 28 |
| `compiler_runtime` | 8 |
| `nuclear` | 4 |
| `diagnostics_io` | 3 |

## Top Priority Backlog (Top 80)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 104 | `core_physics` | `DEPRECATED` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:8` | Core Physics WG | Replace default path or remove lane before next major release. | """DEPRECATED: FNO turbulence surrogate (synthetic-only, rel_L2=0.79). |
| P0 | 104 | `core_physics` | `DEPRECATED` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:21` | Core Physics WG | Replace default path or remove lane before next major release. | "fno_turbulence_suppressor is deprecated (rel_L2=0.79, synthetic-only). " |
| P0 | 101 | `validation` | `DEPRECATED` | `validation/benchmark_deprecated_mode_exclusion.py:3` | Validation WG | Replace default path or remove lane before next major release. | # SCPN Fusion Core -- Deprecated Mode Exclusion Benchmark |
| P0 | 101 | `validation` | `DEPRECATED` | `validation/benchmark_deprecated_mode_exclusion.py:5` | Validation WG | Replace default path or remove lane before next major release. | """Benchmark to ensure deprecated FNO lanes never leak into default runtime.""" |
| P0 | 101 | `validation` | `DEPRECATED` | `validation/benchmark_deprecated_mode_exclusion.py:106` | Validation WG | Replace default path or remove lane before next major release. | "# Deprecated Mode Exclusion Benchmark", |
| P0 | 101 | `validation` | `DEPRECATED` | `validation/benchmark_deprecated_mode_exclusion.py:154` | Validation WG | Replace default path or remove lane before next major release. | print("Deprecated mode exclusion benchmark complete.") |
| P0 | 101 | `validation` | `DEPRECATED` | `validation/collect_results.py:531` | Validation WG | Replace default path or remove lane before next major release. | rows.append(f"| JAX FNO turbulence surrogate relative L2 (mean) | {_fmt(surrogates['fno_rel_l2_mean'], '.4f')} | — | DEPRECATED — synthet... |
| P0 | 101 | `validation` | `DEPRECATED` | `validation/collect_results.py:532` | Validation WG | Replace default path or remove lane before next major release. | rows.append(f"| JAX FNO turbulence surrogate relative L2 (P95) | {_fmt(surrogates['fno_rel_l2_p95'], '.4f')} | — | DEPRECATED — use QLKNN... |
| P0 | 101 | `validation` | `DEPRECATED` | `validation/collect_results.py:590` | Validation WG | Replace default path or remove lane before next major release. | sections.append(f"\| FNO EUROfusion \| DEPRECATED \| {fno_metric} \|") |
| P0 | 95 | `core_physics` | `NOT_VALIDATED` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:10` | Core Physics WG | Add real-data validation campaign and publish error bars. | Trained on 60 Hasegawa-Wakatani samples; not validated against gyrokinetics. |
| P0 | 95 | `other` | `DEPRECATED` | `CHANGELOG.md:33` | Architecture WG | Replace default path or remove lane before next major release. | - Runtime hardening: deprecated FNO suppressor now logs missing-weight fallback via standard `logging` without raising when optional weig... |
| P0 | 95 | `other` | `DEPRECATED` | `CHANGELOG.md:46` | Architecture WG | Replace default path or remove lane before next major release. | - Regression test for deprecated FNO-controller missing-weight fallback path. |
| P0 | 95 | `other` | `DEPRECATED` | `CHANGELOG.md:370` | Architecture WG | Replace default path or remove lane before next major release. | #### FNO Turbulence Surrogate Deprecated (Task 3.3) |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:245` | Validation WG | Gate behind explicit flag and define validation exit criteria. | "Experimental-only pytest suite", |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:246` | Validation WG | Gate behind explicit flag and define validation exit criteria. | [sys.executable, "-m", "pytest", "tests/", "-q", "-m", "experimental"], |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:341` | Validation WG | Gate behind explicit flag and define validation exit criteria. | "'release' excludes experimental-only lanes, " |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:342` | Validation WG | Gate behind explicit flag and define validation exit criteria. | "'research' runs experimental-only lanes, " |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:450` | Validation WG | Gate behind explicit flag and define validation exit criteria. | help="Skip pytest experimental-only lane (tests/ -m experimental).", |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `validation/benchmark_deprecated_mode_exclusion.py:42` | Validation WG | Gate behind explicit flag and define validation exit criteria. | return "pytestmark = pytest.mark.experimental" in text |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `validation/benchmark_deprecated_mode_exclusion.py:123` | Validation WG | Gate behind explicit flag and define validation exit criteria. | "\| Research file \| Experimental marker \|", |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `validation/full_validation_pipeline.py:12` | Validation WG | Gate behind explicit flag and define validation exit criteria. | - MPC/RL controller metrics vs experimental-profile proxies |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `validation/reference_data/README.md:3` | Validation WG | Gate behind explicit flag and define validation exit criteria. | Experimental and design reference data for cross-validating SCPN Fusion Core |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `validation/run_experimental_validation.py:2` | Validation WG | Gate behind explicit flag and define validation exit criteria. | # SCPN Fusion Core — Unified Experimental Validation Runner |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `validation/run_experimental_validation.py:190` | Validation WG | Gate behind explicit flag and define validation exit criteria. | print(" SCPN Fusion Core — Unified Experimental Validation") |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `validation/validate_against_sparc.py:185` | Validation WG | Gate behind explicit flag and define validation exit criteria. | print(" SCPN Fusion Core - Validation Against Experimental Data") |
| P1 | 89 | `docs_claims` | `DEPRECATED` | `RESULTS.md:145` | Docs WG | Replace default path or remove lane before next major release. | \| JAX FNO turbulence surrogate relative L2 (mean) \| 0.7925 \| — \| DEPRECATED — synthetic-only, removal in v4.0 \| |
| P1 | 89 | `docs_claims` | `DEPRECATED` | `RESULTS.md:146` | Docs WG | Replace default path or remove lane before next major release. | \| JAX FNO turbulence surrogate relative L2 (P95) \| 0.7933 \| — \| DEPRECATED — use QLKNN-10D instead \| |
| P1 | 89 | `docs_claims` | `DEPRECATED` | `RESULTS.md:162` | Docs WG | Replace default path or remove lane before next major release. | \| FNO EUROfusion \| DEPRECATED \| rel_L2 = 0.7925 (synthetic-only, removal in v4.0) \| |
| P1 | 89 | `docs_claims` | `DEPRECATED` | `docs/HONEST_SCOPE.md:37` | Docs WG | Replace default path or remove lane before next major release. | \| FNO turbulence \| Synthetic-data trained; **not validated against gyrokinetics** \| Proxy mapping only; DEPRECATED in v3.9 \| |
| P1 | 89 | `docs_claims` | `DEPRECATED` | `docs/sphinx/changelog.rst:27` | Docs WG | Replace default path or remove lane before next major release. | - FNO turbulence surrogate deprecated (runtime ``FutureWarning``) |
| P1 | 88 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:192` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | - Added research-only pytest marker contract (`@pytest.mark.experimental`) |
| P1 | 88 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:193` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | - Added CI split lane `python-research-gate` and release-only pytest execution (`-m "not experimental"`) |
| P1 | 88 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:274` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | - Added single-command suite execution via `scpn-fusion all --surrogate --experimental`. |
| P1 | 88 | `other` | `EXPERIMENTAL` | `VALIDATION.md:106` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | - Real DIII-D experimental data are not bundled in this specific `disruption_shots` folder. |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:75` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | # Experimental modes (opt-in). |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:76` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | "quantum": ModeSpec("scpn_fusion.core.quantum_bridge", "experimental", "Quantum bridge"), |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:77` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | "q-control": ModeSpec("scpn_fusion.core.quantum_bridge", "experimental", "Quantum control bridge"), |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:80` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | "experimental", |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:83` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | "lazarus": ModeSpec("scpn_fusion.core.lazarus_bridge", "experimental", "Lazarus bridge"), |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:84` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | "director": ModeSpec("scpn_fusion.control.director_interface", "experimental", "Director interface"), |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:85` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | "vibrana": ModeSpec("scpn_fusion.core.vibrana_bridge", "experimental", "Vibrana bridge"), |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:121` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | elif spec.maturity == "experimental" and include_experimental: |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:135` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | if spec.maturity == "experimental" and not include_experimental: |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:136` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | return "experimental mode locked; pass --experimental or set SCPN_EXPERIMENTAL=1" |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:256` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | @click.option("--experimental", is_flag=True, help="Unlock experimental modes.") |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:280` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | experimental: bool, |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:295` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | include_experimental = experimental or _env_enabled("SCPN_EXPERIMENTAL") |
| P1 | 84 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/fokker_planck_re.py:123` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | Simplified from Rosenbluth-Putvinski, Nucl. Fusion 37 (1997). |
| P1 | 84 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/spi_ablation.py:15` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | (Simplified scaling) |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/eped_pedestal.py:9` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | EPED-like simplified pedestal scaling model for H-mode tokamak plasmas. |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/eped_pedestal.py:104` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | """Simplified EPED-like pedestal model. |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:154` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Divergence projection (Simplified for scalar proxy) |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/fusion_ignition_sim.py:52` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # In our simplified kernel, J was modeled directly. |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/fusion_ignition_sim.py:423` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Impurity line radiation (simplified) |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/global_design_scanner.py:68` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Actually simplified scaling: beta_N_eff = beta_N_nominal * (1 + 0.2*(kappa-1.5)) |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_transport_solver.py:766` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Z_eff = (n_e * 1 + n_imp * Z^2) / (n_e + n_imp * Z) ≈ simplified |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_transport_solver.py:797` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | L34 = -0.1 * L31 # often stabilizing/negative in simplified models |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/jax_transport_solver.py:37` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified for 1.5D: d/drho (D * dT/drho) |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neural_transport.py:128` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | _CRIT_TEM = 5.0 # R/L_Te threshold for TEM (simplified) |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/rf_heating.py:89` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified: Gaussian blob |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/sandpile_fusion_reactor.py:95` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | A simplified Reinforcement Learning agent. |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/stability_mhd.py:375` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | r"""Simplified neoclassical tearing mode (NTM) seeding analysis. |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/stability_mhd.py:377` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | The modified Rutherford equation for island width *w* is (simplified): |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `README.md:18` | Docs WG | Gate behind explicit flag and define validation exit criteria. | > This repo is the full physics + experimental suite. |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `README.md:255` | Docs WG | Gate behind explicit flag and define validation exit criteria. | ### Experimental |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/BENCHMARKS.md:42` | Docs WG | Gate behind explicit flag and define validation exit criteria. | \| Experimental validation \| SPARC, ITPA, JET \| DIII-D \| ITER, DEMO \| JET \| |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/BENCHMARKS.md:327` | Docs WG | Gate behind explicit flag and define validation exit criteria. | scpn-fusion all --surrogate --experimental |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:102` | Docs WG | Gate behind explicit flag and define validation exit criteria. | - Training on real experimental data via `train_from_geqdsk()` with profile perturbations, not synthetic data only |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:343` | Docs WG | Gate behind explicit flag and define validation exit criteria. | ## 6. Experimental Validation Portfolio |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:478` | Docs WG | Gate behind explicit flag and define validation exit criteria. | - **HSX** (University of Wisconsin): 4 field periods, QHS configuration, US-based experimental access |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:546` | Docs WG | Gate behind explicit flag and define validation exit criteria. | \| Q4 \| DIII-D and JET GEQDSK validation (5+5 shots) \| Extended validation database \| Axis error < 10 mm on experimental shots \| |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:554` | Docs WG | Gate behind explicit flag and define validation exit criteria. | - Experimental data access (DIII-D, JET): $50K |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:611` | Docs WG | Gate behind explicit flag and define validation exit criteria. | \| **Experimental Access** \| \| \| \| \| |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:634` | Docs WG | Gate behind explicit flag and define validation exit criteria. | **Experimental Access:** DIII-D and JET GEQDSK data are needed for expanding the validation database beyond the current 8 SPARC files. Co... |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:788` | Docs WG | Gate behind explicit flag and define validation exit criteria. | 2. **General Atomics** — DIII-D experimental data access, validation collaboration |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:2534` | Docs WG | Gate behind explicit flag and define validation exit criteria. | Each paper would require additional experimental results (plant model integration, hardware benchmarks, comparison against conventional c... |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/SOLVER_TUNING_GUIDE.md:74` | Docs WG | Gate behind explicit flag and define validation exit criteria. | 4. For real experimental data, start with `0.01` and adjust based on |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/SOLVER_TUNING_GUIDE.md:98` | Docs WG | Gate behind explicit flag and define validation exit criteria. | \| **0.05–0.5** \| Standard robust range \| Real experimental data with occasional probe failures or calibration drift \| |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:8` | Docs WG | Gate behind explicit flag and define validation exit criteria. | - Preserve visibility on experimental/research lanes without letting them silently contaminate release acceptance. |
| P1 | 82 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:15` | Docs WG | Gate behind explicit flag and define validation exit criteria. | | `release` | Version/claims integrity, disruption data provenance + split + calibration holdout checks, disruption replay pipeline contr... |

## Full Register (Top 250)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `core_physics` | `DEPRECATED` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:8` | """DEPRECATED: FNO turbulence surrogate (synthetic-only, rel_L2=0.79). |
| P0 | `core_physics` | `DEPRECATED` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:21` | "fno_turbulence_suppressor is deprecated (rel_L2=0.79, synthetic-only). " |
| P0 | `validation` | `DEPRECATED` | `validation/benchmark_deprecated_mode_exclusion.py:3` | # SCPN Fusion Core -- Deprecated Mode Exclusion Benchmark |
| P0 | `validation` | `DEPRECATED` | `validation/benchmark_deprecated_mode_exclusion.py:5` | """Benchmark to ensure deprecated FNO lanes never leak into default runtime.""" |
| P0 | `validation` | `DEPRECATED` | `validation/benchmark_deprecated_mode_exclusion.py:106` | "# Deprecated Mode Exclusion Benchmark", |
| P0 | `validation` | `DEPRECATED` | `validation/benchmark_deprecated_mode_exclusion.py:154` | print("Deprecated mode exclusion benchmark complete.") |
| P0 | `validation` | `DEPRECATED` | `validation/collect_results.py:531` | rows.append(f"| JAX FNO turbulence surrogate relative L2 (mean) | {_fmt(surrogates['fno_rel_l2_mean'], '.4f')} | — | DEPRECATED — synthet... |
| P0 | `validation` | `DEPRECATED` | `validation/collect_results.py:532` | rows.append(f"| JAX FNO turbulence surrogate relative L2 (P95) | {_fmt(surrogates['fno_rel_l2_p95'], '.4f')} | — | DEPRECATED — use QLKNN... |
| P0 | `validation` | `DEPRECATED` | `validation/collect_results.py:590` | sections.append(f"\| FNO EUROfusion \| DEPRECATED \| {fno_metric} \|") |
| P0 | `core_physics` | `NOT_VALIDATED` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:10` | Trained on 60 Hasegawa-Wakatani samples; not validated against gyrokinetics. |
| P0 | `other` | `DEPRECATED` | `CHANGELOG.md:33` | - Runtime hardening: deprecated FNO suppressor now logs missing-weight fallback via standard `logging` without raising when optional weig... |
| P0 | `other` | `DEPRECATED` | `CHANGELOG.md:46` | - Regression test for deprecated FNO-controller missing-weight fallback path. |
| P0 | `other` | `DEPRECATED` | `CHANGELOG.md:370` | #### FNO Turbulence Surrogate Deprecated (Task 3.3) |
| P1 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:245` | "Experimental-only pytest suite", |
| P1 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:246` | [sys.executable, "-m", "pytest", "tests/", "-q", "-m", "experimental"], |
| P1 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:341` | "'release' excludes experimental-only lanes, " |
| P1 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:342` | "'research' runs experimental-only lanes, " |
| P1 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:450` | help="Skip pytest experimental-only lane (tests/ -m experimental).", |
| P1 | `validation` | `EXPERIMENTAL` | `validation/benchmark_deprecated_mode_exclusion.py:42` | return "pytestmark = pytest.mark.experimental" in text |
| P1 | `validation` | `EXPERIMENTAL` | `validation/benchmark_deprecated_mode_exclusion.py:123` | "\| Research file \| Experimental marker \|", |
| P1 | `validation` | `EXPERIMENTAL` | `validation/full_validation_pipeline.py:12` | - MPC/RL controller metrics vs experimental-profile proxies |
| P1 | `validation` | `EXPERIMENTAL` | `validation/reference_data/README.md:3` | Experimental and design reference data for cross-validating SCPN Fusion Core |
| P1 | `validation` | `EXPERIMENTAL` | `validation/run_experimental_validation.py:2` | # SCPN Fusion Core — Unified Experimental Validation Runner |
| P1 | `validation` | `EXPERIMENTAL` | `validation/run_experimental_validation.py:190` | print(" SCPN Fusion Core — Unified Experimental Validation") |
| P1 | `validation` | `EXPERIMENTAL` | `validation/validate_against_sparc.py:185` | print(" SCPN Fusion Core - Validation Against Experimental Data") |
| P1 | `docs_claims` | `DEPRECATED` | `RESULTS.md:145` | \| JAX FNO turbulence surrogate relative L2 (mean) \| 0.7925 \| — \| DEPRECATED — synthetic-only, removal in v4.0 \| |
| P1 | `docs_claims` | `DEPRECATED` | `RESULTS.md:146` | \| JAX FNO turbulence surrogate relative L2 (P95) \| 0.7933 \| — \| DEPRECATED — use QLKNN-10D instead \| |
| P1 | `docs_claims` | `DEPRECATED` | `RESULTS.md:162` | \| FNO EUROfusion \| DEPRECATED \| rel_L2 = 0.7925 (synthetic-only, removal in v4.0) \| |
| P1 | `docs_claims` | `DEPRECATED` | `docs/HONEST_SCOPE.md:37` | \| FNO turbulence \| Synthetic-data trained; **not validated against gyrokinetics** \| Proxy mapping only; DEPRECATED in v3.9 \| |
| P1 | `docs_claims` | `DEPRECATED` | `docs/sphinx/changelog.rst:27` | - FNO turbulence surrogate deprecated (runtime ``FutureWarning``) |
| P1 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:192` | - Added research-only pytest marker contract (`@pytest.mark.experimental`) |
| P1 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:193` | - Added CI split lane `python-research-gate` and release-only pytest execution (`-m "not experimental"`) |
| P1 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:274` | - Added single-command suite execution via `scpn-fusion all --surrogate --experimental`. |
| P1 | `other` | `EXPERIMENTAL` | `VALIDATION.md:106` | - Real DIII-D experimental data are not bundled in this specific `disruption_shots` folder. |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:75` | # Experimental modes (opt-in). |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:76` | "quantum": ModeSpec("scpn_fusion.core.quantum_bridge", "experimental", "Quantum bridge"), |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:77` | "q-control": ModeSpec("scpn_fusion.core.quantum_bridge", "experimental", "Quantum control bridge"), |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:80` | "experimental", |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:83` | "lazarus": ModeSpec("scpn_fusion.core.lazarus_bridge", "experimental", "Lazarus bridge"), |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:84` | "director": ModeSpec("scpn_fusion.control.director_interface", "experimental", "Director interface"), |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:85` | "vibrana": ModeSpec("scpn_fusion.core.vibrana_bridge", "experimental", "Vibrana bridge"), |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:121` | elif spec.maturity == "experimental" and include_experimental: |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:135` | if spec.maturity == "experimental" and not include_experimental: |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:136` | return "experimental mode locked; pass --experimental or set SCPN_EXPERIMENTAL=1" |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:256` | @click.option("--experimental", is_flag=True, help="Unlock experimental modes.") |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:280` | experimental: bool, |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:295` | include_experimental = experimental or _env_enabled("SCPN_EXPERIMENTAL") |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/fokker_planck_re.py:123` | Simplified from Rosenbluth-Putvinski, Nucl. Fusion 37 (1997). |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/spi_ablation.py:15` | (Simplified scaling) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/eped_pedestal.py:9` | EPED-like simplified pedestal scaling model for H-mode tokamak plasmas. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/eped_pedestal.py:104` | """Simplified EPED-like pedestal model. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:154` | # Divergence projection (Simplified for scalar proxy) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/fusion_ignition_sim.py:52` | # In our simplified kernel, J was modeled directly. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/fusion_ignition_sim.py:423` | # Impurity line radiation (simplified) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/global_design_scanner.py:68` | # Actually simplified scaling: beta_N_eff = beta_N_nominal * (1 + 0.2*(kappa-1.5)) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_transport_solver.py:766` | # Z_eff = (n_e * 1 + n_imp * Z^2) / (n_e + n_imp * Z) ≈ simplified |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_transport_solver.py:797` | L34 = -0.1 * L31 # often stabilizing/negative in simplified models |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/jax_transport_solver.py:37` | # Simplified for 1.5D: d/drho (D * dT/drho) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neural_transport.py:128` | _CRIT_TEM = 5.0 # R/L_Te threshold for TEM (simplified) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/rf_heating.py:89` | # Simplified: Gaussian blob |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/sandpile_fusion_reactor.py:95` | A simplified Reinforcement Learning agent. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/stability_mhd.py:375` | r"""Simplified neoclassical tearing mode (NTM) seeding analysis. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/stability_mhd.py:377` | The modified Rutherford equation for island width *w* is (simplified): |
| P1 | `docs_claims` | `EXPERIMENTAL` | `README.md:18` | > This repo is the full physics + experimental suite. |
| P1 | `docs_claims` | `EXPERIMENTAL` | `README.md:255` | ### Experimental |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/BENCHMARKS.md:42` | \| Experimental validation \| SPARC, ITPA, JET \| DIII-D \| ITER, DEMO \| JET \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/BENCHMARKS.md:327` | scpn-fusion all --surrogate --experimental |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:102` | - Training on real experimental data via `train_from_geqdsk()` with profile perturbations, not synthetic data only |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:343` | ## 6. Experimental Validation Portfolio |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:478` | - **HSX** (University of Wisconsin): 4 field periods, QHS configuration, US-based experimental access |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:546` | \| Q4 \| DIII-D and JET GEQDSK validation (5+5 shots) \| Extended validation database \| Axis error < 10 mm on experimental shots \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:554` | - Experimental data access (DIII-D, JET): $50K |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:611` | \| **Experimental Access** \| \| \| \| \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:634` | **Experimental Access:** DIII-D and JET GEQDSK data are needed for expanding the validation database beyond the current 8 SPARC files. Co... |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:788` | 2. **General Atomics** — DIII-D experimental data access, validation collaboration |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:2534` | Each paper would require additional experimental results (plant model integration, hardware benchmarks, comparison against conventional c... |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/SOLVER_TUNING_GUIDE.md:74` | 4. For real experimental data, start with `0.01` and adjust based on |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/SOLVER_TUNING_GUIDE.md:98` | \| **0.05–0.5** \| Standard robust range \| Real experimental data with occasional probe failures or calibration drift \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:8` | - Preserve visibility on experimental/research lanes without letting them silently contaminate release acceptance. |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:15` | | `release` | Version/claims integrity, disruption data provenance + split + calibration holdout checks, disruption replay pipeline contr... |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:16` | \| `research` \| Experimental-only pytest lane (`pytest -m experimental`). \| `python tools/run_python_preflight.py --gate research` \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:24` | \| `python-research-gate` \| Experimental validation lane (3.12) \| `research` \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:25` | \| `validation-regression` \| Cross-language physics validation lane \| `release` (`pytest -m "not experimental"`) \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:27` | ## Experimental Marker Contract |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:29` | Tests marked with `@pytest.mark.experimental` are considered research-only and are excluded from release acceptance runs. As of v3.9.x th... |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:42` | 1. Remove `@pytest.mark.experimental` from the test module(s). |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/competitive_analysis.md:79` | \| Experimental validation \| SPARC, ITPA, JET \| DIII-D \| ITER, DEMO \| JET \| DIII-D \| ITER \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/api/core.rst:225` | Experimental Bridges |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/userguide/validation.rst:5` | SCPN-Fusion-Core is validated against published experimental data from |
| P1 | `nuclear` | `SIMPLIFIED` | `src/scpn_fusion/nuclear/blanket_neutronics.py:75` | # Cross Sections (Macroscopic Sigma in cm^-1) - Simplified for 14 MeV neutrons |
| P1 | `nuclear` | `SIMPLIFIED` | `src/scpn_fusion/nuclear/nuclear_wall_interaction.py:256` | # Material Parameters (Simplified for D on Target) |
| P1 | `nuclear` | `SIMPLIFIED` | `src/scpn_fusion/nuclear/nuclear_wall_interaction.py:271` | # f(alpha) ~ 1 / cos(alpha) -> simplified enhancement |
| P2 | `docs_claims` | `NOT_VALIDATED` | `docs/HONEST_SCOPE.md:37` | \| FNO turbulence \| Synthetic-data trained; **not validated against gyrokinetics** \| Proxy mapping only; DEPRECATED in v3.9 \| |
| P2 | `docs_claims` | `NOT_VALIDATED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:1731` | 3. **Schema validation.** The JSON Schema (`scpnctl.schema.json`) is not validated against the artifact files in the test suite. Schema v... |
| P2 | `docs_claims` | `NOT_VALIDATED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:3084` | 5. **No schema validation.** The JSON Schema exists but is not validated at load time. This means a malformed artifact that satisfies the... |
| P2 | `validation` | `SIMPLIFIED` | `tools/train_fno_qlknn_spatial.py:120` | # Simplified: spectral conv in truncated mode space |
| P2 | `validation` | `SIMPLIFIED` | `validation/stress_test_campaign.py:375` | implements a 10 kHz PID control loop with simplified linear plasma |
| P2 | `validation` | `SIMPLIFIED` | `validation/test_full_pulse_scenario.py:62` | # Step Physics (Single step of simulate logic simplified) |
| P2 | `validation` | `SIMPLIFIED` | `validation/validate_iter.py:59` | # Our model is simplified (L-mode profiles mostly), so if we get > 100MW it's a "physics pass" |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/analytic_solver.py:208` | fallback = repo_root / "validation" / "iter_validated_config.json" |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/analytic_solver.py:209` | config_path = str(preferred if preferred.exists() else fallback) |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/director_interface.py:45` | """Deterministic fallback director when external DIRECTOR_AI is unavailable.""" |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:729` | "fallback": True, |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:758` | "fallback": False, |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:767` | "fallback": True, |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:777` | "fallback": True, |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:793` | "fallback": True, |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:799` | info["fallback"] = False |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:813` | Predict disruption risk with checkpoint path if available, else deterministic fallback. |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:819` | ``metadata`` includes whether fallback mode was used. |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:822` | failures instead of returning fallback risk from ``predict_disruption_risk``. |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:839` | "predict_disruption_risk_safe fallback disabled: " |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:843` | out_meta["mode"] = "fallback" |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:874` | "predict_disruption_risk_safe fallback disabled: " |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:878` | out_meta["mode"] = "fallback" |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/fusion_control_room.py:391` | logger.warning("Kernel init failed, fallback to analytic Psi: %s", kernel_error) |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/fusion_control_room.py:423` | "Kernel coil update failed; continuing with fallback controls: %s", |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/fusion_nmpc_jax.py:12` | and minimises tracking cost via gradient descent (JAX) or L-BFGS-B (NumPy fallback). |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/fusion_optimal_control.py:122` | # Fallback: infer from grid arrays when dimensions not in config |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/h_infinity_controller.py:166` | self._Fd: np.ndarray = self.F # fallback: continuous gain |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/jax_traceable_runtime.py:4` | """Optional JAX-traceable control-loop utilities with NumPy fallback.""" |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/nengo_snn_wrapper.py:35` | # Lazy Nengo import — graceful fallback |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/nengo_snn_wrapper.py:385` | # ── Fallback for when Nengo is not available ───────────────────────── |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/neuro_cybernetic_controller.py:58` | Push-pull spiking control population with deterministic fallback. |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/neuro_cybernetic_controller.py:151` | # Reduced threshold keeps fallback lane responsive in low-current control |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_rust_compat.py:276` | Seed used by deterministic NumPy fallback backend. |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_rust_compat.py:346` | Seed used by deterministic NumPy fallback backend. |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_rust_compat.py:399` | """Deterministic local fallback matching the Rust SNN pool interface.""" |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_rust_compat.py:459` | """Deterministic local fallback matching the Rust SNN controller interface.""" |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_rust_compat.py:500` | "Use Python multigrid fallback in FusionKernel._multigrid_vcycle()." |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_rust_compat.py:509` | "Use Python multigrid fallback." |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/fusion_kernel.py:198` | "HPC Acceleration UNAVAILABLE (using Python fallback)." |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/fusion_kernel.py:794` | """Run the inner elliptic solve (HPC or Python fallback). |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/fusion_kernel.py:1644` | fallback = np.asarray(coils.currents, dtype=np.float64).copy() |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/fusion_kernel.py:1645` | return np.clip(fallback, lb, ub).astype(np.float64) |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/geometry_3d.py:122` | """Build a conservative elliptical boundary fallback inside the domain.""" |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/gpu_runtime.py:114` | raise RuntimeError("PyTorch fallback requested but torch is not installed.") |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/gpu_runtime.py:300` | "PyTorch fallback requested but torch is not installed." |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:74` | _GYRO_BOHM_DEFAULT = 0.1 # Fallback if JSON not found |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:284` | # Vectorized Python fallback |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:385` | # Vectorized Python fallback |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:930` | # Base Level: neoclassical + gyro-Bohm, or constant fallback |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:1008` | # Fallback: simple edge suppression |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:1147` | fallback: np.ndarray, |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:1154` | fb = np.asarray(fallback, dtype=np.float64) |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/neural_transport.py:141` | """Analytic critical-gradient transport model (fallback).""" |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/neural_transport.py:324` | """Neural transport surrogate with analytic fallback. |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/neural_transport.py:360` | "critical-gradient fallback", |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/quasi_3d_contracts.py:178` | fallback = _require_finite_float("fallback_asymmetry", fallback_asymmetry) |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/quasi_3d_contracts.py:183` | ratio = float(np.clip(0.06 + 0.40 * abs(fallback), 0.0, 0.9)) |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/quasi_3d_contracts.py:200` | ratio = float(np.clip(0.06 + 0.40 * abs(fallback), 0.0, 0.9)) |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/tglf_interface.py:497` | # Fallback: check for JSON output |
| P2 | `other` | `SIMPLIFIED` | `CHANGELOG.md:109` | - **D-T Reactivity Fix**: Replaced simplified Huba fit with NRL Plasma Formulary 5-coefficient Bosch-Hale parameterisation for `sigma_v_d... |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/compiler.py:14` | - Float-path fallback when sc_neurocore is not installed. |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/compiler.py:36` | # ── sc_neurocore import (graceful fallback) ────────────────────────────────── |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/compiler.py:127` | Holds both the dense float matrices (for validation / fallback) and |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/compiler.py:189` | "Use dense_forward_float for the numpy fallback." |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/compiler.py:204` | # Vectorized path when numpy bit_count is available; fallback keeps |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/safety_interlocks.py:98` | def _safe_float(state: Mapping[str, float], key: str, fallback: float) -> float: |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/safety_interlocks.py:99` | value = float(state.get(key, fallback)) |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/safety_interlocks.py:100` | return value if np.isfinite(value) else float(fallback) |
| P2 | `nuclear` | `FALLBACK` | `src/scpn_fusion/nuclear/blanket_neutronics.py:159` | else: # pragma: no cover - legacy NumPy fallback |
| P2 | `diagnostics_io` | `FALLBACK` | `src/scpn_fusion/diagnostics/tomography.py:125` | # Analytic Ridge fallback |
| P2 | `diagnostics_io` | `FALLBACK` | `src/scpn_fusion/io/tokamak_archive.py:8` | """Empirical tokamak profile loaders with optional live MDSplus fallback.""" |
| P2 | `diagnostics_io` | `FALLBACK` | `src/scpn_fusion/io/tokamak_archive.py:362` | Poll live MDSplus feed snapshots with deterministic merge + fallback metadata. |
| P2 | `validation` | `FALLBACK` | `tools/download_diiid_data.py:185` | # ── Reference data fallback ───────────────────────────────────────── |
| P2 | `validation` | `FALLBACK` | `tools/download_diiid_data.py:230` | logger.debug("tokamak_archive fallback failed: %s", exc) |
| P2 | `validation` | `FALLBACK` | `tools/download_efit_geqdsk.py:91` | # Fallback: any .geqdsk file whose name contains the shot number |
| P2 | `validation` | `FALLBACK` | `tools/generate_fno_qlknn_spatial.py:142` | print(" Will use critical-gradient fallback (less accurate).") |
| P2 | `validation` | `FALLBACK` | `tools/generate_fno_qlknn_spatial.py:200` | # Fallback: use critical gradient model |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_disturbance_rejection.py:26` | ``get_radial_robust_controller()`` with LQR fallback. |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_disturbance_rejection.py:204` | # LQR Robust Controller (fallback for H-infinity) |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_disturbance_rejection.py:720` | """Build the H-infinity controller with LQR fallback.""" |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_disturbance_rejection.py:729` | f" [H-infinity] ARE failed ({exc}); using LQR fallback" |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_vs_torax.py:134` | # Analytic fallback: add small model-dependent perturbation |
| P2 | `validation` | `FALLBACK` | `validation/rmse_dashboard.py:303` | # Fallback: legacy FusionBurnPhysics path |
| P2 | `docs_claims` | `SIMPLIFIED` | `docs/assets/generate_header.py:27` | # Simplified Grad-Shafranov flux function with Shafranov shift + triangularity |
| P2 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/control.rst:197` | simplified equilibrium and transport problems, used primarily for: |
| P2 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/transport.rst:82` | using simplified ray-tracing: |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:19` | - Runtime hardening: compiler git SHA probe now uses bounded subprocess timeout with deterministic fallback. |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:27` | - Tooling hardening: claims audit git file discovery now uses a bounded subprocess timeout with safe fallback. |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:33` | - Runtime hardening: deprecated FNO suppressor now logs missing-weight fallback via standard `logging` without raising when optional weig... |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:38` | - Validation hardening: TORAX cross-validation fallback now uses deterministic stable seeding (BLAKE2-based) instead of Python process-ra... |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:39` | - Validation observability: TORAX benchmark artifacts now expose per-case backend, fallback reason, and fallback seed fields. |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:40` | - Validation hardening: TORAX benchmark adds strict backend gating (`--strict-backend`) to fail when reduced-order fallback is used. |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:41` | - Validation hardening: SPARC GEQDSK RMSE benchmark fallback no longer uses identity-like reconstruction; it now uses a deterministic red... |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:42` | - Validation observability: SPARC GEQDSK RMSE benchmark now records surrogate backend and fallback reason per case, plus strict backend r... |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:46` | - Regression test for deprecated FNO-controller missing-weight fallback path. |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:47` | - Regression tests for TORAX benchmark deterministic fallback seeding and backend/fallback metadata fields. |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:68` | - **Rust Transport Delegation**: `chang_hinton_chi_profile()` → Rust fast-path (4.7x speedup), `calculate_sauter_bootstrap_current_full()... |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:140` | - Robust fallback to SciPy L-BFGS-B when JAX is unavailable. |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:359` | - Python-side wrappers in `_rust_compat.py`: `RustSnnPool`, `RustSnnController` with graceful fallback |
| P3 | `other` | `FALLBACK` | `src/scpn_fusion/ui/app.py:27` | # Fallback for dev mode |
| P3 | `other` | `FALLBACK` | `src/scpn_fusion/ui/dashboard_generator.py:97` | # Fallback: Plot contours of Psi which represent the unperturbed surfaces |
| P3 | `validation` | `PLANNED` | `validation/validate_real_shots.py:565` | f"({THRESHOLDS['disruption_fpr_max']:.0%}); tuning planned for v2.1" |
| P3 | `docs_claims` | `FALLBACK` | `README.md:108` | \| Graceful degradation \| Every path has a pure-Python fallback \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/3d_gaps.md:92` | - fallback if external dependency is unavailable. |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:167` | - Transparent fallback to analytic model when no weights are available |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:259` | with fallback to CPU SIMD for systems without GPU support. |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:88` | ## Figure 2: Ion Thermal Diffusivity — Fallback vs MLP (Line Plot) |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:93` | │ ╱ Fallback (analytic) |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:124` | \| Fallback vectorised \| ~2 ms \| ~7× slower \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARK_FIGURES.md:125` | \| Fallback point-by-point loop \| ~200 ms \| ~670× slower \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:103` | - Transparent fallback to full physics solve when surrogate confidence is below threshold |
| P3 | `docs_claims` | `FALLBACK` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:276` | - f32 precision with tolerance-guarded fallback to f64 CPU path |
| P3 | `docs_claims` | `FALLBACK` | `docs/DOE_ARPA_E_CONVERGENCE_PITCH.md:508` | **2. Rust Performance with Python Accessibility:** The dual-language architecture provides 10-50x performance over pure-Python alternativ... |
| P3 | `docs_claims` | `FALLBACK` | `docs/GPU_ACCELERATION_ROADMAP.md:35` | - Host orchestration in Rust with deterministic CPU fallback |
| P3 | `docs_claims` | `FALLBACK` | `docs/GPU_ACCELERATION_ROADMAP.md:64` | - CPU fallback for unsupported nodes |
| P3 | `docs_claims` | `FALLBACK` | `docs/GPU_ACCELERATION_ROADMAP.md:122` | - automatic CPU fallback |
| P3 | `docs_claims` | `FALLBACK` | `docs/HONEST_SCOPE.md:18` | \| Graceful degradation (no Rust / no GPU / no SC-NeuroCore) \| Every module has a pure-Python fallback \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:41` | - [6.2 Import Boundary and Fallback Strategy](#62-import-boundary-and-fallback-strategy) |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:269` | **Dependencies:** numpy, sc_neurocore (optional — graceful fallback) |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:337` | **`dense_forward_float(W, inputs)`** — The float path: a simple `W @ inputs` using numpy. Used for validation and as a fallback when sc_n... |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:486` | ### 6.2 Import Boundary and Fallback Strategy |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:502` | **Rationale:** The Logic Compiler is a component of SCPN-Fusion-Core, which may be installed in environments where sc_neurocore is not pr... |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:660` | | **Graceful sc_neurocore fallback** | SCPN-Fusion-Core may be used without neuromorphic hardware. The Petri Net API should always be ava... |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURO_SYMBOLIC_LOGIC_COMPILER_REPORT.md:740` | The "Tokens are Bits" philosophy creates a direct isomorphism between Petri Net semantics and stochastic computing primitives, guaranteei... |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEXT_SPRINT_EXECUTION_QUEUE.md:37` | - Completed: `S1-003` (added low-point LCFS fallback regression test and VMEC-like geometry CI smoke coverage). |
| P3 | `docs_claims` | `FALLBACK` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:779` | commit SC state (or oracle fallback) |
| P3 | `docs_claims` | `FALLBACK` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:786` | The oracle path is always available and always correct. The SC path currently returns the oracle result (fallback) but is structured so t... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:1721` | The untested lines in `artifact.py` are error-handling paths for malformed JSON that would require intentionally corrupted artifacts to e... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:2904` | 4. **Fallback policy.** If the divergence exceeds a threshold, the controller should revert to the oracle path. This provides a safety ne... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE2_ADVANCED_RFC_TEMPLATE.md:36` | - Offline fallback path: |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:44` | | S2-004 | P1 | Control | Add robust model-loading fallback path in disruption predictor | `src/scpn_fusion/control/disruption_predictor.... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:57` | | S3-004 | P1 | Control | Normalize control simulation imports and deterministic fallback entry points | `src/scpn_fusion/control/disrupt... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:86` | | H5-013 | P1 | SCPN | Add Rust stochastic-firing kernel bridge for runtime backend and validate Rust sample path execution | `src/scpn_f... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:142` | | H7-005 | P1 | Control | Add deterministic NumPy LIF fallback backend for neuro-cybernetic spiking pool when sc-neurocore is unavailable... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:143` | | H7-006 | P1 | Control | Harden Director interface with deterministic built-in fallback oversight and CI-safe summary execution | `src/s... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:145` | | H7-008 | P1 | Nuclear | Restore NumPy 2.4 compatibility for blanket TBR integration in Python 3.11 CI lane | `src/scpn_fusion/nuclear/b... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:152` | | H7-015 | P1 | Diagnostics | Harden tomography inversion with strict signal guards and deterministic non-SciPy fallback solve path | `sr... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:165` | | H7-028 | P1 | Control | Remove residual global NumPy RNG mutation from tearing-mode simulation fallback path | `src/scpn_fusion/control... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:198` | | H7-061 | P1 | Control | Harden director-interface runtime mission with strict duration/glitch input guards | `src/scpn_fusion/control/d... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:200` | | H7-063 | P1 | Control | Harden disruption-predictor sequence-length handling with strict guardrails | `src/scpn_fusion/control/disrupti... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:201` | | H7-064 | P1 | Control | Harden fallback-director constructor with strict entropy/window input guards | `src/scpn_fusion/control/directo... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:202` | | H7-065 | P1 | Control | Harden disruption-predictor training campaign with strict shot/epoch input guards | `src/scpn_fusion/control/di... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:205` | | H7-068 | P1 | HPC | Harden native convergence bridge with strict max-iteration/tolerance/omega guards | `src/scpn_fusion/hpc/hpc_bridge... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:208` | | H7-071 | P1 | Control | Harden spiking-controller constructor with strict neuron/window/timebase/noise guards | `src/scpn_fusion/contro... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:209` | | H7-072 | P1 | Diagnostics | Harden tomography constructor with strict grid/regularization input guards | `src/scpn_fusion/diagnostics/t... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:216` | | H7-079 | P1 | Control | Harden disruption predictor sequence/training integer knobs with strict type/range guards | `src/scpn_fusion/co... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:252` | | H8-025 | P1 | Control | Harden analytic Shafranov and coil-solve APIs with strict finite/conditioning validation | `scpn-fusion-rs/crat... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:256` | | H8-029 | P1 | Operations | Harden digital-twin bit-flip fault injection with strict input/index validation | `scpn-fusion-rs/crates/fus... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:261` | | H8-034 | P1 | Runtime | Harden JIT kernel runtime execution with strict active-kernel and vector validation | `scpn-fusion-rs/crates/fu... |
| P3 | `docs_claims` | `FALLBACK` | `docs/PHASE3_EXECUTION_REGISTRY.md:263` | | H8-036 | P1 | Runtime | Harden equilibrium probe-sampling APIs with strict coordinate/axis validation | `scpn-fusion-rs/crates/fusion-c... |
| P3 | `docs_claims` | `FALLBACK` | `docs/SOLVER_TUNING_GUIDE.md:187` | \| 1 \| 1× (serial fallback) \| Same as before \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/SOLVER_TUNING_GUIDE.md:258` | \| Discontinuity at threshold \| Fallback model uses hard cutoff \| Switch to MLP weights (smooth transition) \| |
| P3 | `docs_claims` | `FALLBACK` | `docs/promotions/HN_SHOW.md:11` | SCPN Fusion Core is an open-source tokamak plasma physics simulator that covers the full lifecycle of a fusion reactor: Grad-Shafranov eq... |
| P3 | `docs_claims` | `FALLBACK` | `docs/promotions/REDDIT_COMPILERS.md:15` | 2. **Compilation** (`scpn/compiler.py`) -- Each Petri net transition maps to a stochastic leaky-integrate-and-fire (LIF) neuron. Places b... |
| P3 | `docs_claims` | `FALLBACK` | `docs/promotions/REDDIT_PROGRAMMING.md:13` | - Transparent fallback: `try: import scpn_fusion_rs` -- if the Rust extension isn't built, NumPy kicks in |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GAI-01_RFC.md:43` | - Offline fallback path: benchmark remains fully offline (`numpy`, existing project modules). |
| P3 | `docs_claims` | `FALLBACK` | `docs/rfc/GAI-02_RFC.md:42` | - Offline fallback path: full campaign runs with stdlib + NumPy + in-repo modules. |
