# Underdeveloped Register

- Generated at: `2026-02-19T18:36:55.320442+00:00`
- Generator: `tools/generate_underdeveloped_register.py`
- Scope: production code + docs claims markers (tests/reports/html excluded)

## Executive Summary

| Metric | Value |
|---|---:|
| Total flagged entries | 312 |
| P0 + P1 entries | 114 |
| Domains affected | 8 |

## Marker Distribution

| Key | Count |
|---|---:|
| `FALLBACK` | 160 |
| `EXPERIMENTAL` | 65 |
| `SIMPLIFIED` | 44 |
| `PLANNED` | 27 |
| `DEPRECATED` | 10 |
| `NOT_VALIDATED` | 6 |

## Domain Distribution

| Key | Count |
|---|---:|
| `docs_claims` | 152 |
| `core_physics` | 58 |
| `other` | 35 |
| `control` | 28 |
| `validation` | 24 |
| `compiler_runtime` | 8 |
| `nuclear` | 5 |
| `diagnostics_io` | 2 |

## Top Priority Backlog (Top 80)

| Priority | Score | Domain | Marker | Location | Owner | Proposed Action | Snippet |
|---|---:|---|---|---|---|---|---|
| P0 | 104 | `core_physics` | `DEPRECATED` | `src/scpn_fusion/core/fno_training.py:11` | Core Physics WG | Replace default path or remove lane before next major release. | .. deprecated:: 2.1.0 |
| P0 | 104 | `core_physics` | `DEPRECATED` | `src/scpn_fusion/core/fno_training.py:13` | Core Physics WG | Replace default path or remove lane before next major release. | **DEPRECATED / EXPERIMENTAL** — Trained on 60 synthetic Hasegawa-Wakatani |
| P0 | 104 | `core_physics` | `DEPRECATED` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:10` | Core Physics WG | Replace default path or remove lane before next major release. | .. deprecated:: 2.1.0 |
| P0 | 104 | `core_physics` | `DEPRECATED` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:12` | Core Physics WG | Replace default path or remove lane before next major release. | **DEPRECATED / EXPERIMENTAL** — The FNO turbulence surrogate has a |
| P0 | 104 | `core_physics` | `DEPRECATED` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:101` | Core Physics WG | Replace default path or remove lane before next major release. | "FNO turbulence surrogate is DEPRECATED (relative L2 ~ 0.79). " |
| P0 | 97 | `core_physics` | `EXPERIMENTAL` | `src/scpn_fusion/core/fno_training.py:13` | Core Physics WG | Gate behind explicit flag and define validation exit criteria. | **DEPRECATED / EXPERIMENTAL** — Trained on 60 synthetic Hasegawa-Wakatani |
| P0 | 97 | `core_physics` | `EXPERIMENTAL` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:12` | Core Physics WG | Gate behind explicit flag and define validation exit criteria. | **DEPRECATED / EXPERIMENTAL** — The FNO turbulence surrogate has a |
| P0 | 95 | `core_physics` | `NOT_VALIDATED` | `src/scpn_fusion/core/fno_training.py:14` | Core Physics WG | Add real-data validation campaign and publish error bars. | samples. Not validated against production gyrokinetic codes (GENE, GS2, |
| P0 | 95 | `core_physics` | `NOT_VALIDATED` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:102` | Core Physics WG | Add real-data validation campaign and publish error bars. | "Results are not validated against gyrokinetic codes (GENE/GS2/QuaLiKiz). " |
| P0 | 95 | `other` | `DEPRECATED` | `CHANGELOG.md:158` | Architecture WG | Replace default path or remove lane before next major release. | #### FNO Turbulence Surrogate Deprecated (Task 3.3) |
| P0 | 95 | `other` | `DEPRECATED` | `CHANGELOG.md:159` | Architecture WG | Replace default path or remove lane before next major release. | - Module docstrings updated: EXPERIMENTAL → DEPRECATED/EXPERIMENTAL |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:82` | Validation WG | Gate behind explicit flag and define validation exit criteria. | "Experimental-only pytest suite", |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:83` | Validation WG | Gate behind explicit flag and define validation exit criteria. | [sys.executable, "-m", "pytest", "tests/", "-q", "-m", "experimental"], |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:136` | Validation WG | Gate behind explicit flag and define validation exit criteria. | "'release' excludes experimental-only lanes, " |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:137` | Validation WG | Gate behind explicit flag and define validation exit criteria. | "'research' runs experimental-only lanes, " |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:173` | Validation WG | Gate behind explicit flag and define validation exit criteria. | help="Skip pytest experimental-only lane (tests/ -m experimental).", |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `validation/collect_results.py:360` | Validation WG | Gate behind explicit flag and define validation exit criteria. | rows_s.append(f"| FNO (EUROfusion JET) relative L2 (mean) | {_fmt(surrogates['fno_rel_l2_mean'], '.4f')} | — | ψ(R,Z) reconstruction (**E... |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `validation/collect_results.py:370` | Validation WG | Gate behind explicit flag and define validation exit criteria. | sections.append("> **EXPERIMENTAL — FNO turbulence surrogate:** Relative L2 ~ 0.79 means the model") |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `validation/full_validation_pipeline.py:12` | Validation WG | Gate behind explicit flag and define validation exit criteria. | - MPC/RL controller metrics vs experimental-profile proxies |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `validation/reference_data/README.md:3` | Validation WG | Gate behind explicit flag and define validation exit criteria. | Experimental and design reference data for cross-validating SCPN Fusion Core |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `validation/run_experimental_validation.py:2` | Validation WG | Gate behind explicit flag and define validation exit criteria. | # SCPN Fusion Core — Unified Experimental Validation Runner |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `validation/run_experimental_validation.py:190` | Validation WG | Gate behind explicit flag and define validation exit criteria. | print(" SCPN Fusion Core — Unified Experimental Validation") |
| P1 | 94 | `validation` | `EXPERIMENTAL` | `validation/validate_against_sparc.py:185` | Validation WG | Gate behind explicit flag and define validation exit criteria. | print(" SCPN Fusion Core - Validation Against Experimental Data") |
| P1 | 92 | `validation` | `NOT_VALIDATED` | `validation/collect_results.py:371` | Validation WG | Add real-data validation campaign and publish error bars. | sections.append("> explains only ~21% of the variance. Trained on 60 synthetic samples; NOT validated") |
| P1 | 89 | `docs_claims` | `DEPRECATED` | `RESULTS.md:200` | Docs WG | Replace default path or remove lane before next major release. | > Retraining on real gyrokinetic data or retirement planned for v4.0. Deprecated since v3.0. |
| P1 | 89 | `docs_claims` | `DEPRECATED` | `RESULTS.md:472` | Docs WG | Replace default path or remove lane before next major release. | \| FNO L2 = 0.79 (21% variance explained) \| Medium \| DEPRECATED \| Runtime FutureWarning; retire in v4.0 \| |
| P1 | 89 | `docs_claims` | `DEPRECATED` | `docs/sphinx/changelog.rst:27` | Docs WG | Replace default path or remove lane before next major release. | - FNO turbulence surrogate deprecated (runtime ``FutureWarning``) |
| P1 | 88 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:51` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | - Added research-only pytest marker contract (`@pytest.mark.experimental`) |
| P1 | 88 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:52` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | - Added CI split lane `python-research-gate` and release-only pytest execution (`-m "not experimental"`) |
| P1 | 88 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:62` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | - Added single-command suite execution via `scpn-fusion all --surrogate --experimental`. |
| P1 | 88 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:159` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | - Module docstrings updated: EXPERIMENTAL → DEPRECATED/EXPERIMENTAL |
| P1 | 88 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:298` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | #### Validation and Experimental Data |
| P1 | 88 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:351` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | - Experimental modes gated behind `--experimental` flag |
| P1 | 88 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:473` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | Reduced-order / Experimental) with hardening task counts. |
| P1 | 88 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:478` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | ### Multigrid Wiring and Experimental Validation |
| P1 | 88 | `other` | `EXPERIMENTAL` | `VALIDATION.md:61` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | - Real DIII-D experimental data are not bundled in this specific `disruption_shots` folder. |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:64` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | # Experimental modes (opt-in). |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:65` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | "quantum": ModeSpec("scpn_fusion.core.quantum_bridge", "experimental", "Quantum bridge"), |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:66` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | "q-control": ModeSpec("scpn_fusion.core.quantum_bridge", "experimental", "Quantum control bridge"), |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:69` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | "experimental", |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:72` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | "lazarus": ModeSpec("scpn_fusion.core.lazarus_bridge", "experimental", "Lazarus bridge"), |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:73` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | "director": ModeSpec("scpn_fusion.control.director_interface", "experimental", "Director interface"), |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:74` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | "vibrana": ModeSpec("scpn_fusion.core.vibrana_bridge", "experimental", "Vibrana bridge"), |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:101` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | elif spec.maturity == "experimental" and include_experimental: |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:115` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | if spec.maturity == "experimental" and not include_experimental: |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:116` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | return "experimental mode locked; pass --experimental or set SCPN_EXPERIMENTAL=1" |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:172` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | @click.option("--experimental", is_flag=True, help="Unlock experimental modes.") |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:188` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | experimental: bool, |
| P1 | 88 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:201` | Architecture WG | Gate behind explicit flag and define validation exit criteria. | include_experimental = experimental or _env_enabled("SCPN_EXPERIMENTAL") |
| P1 | 84 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/disruption_predictor.py:71` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Rutherford Equation: dw/dt = Delta' + Const/w (simplified) |
| P1 | 84 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/halo_re_physics.py:455` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | Simplified Relativistic Fokker-Planck generation rate. |
| P1 | 84 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/tokamak_digital_twin.py:141` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | radiation = 0.0001 * (self.T**2) # Simplified T^2 for numeric stability |
| P1 | 84 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/tokamak_digital_twin.py:184` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | Simplified Policy Gradient update (REINFORCE-like). |
| P1 | 84 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/tokamak_digital_twin.py:192` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified: We want to push 'pred' in direction of Advantage |
| P1 | 84 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/tokamak_digital_twin.py:249` | Control WG | Upgrade with higher-fidelity closure or tighten domain contract. | # State: Simplified to radial profile samples (to keep NN small) |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/divertor_thermal_sim.py:71` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | Vapor Shielding Physics (Simplified). |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/divertor_thermal_sim.py:90` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified: Radiative fraction increases with T (more vapor) |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/eped_pedestal.py:9` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | EPED-like simplified pedestal scaling model for H-mode tokamak plasmas. |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/eped_pedestal.py:64` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | """Simplified EPED-like pedestal model. |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/eped_pedestal.py:164` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Pressure gradient constraint (simplified KBM limit): |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/fusion_ignition_sim.py:52` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # In our simplified kernel, J was modeled directly. |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/fusion_ignition_sim.py:100` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # 4. Losses (Simplified Confinement scaling) |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/fusion_ignition_sim.py:102` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified: Fixed loss + Bremsstrahlung |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/fusion_ignition_sim.py:393` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Impurity line radiation (simplified) |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/global_design_scanner.py:61` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | max_pressure = beta_N * (I_plasma / (a_min * B_field)) * (B_field**2) # Simplified |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_transport_solver.py:556` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | using a simplified Sauter model. |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_transport_solver.py:559` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified Sauter model coefficients |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_transport_solver.py:1115` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | Z_W = 10.0 # effective charge state for tungsten (simplified) |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_transport_solver.py:1246` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified: S_eq = (Ti - Te) / tau_eq, tau_eq ~ 0.1 s for ITER |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/mhd_sawtooth.py:70` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified linear growth rate drive: |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/mhd_sawtooth.py:98` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Del^2 phi = U. Using simplified relaxation or direct solve. |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neural_transport.py:131` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | _CRIT_TEM = 5.0 # R/L_Te threshold for TEM (simplified) |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neural_transport.py:143` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | D_e = chi_e / 3 (simplified Ware pinch) |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/rf_heating.py:89` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | # Simplified: Gaussian blob |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/rf_heating.py:102` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | Simplified Cold Plasma (Alfven wave approx). |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/sandpile_fusion_reactor.py:92` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | A simplified Reinforcement Learning agent. |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/stability_mhd.py:372` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | r"""Simplified neoclassical tearing mode (NTM) seeding analysis. |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/stability_mhd.py:374` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | The modified Rutherford equation for island width *w* is (simplified): |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/uncertainty.py:205` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | Estimate fusion power from confinement time using simplified power balance. |
| P1 | 83 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/uncertainty.py:210` | Core Physics WG | Upgrade with higher-fidelity closure or tighten domain contract. | P_fus ≈ (n_e * tau_E * T_i)^2 scaling, simplified to: |

## Full Register (Top 250)

| Priority | Domain | Marker | Location | Snippet |
|---|---|---|---|---|
| P0 | `core_physics` | `DEPRECATED` | `src/scpn_fusion/core/fno_training.py:11` | .. deprecated:: 2.1.0 |
| P0 | `core_physics` | `DEPRECATED` | `src/scpn_fusion/core/fno_training.py:13` | **DEPRECATED / EXPERIMENTAL** — Trained on 60 synthetic Hasegawa-Wakatani |
| P0 | `core_physics` | `DEPRECATED` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:10` | .. deprecated:: 2.1.0 |
| P0 | `core_physics` | `DEPRECATED` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:12` | **DEPRECATED / EXPERIMENTAL** — The FNO turbulence surrogate has a |
| P0 | `core_physics` | `DEPRECATED` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:101` | "FNO turbulence surrogate is DEPRECATED (relative L2 ~ 0.79). " |
| P0 | `core_physics` | `EXPERIMENTAL` | `src/scpn_fusion/core/fno_training.py:13` | **DEPRECATED / EXPERIMENTAL** — Trained on 60 synthetic Hasegawa-Wakatani |
| P0 | `core_physics` | `EXPERIMENTAL` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:12` | **DEPRECATED / EXPERIMENTAL** — The FNO turbulence surrogate has a |
| P0 | `core_physics` | `NOT_VALIDATED` | `src/scpn_fusion/core/fno_training.py:14` | samples. Not validated against production gyrokinetic codes (GENE, GS2, |
| P0 | `core_physics` | `NOT_VALIDATED` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:102` | "Results are not validated against gyrokinetic codes (GENE/GS2/QuaLiKiz). " |
| P0 | `other` | `DEPRECATED` | `CHANGELOG.md:158` | #### FNO Turbulence Surrogate Deprecated (Task 3.3) |
| P0 | `other` | `DEPRECATED` | `CHANGELOG.md:159` | - Module docstrings updated: EXPERIMENTAL → DEPRECATED/EXPERIMENTAL |
| P1 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:82` | "Experimental-only pytest suite", |
| P1 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:83` | [sys.executable, "-m", "pytest", "tests/", "-q", "-m", "experimental"], |
| P1 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:136` | "'release' excludes experimental-only lanes, " |
| P1 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:137` | "'research' runs experimental-only lanes, " |
| P1 | `validation` | `EXPERIMENTAL` | `tools/run_python_preflight.py:173` | help="Skip pytest experimental-only lane (tests/ -m experimental).", |
| P1 | `validation` | `EXPERIMENTAL` | `validation/collect_results.py:360` | rows_s.append(f"| FNO (EUROfusion JET) relative L2 (mean) | {_fmt(surrogates['fno_rel_l2_mean'], '.4f')} | — | ψ(R,Z) reconstruction (**E... |
| P1 | `validation` | `EXPERIMENTAL` | `validation/collect_results.py:370` | sections.append("> **EXPERIMENTAL — FNO turbulence surrogate:** Relative L2 ~ 0.79 means the model") |
| P1 | `validation` | `EXPERIMENTAL` | `validation/full_validation_pipeline.py:12` | - MPC/RL controller metrics vs experimental-profile proxies |
| P1 | `validation` | `EXPERIMENTAL` | `validation/reference_data/README.md:3` | Experimental and design reference data for cross-validating SCPN Fusion Core |
| P1 | `validation` | `EXPERIMENTAL` | `validation/run_experimental_validation.py:2` | # SCPN Fusion Core — Unified Experimental Validation Runner |
| P1 | `validation` | `EXPERIMENTAL` | `validation/run_experimental_validation.py:190` | print(" SCPN Fusion Core — Unified Experimental Validation") |
| P1 | `validation` | `EXPERIMENTAL` | `validation/validate_against_sparc.py:185` | print(" SCPN Fusion Core - Validation Against Experimental Data") |
| P1 | `validation` | `NOT_VALIDATED` | `validation/collect_results.py:371` | sections.append("> explains only ~21% of the variance. Trained on 60 synthetic samples; NOT validated") |
| P1 | `docs_claims` | `DEPRECATED` | `RESULTS.md:200` | > Retraining on real gyrokinetic data or retirement planned for v4.0. Deprecated since v3.0. |
| P1 | `docs_claims` | `DEPRECATED` | `RESULTS.md:472` | \| FNO L2 = 0.79 (21% variance explained) \| Medium \| DEPRECATED \| Runtime FutureWarning; retire in v4.0 \| |
| P1 | `docs_claims` | `DEPRECATED` | `docs/sphinx/changelog.rst:27` | - FNO turbulence surrogate deprecated (runtime ``FutureWarning``) |
| P1 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:51` | - Added research-only pytest marker contract (`@pytest.mark.experimental`) |
| P1 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:52` | - Added CI split lane `python-research-gate` and release-only pytest execution (`-m "not experimental"`) |
| P1 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:62` | - Added single-command suite execution via `scpn-fusion all --surrogate --experimental`. |
| P1 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:159` | - Module docstrings updated: EXPERIMENTAL → DEPRECATED/EXPERIMENTAL |
| P1 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:298` | #### Validation and Experimental Data |
| P1 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:351` | - Experimental modes gated behind `--experimental` flag |
| P1 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:473` | Reduced-order / Experimental) with hardening task counts. |
| P1 | `other` | `EXPERIMENTAL` | `CHANGELOG.md:478` | ### Multigrid Wiring and Experimental Validation |
| P1 | `other` | `EXPERIMENTAL` | `VALIDATION.md:61` | - Real DIII-D experimental data are not bundled in this specific `disruption_shots` folder. |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:64` | # Experimental modes (opt-in). |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:65` | "quantum": ModeSpec("scpn_fusion.core.quantum_bridge", "experimental", "Quantum bridge"), |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:66` | "q-control": ModeSpec("scpn_fusion.core.quantum_bridge", "experimental", "Quantum control bridge"), |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:69` | "experimental", |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:72` | "lazarus": ModeSpec("scpn_fusion.core.lazarus_bridge", "experimental", "Lazarus bridge"), |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:73` | "director": ModeSpec("scpn_fusion.control.director_interface", "experimental", "Director interface"), |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:74` | "vibrana": ModeSpec("scpn_fusion.core.vibrana_bridge", "experimental", "Vibrana bridge"), |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:101` | elif spec.maturity == "experimental" and include_experimental: |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:115` | if spec.maturity == "experimental" and not include_experimental: |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:116` | return "experimental mode locked; pass --experimental or set SCPN_EXPERIMENTAL=1" |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:172` | @click.option("--experimental", is_flag=True, help="Unlock experimental modes.") |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:188` | experimental: bool, |
| P1 | `other` | `EXPERIMENTAL` | `src/scpn_fusion/cli.py:201` | include_experimental = experimental or _env_enabled("SCPN_EXPERIMENTAL") |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/disruption_predictor.py:71` | # Rutherford Equation: dw/dt = Delta' + Const/w (simplified) |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/halo_re_physics.py:455` | Simplified Relativistic Fokker-Planck generation rate. |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/tokamak_digital_twin.py:141` | radiation = 0.0001 * (self.T**2) # Simplified T^2 for numeric stability |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/tokamak_digital_twin.py:184` | Simplified Policy Gradient update (REINFORCE-like). |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/tokamak_digital_twin.py:192` | # Simplified: We want to push 'pred' in direction of Advantage |
| P1 | `control` | `SIMPLIFIED` | `src/scpn_fusion/control/tokamak_digital_twin.py:249` | # State: Simplified to radial profile samples (to keep NN small) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/divertor_thermal_sim.py:71` | Vapor Shielding Physics (Simplified). |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/divertor_thermal_sim.py:90` | # Simplified: Radiative fraction increases with T (more vapor) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/eped_pedestal.py:9` | EPED-like simplified pedestal scaling model for H-mode tokamak plasmas. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/eped_pedestal.py:64` | """Simplified EPED-like pedestal model. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/eped_pedestal.py:164` | # Pressure gradient constraint (simplified KBM limit): |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/fusion_ignition_sim.py:52` | # In our simplified kernel, J was modeled directly. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/fusion_ignition_sim.py:100` | # 4. Losses (Simplified Confinement scaling) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/fusion_ignition_sim.py:102` | # Simplified: Fixed loss + Bremsstrahlung |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/fusion_ignition_sim.py:393` | # Impurity line radiation (simplified) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/global_design_scanner.py:61` | max_pressure = beta_N * (I_plasma / (a_min * B_field)) * (B_field**2) # Simplified |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_transport_solver.py:556` | using a simplified Sauter model. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_transport_solver.py:559` | # Simplified Sauter model coefficients |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_transport_solver.py:1115` | Z_W = 10.0 # effective charge state for tungsten (simplified) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/integrated_transport_solver.py:1246` | # Simplified: S_eq = (Ti - Te) / tau_eq, tau_eq ~ 0.1 s for ITER |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/mhd_sawtooth.py:70` | # Simplified linear growth rate drive: |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/mhd_sawtooth.py:98` | # Del^2 phi = U. Using simplified relaxation or direct solve. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neural_transport.py:131` | _CRIT_TEM = 5.0 # R/L_Te threshold for TEM (simplified) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/neural_transport.py:143` | D_e = chi_e / 3 (simplified Ware pinch) |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/rf_heating.py:89` | # Simplified: Gaussian blob |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/rf_heating.py:102` | Simplified Cold Plasma (Alfven wave approx). |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/sandpile_fusion_reactor.py:92` | A simplified Reinforcement Learning agent. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/stability_mhd.py:372` | r"""Simplified neoclassical tearing mode (NTM) seeding analysis. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/stability_mhd.py:374` | The modified Rutherford equation for island width *w* is (simplified): |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/uncertainty.py:205` | Estimate fusion power from confinement time using simplified power balance. |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/uncertainty.py:210` | P_fus ≈ (n_e * tau_E * T_i)^2 scaling, simplified to: |
| P1 | `core_physics` | `SIMPLIFIED` | `src/scpn_fusion/core/uncertainty.py:218` | # Simplified fusion power model calibrated to ITER: |
| P1 | `docs_claims` | `EXPERIMENTAL` | `README.md:138` | scpn-fusion all --surrogate --experimental # one command for full unlocked suite |
| P1 | `docs_claims` | `EXPERIMENTAL` | `README.md:240` | ## Validation Against Experimental Data |
| P1 | `docs_claims` | `EXPERIMENTAL` | `README.md:314` | ### Experimental — Requires external SCPN framework components |
| P1 | `docs_claims` | `EXPERIMENTAL` | `README.md:317` | scpn-fusion quantum --experimental |
| P1 | `docs_claims` | `EXPERIMENTAL` | `RESULTS.md:193` | \| FNO (EUROfusion JET) relative L2 (mean) \| 0.7925 \| — \| psi(R,Z) reconstruction (**EXPERIMENTAL**) \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `RESULTS.md:196` | > **EXPERIMENTAL — FNO turbulence surrogate:** Relative L2 = 0.79 means the model |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/BENCHMARKS.md:46` | \| Experimental validation \| SPARC, ITPA, JET \| DIII-D \| ITER, DEMO \| JET \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/BENCHMARKS.md:331` | scpn-fusion all --surrogate --experimental |
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
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:15` | | `release` | Version/claims integrity, notebook quality gate, Task 5/6 threshold smoke, strict typing; excludes experimental tests from ... |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:16` | \| `research` \| Experimental-only pytest lane (`pytest -m experimental`). \| `python tools/run_python_preflight.py --gate research` \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:24` | \| `python-research-gate` \| Experimental validation lane (3.12) \| `research` \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:25` | \| `validation-regression` \| Cross-language physics validation lane \| `release` (`pytest -m "not experimental"`) \| |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:27` | ## Experimental Marker Contract |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:29` | Tests marked with `@pytest.mark.experimental` are considered research-only and are excluded from release acceptance runs. As of v3.5.x th... |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/VALIDATION_GATE_MATRIX.md:42` | 1. Remove `@pytest.mark.experimental` from the test module(s). |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/api/core.rst:225` | Experimental Bridges |
| P1 | `docs_claims` | `EXPERIMENTAL` | `docs/sphinx/userguide/validation.rst:5` | SCPN-Fusion-Core is validated against published experimental data from |
| P1 | `nuclear` | `SIMPLIFIED` | `src/scpn_fusion/nuclear/blanket_neutronics.py:71` | # Cross Sections (Macroscopic Sigma in cm^-1) - Simplified for 14 MeV neutrons |
| P1 | `nuclear` | `SIMPLIFIED` | `src/scpn_fusion/nuclear/blanket_neutronics.py:146` | # Incoming Current (Approx D * dPhi/dx at boundary, or simplified Incident Flux) |
| P1 | `nuclear` | `SIMPLIFIED` | `src/scpn_fusion/nuclear/blanket_neutronics.py:151` | # Simplified: We normalize by the source that sustains Phi[0]. |
| P1 | `nuclear` | `SIMPLIFIED` | `src/scpn_fusion/nuclear/nuclear_wall_interaction.py:210` | # Simplified flux: Flux ~ Source / Distance (Cylindrical decay approx) |
| P2 | `docs_claims` | `NOT_VALIDATED` | `RESULTS.md:198` | > NOT validated against production gyrokinetic codes (GENE, GS2, QuaLiKiz). A runtime |
| P2 | `docs_claims` | `NOT_VALIDATED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:1731` | 3. **Schema validation.** The JSON Schema (`scpnctl.schema.json`) is not validated against the artifact files in the test suite. Schema v... |
| P2 | `docs_claims` | `NOT_VALIDATED` | `docs/PACKET_C_CONTROL_API_COMPREHENSIVE_STUDY.md:3084` | 5. **No schema validation.** The JSON Schema exists but is not validated at load time. This means a malformed artifact that satisfies the... |
| P2 | `validation` | `SIMPLIFIED` | `validation/benchmark_vs_freegs.py:344` | 1e3, # Axis pressure [Pa] (simplified) |
| P2 | `validation` | `SIMPLIFIED` | `validation/validate_iter.py:59` | # Our model is simplified (L-mode profiles mostly), so if we get > 100MW it's a "physics pass" |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/analytic_solver.py:205` | fallback = repo_root / "validation" / "iter_validated_config.json" |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/analytic_solver.py:206` | config_path = str(preferred if preferred.exists() else fallback) |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/director_interface.py:42` | """Deterministic fallback director when external DIRECTOR_AI is unavailable.""" |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:637` | "fallback": True, |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:663` | "fallback": False, |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:672` | "fallback": True, |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:682` | "fallback": True, |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:698` | "fallback": True, |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:704` | info["fallback"] = False |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:718` | Predict disruption risk with checkpoint path if available, else deterministic fallback. |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:724` | ``metadata`` includes whether fallback mode was used. |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:727` | failures instead of returning fallback risk from ``predict_disruption_risk``. |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:744` | "predict_disruption_risk_safe fallback disabled: " |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:748` | out_meta["mode"] = "fallback" |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:766` | "predict_disruption_risk_safe fallback disabled: " |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/disruption_predictor.py:770` | out_meta["mode"] = "fallback" |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/fusion_control_room.py:317` | print(f"Kernel init failed, fallback to analytic Psi: {kernel_error}") |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/jax_traceable_runtime.py:4` | """Optional JAX-traceable control-loop utilities with NumPy fallback.""" |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/nengo_snn_wrapper.py:35` | # Lazy Nengo import — graceful fallback |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/nengo_snn_wrapper.py:379` | # ── Fallback for when Nengo is not available ───────────────────────── |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/neuro_cybernetic_controller.py:56` | Push-pull spiking control population with deterministic fallback. |
| P2 | `control` | `FALLBACK` | `src/scpn_fusion/control/neuro_cybernetic_controller.py:149` | # Reduced threshold keeps fallback lane responsive in low-current control |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_rust_compat.py:283` | "Use Python multigrid fallback in FusionKernel._multigrid_vcycle()." |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/_rust_compat.py:292` | "Use Python multigrid fallback." |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/fno_turbulence_suppressor.py:34` | # Script-mode fallback |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/fusion_kernel.py:187` | "HPC Acceleration UNAVAILABLE (using Python fallback)." |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/fusion_kernel.py:783` | """Run the inner elliptic solve (HPC or Python fallback). |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/geometry_3d.py:122` | """Build a conservative elliptical boundary fallback inside the domain.""" |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/gpu_runtime.py:106` | raise RuntimeError("PyTorch fallback requested but torch is not installed.") |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/gpu_runtime.py:263` | "PyTorch fallback requested but torch is not installed." |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:41` | _GYRO_BOHM_DEFAULT = 0.1 # Fallback if JSON not found |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:656` | # Base Level: neoclassical + gyro-Bohm, or constant fallback |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:708` | # Fallback: simple edge suppression |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:836` | fallback: np.ndarray, |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/integrated_transport_solver.py:843` | fb = np.asarray(fallback, dtype=np.float64) |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/neural_transport.py:127` | # ── Analytic fallback (critical-gradient model) ────────────────────── |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/neural_transport.py:137` | """Analytic critical-gradient transport model (fallback). |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/neural_transport.py:234` | """Neural transport surrogate with analytic fallback. |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/neural_transport.py:249` | >>> model = NeuralTransportModel() # fallback mode |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/neural_transport.py:275` | "critical-gradient fallback", |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/neural_transport.py:483` | # ── Fallback: vectorised critical-gradient model ───────── |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/quasi_3d_contracts.py:178` | fallback = _require_finite_float("fallback_asymmetry", fallback_asymmetry) |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/quasi_3d_contracts.py:183` | ratio = float(np.clip(0.06 + 0.40 * abs(fallback), 0.0, 0.9)) |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/quasi_3d_contracts.py:200` | ratio = float(np.clip(0.06 + 0.40 * abs(fallback), 0.0, 0.9)) |
| P2 | `core_physics` | `FALLBACK` | `src/scpn_fusion/core/tglf_interface.py:451` | # Fallback: check for JSON output |
| P2 | `other` | `SIMPLIFIED` | `CHANGELOG.md:202` | - NTM seeding: simplified Modified Rutherford equation with bootstrap-drive marginal island width |
| P2 | `other` | `SIMPLIFIED` | `CHANGELOG.md:241` | - EPED-like pedestal model (Snyder 2009 simplified scaling) for H-mode boundary conditions |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/compiler.py:14` | - Float-path fallback when sc_neurocore is not installed. |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/compiler.py:36` | # ── sc_neurocore import (graceful fallback) ────────────────────────────────── |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/compiler.py:125` | Holds both the dense float matrices (for validation / fallback) and |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/compiler.py:187` | "Use dense_forward_float for the numpy fallback." |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/compiler.py:202` | # Vectorized path when numpy bit_count is available; fallback keeps |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/safety_interlocks.py:98` | def _safe_float(state: Mapping[str, float], key: str, fallback: float) -> float: |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/safety_interlocks.py:99` | value = float(state.get(key, fallback)) |
| P2 | `compiler_runtime` | `FALLBACK` | `src/scpn_fusion/scpn/safety_interlocks.py:100` | return value if np.isfinite(value) else float(fallback) |
| P2 | `nuclear` | `FALLBACK` | `src/scpn_fusion/nuclear/blanket_neutronics.py:143` | else: # pragma: no cover - legacy NumPy fallback |
| P2 | `diagnostics_io` | `FALLBACK` | `src/scpn_fusion/io/tokamak_archive.py:8` | """Empirical tokamak profile loaders with optional live MDSplus fallback.""" |
| P2 | `diagnostics_io` | `FALLBACK` | `src/scpn_fusion/io/tokamak_archive.py:362` | Poll live MDSplus feed snapshots with deterministic merge + fallback metadata. |
| P2 | `validation` | `FALLBACK` | `tools/download_diiid_data.py:185` | # ── Reference data fallback ───────────────────────────────────────── |
| P2 | `validation` | `FALLBACK` | `tools/download_diiid_data.py:230` | logger.debug("tokamak_archive fallback failed: %s", exc) |
| P2 | `validation` | `FALLBACK` | `tools/download_efit_geqdsk.py:91` | # Fallback: any .geqdsk file whose name contains the shot number |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_disturbance_rejection.py:26` | ``get_radial_robust_controller()`` with LQR fallback. |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_disturbance_rejection.py:204` | # LQR Robust Controller (fallback for H-infinity) |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_disturbance_rejection.py:722` | """Build the H-infinity controller with LQR fallback.""" |
| P2 | `validation` | `FALLBACK` | `validation/benchmark_disturbance_rejection.py:731` | f" [H-infinity] ARE failed ({exc}); using LQR fallback" |
| P2 | `validation` | `FALLBACK` | `validation/rmse_dashboard.py:303` | # Fallback: legacy FusionBurnPhysics path |
| P2 | `docs_claims` | `SIMPLIFIED` | `RESULTS.md:290` | \| Scaling \| Snyder (2009) simplified \| Delta_ped ~ 0.076 * beta_p_ped^0.5 * nu_star_ped^-0.2 \| |
| P2 | `docs_claims` | `SIMPLIFIED` | `docs/assets/generate_header.py:27` | # Simplified Grad-Shafranov flux function with Shafranov shift + triangularity |
| P2 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/control.rst:197` | simplified equilibrium and transport problems, used primarily for: |
| P2 | `docs_claims` | `SIMPLIFIED` | `docs/sphinx/userguide/transport.rst:82` | using simplified ray-tracing: |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:147` | - Python-side wrappers in `_rust_compat.py`: `RustSnnPool`, `RustSnnController` with graceful fallback |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:214` | - `validation/benchmark_vs_freegs.py` with Solov'ev analytic fallback (no freegs dependency) |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:235` | - PyO3 binding for Rust multigrid `multigrid_vcycle()` with Python fallback |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:359` | - NumPy 2.4 compatibility restored for blanket TBR integration (trapezoid fallback) |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:372` | SCPN benchmark stochastic-vs-float equivalence gate, disruption predictor fallback, |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:378` | control simulation fallback entry points, HPC bridge edge-path validation, |
| P3 | `other` | `FALLBACK` | `CHANGELOG.md:401` | NumPy LIF fallback for neuro-cybernetic controller; director interface fallback |
| P3 | `other` | `FALLBACK` | `src/scpn_fusion/ui/app.py:28` | # Fallback for dev mode |
| P3 | `validation` | `PLANNED` | `validation/validate_real_shots.py:344` | f"({THRESHOLDS['disruption_fpr_max']:.0%}); tuning planned for v2.1" |
| P3 | `docs_claims` | `FALLBACK` | `README.md:283` | | `neuro-control` | SNN-based cybernetic controller (SC-NeuroCore or NumPy LIF fallback) | Deterministic replay, fault injection | H5: 37... |
| P3 | `docs_claims` | `FALLBACK` | `README.md:287` | | `safety` | ML disruption predictor (deterministic scoring + optional Transformer) | Anomaly campaigns, checkpoint fallback | H7: scoped... |
| P3 | `docs_claims` | `FALLBACK` | `README.md:297` | \| `diagnostics` \| Synthetic sensors + soft X-ray tomographic inversion \| Forward models, SciPy fallback \| |
| P3 | `docs_claims` | `FALLBACK` | `README.md:357` | 2. **Compilation** — Petri net transitions mapped to stochastic LIF neurons using [SC-NeuroCore](https://github.com/anulum/sc-neurocore) ... |
| P3 | `docs_claims` | `FALLBACK` | `README.md:373` | - **Graceful degradation** — every path has a pure-Python fallback |
| P3 | `docs_claims` | `FALLBACK` | `README.md:385` | _HAS_SC_NEUROCORE = False # NumPy float-path fallback |
| P3 | `docs_claims` | `FALLBACK` | `README.md:606` | | **GPU acceleration** | Deterministic runtime bridge + optional torch fallback ([GPU Roadmap](docs/GPU_ACCELERATION_ROADMAP.md)) | CUDA-... |
| P3 | `docs_claims` | `FALLBACK` | `docs/3d_gaps.md:92` | - fallback if external dependency is unavailable. |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:171` | - Transparent fallback to analytic model when no weights are available |
| P3 | `docs_claims` | `FALLBACK` | `docs/BENCHMARKS.md:263` | with fallback to CPU SIMD for systems without GPU support. |
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
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURAL_TRANSPORT_TRAINING.md:11` | critical-gradient fallback and exports weights in the `.npz` format expected by: |
| P3 | `docs_claims` | `FALLBACK` | `docs/NEURAL_TRANSPORT_TRAINING.md:33` | Training data is generated synthetically from the analytic fallback model: |
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
