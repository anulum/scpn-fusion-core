# Validation Gate Matrix

This document defines the split between release-grade validation and research-only validation for SCPN Fusion Core.

## Why this exists

- Keep release CI lanes focused on deterministic, production-facing checks.
- Preserve visibility on research-only lanes without letting them silently contaminate release acceptance.
- Make gate intent explicit in both local preflight and GitHub Actions.

## Gate Profiles

| Profile | Scope | Command |
|---|---|---|
| `release` | Version/claims integrity, packaging contract guard, claim-range guard, underdeveloped/source-backlog drift checks, untested-module linkage guard, deprecated-default-lane guard, disruption data provenance + split + calibration holdout checks, disruption replay pipeline contract benchmark, disruption transfer-generalization benchmark, EPED domain-contract benchmark, transport uncertainty-envelope benchmark, TORAX/SPARC strict-backend parity checks, multi-ion transport conservation benchmark, end-to-end latency benchmark, notebook quality gate, Task 5/6 threshold smoke, strict typing; excludes research-marker tests from global pytest runs. | `python tools/run_python_preflight.py --gate release` |
| `research` | Research-only pytest lane (`pytest -m experimental` marker). | `python tools/run_python_preflight.py --gate research` |
| `all` | Release + research profiles in sequence. | `python tools/run_python_preflight.py --gate all` |
| `real-data-strict` | End-to-end real-shot validation + guard + roadmap checks with raw-ingestion readiness enforcement (`require_disruption_raw_ingestion_ready=true`). | `python tools/run_real_data_strict_gate.py --thresholds tools/real_shot_validation_thresholds_raw_ready.json` |
| `freegs-strict` | FreeGS-only strict backend parity lane with runtime-fallback disallowed and artifact contract checks (`mode=freegs`, no fallback cases). | `python validation/benchmark_vs_freegs.py --strict-backend && python tools/check_freegs_strict_artifact.py --report artifacts/freegs_benchmark.json` |

## CI Mapping

| CI Job | Role | Gate |
|---|---|---|
| `python-tests` | Multi-version core regression lane | `release` |
| `python-research-gate` | Research validation lane (3.12) | `research` |
| `validation-regression` | Cross-language physics validation lane | `release` (`pytest -m "not experimental"`) |
| `strict-real-data` (`real-data-strict.yml`, manual dispatch) | External-data readiness lane; fails when DIII-D raw ingestion contract is unmet (unless dry-run override is set). | `real-data-strict` |
| `freegs-strict` (`freegs-strict.yml`, manual dispatch) | Strict FreeGS backend parity lane; fails on any fallback or non-FreeGS reference backend. | `freegs-strict` |

FreeGS strict-backend parity remains opt-in via
`--enable-freegs-strict-backend-check` (or `SCPN_ENABLE_FREEGS_STRICT_BACKEND_CHECKS=1`)
for the release preflight path, while the dedicated `freegs-strict.yml` workflow
enforces a no-fallback contract when invoked.

## Experimental Marker Contract

Tests marked with `@pytest.mark.experimental` are considered research-only and are excluded from release acceptance runs. As of v3.9.x this includes:

- `tests/test_fno_training.py`
- `tests/test_fno_multi_regime.py`
- `tests/test_gai_01_turbulence_surrogate.py`
- `tests/test_gai_02_torax_hybrid.py`
- `tests/test_gai_03_heat_ml_shadow.py`
- `tests/test_full_validation_pipeline.py`

## Promotion Rule

When a research lane is promoted to release:

1. Remove `@pytest.mark.experimental` from the test module(s).
2. Ensure required dependencies are part of the default CI/runtime contract.
3. Add/update threshold tests proving deterministic pass/fail behavior.
4. Update this matrix and release notes in the same PR.
