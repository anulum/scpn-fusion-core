# Validation Gate Matrix

This document defines the split between release-grade validation and research-only validation for SCPN Fusion Core.

## Why this exists

- Keep release CI lanes focused on deterministic, production-facing checks.
- Preserve visibility on experimental/research lanes without letting them silently contaminate release acceptance.
- Make gate intent explicit in both local preflight and GitHub Actions.

## Gate Profiles

| Profile | Scope | Command |
|---|---|---|
| `release` | Version/claims integrity, disruption data provenance + split + calibration holdout checks, disruption replay pipeline contract benchmark, EPED domain-contract benchmark, transport uncertainty-envelope benchmark, multi-ion transport conservation benchmark, end-to-end latency benchmark, notebook quality gate, Task 5/6 threshold smoke, strict typing; excludes experimental tests from global pytest runs. | `python tools/run_python_preflight.py --gate release` |
| `research` | Experimental-only pytest lane (`pytest -m experimental`). | `python tools/run_python_preflight.py --gate research` |
| `all` | Release + research profiles in sequence. | `python tools/run_python_preflight.py --gate all` |

## CI Mapping

| CI Job | Role | Gate |
|---|---|---|
| `python-tests` | Multi-version core regression lane | `release` |
| `python-research-gate` | Experimental validation lane (3.12) | `research` |
| `validation-regression` | Cross-language physics validation lane | `release` (`pytest -m "not experimental"`) |

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
