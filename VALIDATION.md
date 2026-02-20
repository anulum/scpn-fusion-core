# Golden Base Validation Notes

This document covers data provenance and validation notes for:

- `examples/neuro_symbolic_control_demo_v2.ipynb` (active release)
- `examples/neuro_symbolic_control_demo.ipynb` (legacy frozen published base)

## DIII-D Shot Provenance

- The notebook uses:
  - `validation/reference_data/diiid/disruption_shots/shot_166000_beta_limit.npz`
- In this repository, files in `validation/reference_data/diiid/disruption_shots/` are synthetic DIII-D-like profiles generated from physics-informed templates, not raw downloaded EFIT/MDSplus dumps.
- Generation source:
  - `tools/generate_disruption_profiles.py`
  - The script defines a deterministic manifest of disruption/safe scenarios and seeds each shot with `np.random.default_rng(shot_number)` for reproducibility.
- Provenance manifest (versioned, hash-locked):
  - `validation/reference_data/diiid/disruption_shots_manifest.json`
  - Generated/verified by `tools/generate_disruption_shot_manifest.py` with per-file `sha256`, byte size, scenario label, source type, and license fields.
- Split hygiene contract:
  - `validation/reference_data/diiid/disruption_shot_splits.json`
  - Validated by `tools/check_disruption_shot_splits.py` (no train/val/test overlap, no missing manifest shots).
- Calibration contract (train/val selection + holdout gate):
  - `validation/reference_data/diiid/disruption_risk_calibration.json`
  - `validation/reports/disruption_risk_holdout_report.md`
  - Generated/checked by `tools/generate_disruption_risk_calibration.py`.
- EPED domain-validity contract (bounded extrapolation + report gate):
  - `validation/reports/eped_domain_contract_benchmark.json`
  - `validation/reports/eped_domain_contract_benchmark.md`
  - Generated/checked by `validation/benchmark_eped_domain_contract.py`.
- End-to-end control latency contract (surrogate vs full-physics mode):
  - `validation/reports/scpn_end_to_end_latency.json`
  - `validation/reports/scpn_end_to_end_latency.md`
  - Generated/checked by `validation/scpn_end_to_end_latency.py`.

## Validation Script Linkage

The Golden Base notebook disturbance lane is intentionally linked to existing validation workflows:

- `validation/validate_real_shots.py`
  - Disruption validation lane reads from `validation/reference_data/diiid/disruption_shots/`
- `validation/full_validation_pipeline.py`
  - Multi-lane empirical validation runner

In notebook code this appears as `VALIDATION_SCRIPT` and `FULL_PIPELINE_SCRIPT` path checks, and `load_shot(...)` points at the same reference-data directory.

## Task-5 Proxy Hardening (v3.5.0)

- `src/scpn_fusion/control/disruption_contracts.py` now exposes uncertainty-aware outputs for disruption-lane surrogates:
  - `mcnp_lite_tbr(..., return_uncertainty=True)` returns `tbr_sigma`, `tbr_rel_sigma`, and p95 bounds.
  - `run_disruption_episode()` returns p95 bounds for risk, wall damage, and TBR plus a scalar `uncertainty_envelope`.
- The synthetic disturbance signal remains a contract-test surrogate:
  - stochastic and deterministic-replay friendly,
  - bounded to a documented disturbance domain,
  - not a direct substitute for facility-grade real-shot waveform reconstruction.

## If The Shot File Is Missing

Generate or regenerate the bundled disruption profiles from repo root:

```powershell
python tools/generate_disruption_profiles.py
```

Optional explicit output directory:

```powershell
python tools/generate_disruption_profiles.py --output-dir validation/reference_data/diiid/disruption_shots
```

Verification-only pass:

```powershell
python tools/generate_disruption_profiles.py --verify-only
```

Regenerate or verify provenance manifest:

```powershell
python tools/generate_disruption_shot_manifest.py
python tools/generate_disruption_shot_manifest.py --check
python tools/check_disruption_shot_splits.py
python tools/generate_disruption_risk_calibration.py
python tools/generate_disruption_risk_calibration.py --check
python validation/benchmark_eped_domain_contract.py --strict
python validation/scpn_end_to_end_latency.py --strict
```

## Data License And Copyright Context

- Bundled `disruption_shots/*.npz` in this repo are generated synthetic reference profiles and are distributed with this codebase workflow.
- Real DIII-D experimental data are not bundled in this specific `disruption_shots` folder.
- DIII-D itself is a U.S. DOE Office of Science user facility operated by General Atomics; real-shot data access and reuse follow facility/data-access terms and publication rules.
- If replacing synthetic files with true DIII-D data, document:
  - acquisition source,
  - shot list,
  - processing steps,
  - applicable data rights/attribution terms.

## SPARC / EFIT Replay Note

- Golden Base currently demonstrates disturbance replay against DIII-D-like reference profiles in repo.
- SPARC EFIT replay and broader equilibrium replay workflows are handled by the validation stack (e.g. `validation/validate_real_equilibria.py`) and can be integrated into notebook control loops as a follow-up step.

## Copyright Clarity

- Concepts: Copyright 1996-2026
- Code: Copyright 2024-2026
