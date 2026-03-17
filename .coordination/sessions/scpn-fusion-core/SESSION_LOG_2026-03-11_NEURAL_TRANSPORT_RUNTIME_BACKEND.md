# Session Log: 2026-03-11 — Neural Transport Runtime Backend

**Agent:** Codex
**Project:** SCPN-Fusion-Core
**Status:** COMPLETE
**Timestamp:** 2026-03-11T18:35:31+01:00

## Context

- This session continued directly after the March 11 TGLF deck fidelity upgrade.
- The runtime transport solver already had:
  - reduced multichannel analytic closure
  - external live `tglf_live` backend
- The main remaining runtime gap was that the shipped QLKNN-style transport surrogate existed only as a library class and was not selectable as an integrated solver backend.
- Local protocol reminder re-confirmed during this session:
  - mandatory source-file headers stay in place
  - anti-slop rules do not remove required file headers
  - session logs must be comprehensive enough for agent handoff

## What Was Done

- Promoted the shipped neural transport surrogate into a real runtime backend:
  - `physics.transport_backend = "neural_transport"`
  - alias also accepted: `"qlknn"`
- Added lazy backend resolution/caching inside the integrated transport model mixin so the solver only instantiates the neural surrogate when requested.
- Added explicit failure behavior when the neural backend is requested but valid weights are not loaded:
  - missing weights now trigger explicit fallback instead of silently behaving like the reduced analytic lane
  - fallback still lands on the existing `legacy_ti_threshold_fallback` path with recorded error metadata
- Extended the neural transport model runtime telemetry:
  - backend/model name
  - weights path
  - weights checksum
  - input feature dimension
  - layer count
  - gating / gyro-Bohm / log-transform flags
  - coarse channel classification metadata
  - out-of-distribution fractions at `3 sigma` and `5 sigma`
  - max normalized feature excursion (`max_abs_z`)
- Extended integrated solver transport closure contracts so every backend now exposes a consistent top-level record including:
  - `weights_loaded`
  - `weights_path`
  - `weights_checksum`
  - `classification_mode`
  - `ood_fraction_3sigma`
  - `ood_fraction_5sigma`
  - `max_abs_z`
- Tightened the neural profile path itself:
  - validates `a_minor` and `b_toroidal` in neural mode
  - uses runtime `r_major` and `b_toroidal` for derived gyro-Bohm feature construction instead of hard-coded constants in the optional higher-dimensional feature branch
  - validates normalization-stat vector dimensions against the active feature width before inference

## Files Modified

- `src/scpn_fusion/core/neural_transport.py`
- `src/scpn_fusion/core/integrated_transport_solver.py`
- `src/scpn_fusion/core/integrated_transport_solver_model.py`
- `tests/test_neural_transport.py`
- `tests/test_integrated_transport_solver.py`

## Verification

Primary targeted transport verification:

```powershell
python -m pytest 03_CODE/SCPN-Fusion-Core/tests/test_neural_transport.py 03_CODE/SCPN-Fusion-Core/tests/test_integrated_transport_solver.py -q
```

Result:

```text
107 passed in 11.00s
```

Adjacent backend/coupling verification:

```powershell
python -m pytest 03_CODE/SCPN-Fusion-Core/tests/test_tglf_interface.py 03_CODE/SCPN-Fusion-Core/tests/test_tglf_validation_runtime.py 03_CODE/SCPN-Fusion-Core/tests/test_omas_tglf_coupling.py 03_CODE/SCPN-Fusion-Core/tests/test_gs_transport_coupling.py -q
```

Result:

```text
77 passed, 1 skipped in 11.51s
```

## Resulting Runtime State

- Integrated transport runtime backends now include:
  - reduced analytic multichannel closure
  - live TGLF profile scan backend
  - live neural QLKNN-style surrogate backend
- Neural backend contracts are now explicit about whether the gyrokinetic surrogate was truly active or whether the solver had to fall back.
- The runtime no longer hides missing neural weights behind an implicit reduced-model path.

## Remaining High-Value Gaps

- The neural backend is still a profile-level surrogate, not a full first-principles nonlinear gyrokinetic solver.
- The coarse channel classification in neural mode is honest but limited:
  - it distinguishes ion-dominant vs electron-dominant transport
  - it does not separate TEM from ETG in the surrogate outputs because the shipped inference outputs are aggregate `chi_e`, `chi_i`, `D_e`
- Highest-value next steps from here:
  - add runtime routing that escalates OOD neural points to live TGLF instead of only reporting OOD telemetry
  - expose a hybrid backend such as `transport_backend = "neural_transport_hybrid"` or similar
  - promote more shaping/equilibrium features into the surrogate inference contract where supported by training data
  - add a second external ingestion lane for higher-fidelity gyrokinetic outputs beyond TGLF where the repo has data/contracts ready

## Commit Scope Guidance

- This batch is intentionally narrow because the worktree is globally dirty.
- Only stage the five transport files above plus this in-repo session log when committing.
- Do not push yet; this is a local incremental backend promotion step.
