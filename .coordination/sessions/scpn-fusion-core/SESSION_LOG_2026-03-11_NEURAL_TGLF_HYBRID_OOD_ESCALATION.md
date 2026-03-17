# Session Log: 2026-03-11 — Neural/TGLF Hybrid OOD Escalation

**Agent:** Codex
**Project:** SCPN-Fusion-Core
**Status:** COMPLETE
**Timestamp:** 2026-03-11T18:41:57+01:00

## Context

- This session started from the freshly committed neural transport runtime backend:
  - commit: `bd0ca98`
  - message: `feat: promote neural transport runtime backend`
- That first batch made the shipped QLKNN-style surrogate selectable in the integrated solver and recorded OOD telemetry, but it still only reported out-of-distribution states.
- The next physics gap was to escalate those OOD regions to the higher-fidelity live TGLF lane instead of merely logging them.

## What Was Done

- Added a new hybrid runtime backend:
  - `physics.transport_backend = "neural_transport_hybrid"`
  - alias also accepted: `"qlknn_tglf_hybrid"`
- Hybrid runtime behavior:
  - run the neural transport surrogate over the full profile
  - compute per-surface normalized feature excursion (`max_abs_z`)
  - identify OOD points above a configurable sigma threshold
  - select the highest-severity interior OOD points
  - run live TGLF only on those selected surfaces
  - overwrite the OOD region with TGLF-derived transport before the solver advances
- Added runtime configuration for the hybrid path:
  - `physics.neural_transport_tglf_ood_sigma`
  - `physics.neural_transport_tglf_max_points`
- Extended neural transport telemetry so the solver can drive hybrid escalation:
  - stored per-point `max_abs_z` profile
  - stored boolean OOD masks at `3 sigma` and `5 sigma`
  - recorded OOD point counts in the surrogate contract
- Added honest coarse post-hybrid channel summarization:
  - still only ion-dominant vs electron-dominant at the aggregate surrogate level
  - no fake ETG/TEM split added where the model does not provide it
- Extended integrated transport closure contracts with hybrid-specific recovery metadata:
  - `ood_point_count`
  - `tglf_sample_count`
  - `tglf_sample_indices`
  - `gamma_profile_mean`
  - `classification_mode = "hybrid_neural_tglf_ood_escalation"`

## Files Modified

- `src/scpn_fusion/core/neural_transport.py`
- `src/scpn_fusion/core/integrated_transport_solver.py`
- `src/scpn_fusion/core/integrated_transport_solver_model.py`
- `tests/test_neural_transport.py`
- `tests/test_integrated_transport_solver.py`

## Verification

Primary transport verification:

```powershell
python -m pytest 03_CODE/SCPN-Fusion-Core/tests/test_neural_transport.py 03_CODE/SCPN-Fusion-Core/tests/test_integrated_transport_solver.py -q
```

Result:

```text
108 passed in 10.33s
```

Adjacent TGLF/coupling verification:

```powershell
python -m pytest 03_CODE/SCPN-Fusion-Core/tests/test_tglf_interface.py 03_CODE/SCPN-Fusion-Core/tests/test_tglf_validation_runtime.py 03_CODE/SCPN-Fusion-Core/tests/test_omas_tglf_coupling.py 03_CODE/SCPN-Fusion-Core/tests/test_gs_transport_coupling.py -q
```

Result:

```text
77 passed, 1 skipped in 9.48s
```

## Resulting Runtime State

- Integrated transport now has four practical runtime modes:
  - reduced analytic multichannel closure
  - neural transport surrogate
  - live TGLF profile scan
  - neural/TGLF hybrid OOD escalation
- The hybrid lane now turns neural OOD detection into active higher-fidelity repair instead of passive telemetry.
- This is still not a full nonlinear gyrokinetic solver in the transport loop, but it is materially closer to an elite “surrogate where valid, first-principles where needed” runtime architecture.

## Remaining High-Value Gaps

- Hybrid escalation currently uses a coarse severity ranking over `max_abs_z`; it does not yet cluster OOD regions by mode physics.
- Hybrid replacement currently patches the OOD region with interpolated TGLF profiles from selected sampled points; it does not yet adaptively refine until convergence of the transport response.
- Highest-value next steps from here:
  - add adaptive second-pass sampling when hybrid replacement still leaves large residual OOD pockets
  - incorporate more shaping/current-gradient/equilibrium-derived features into the neural input contract where training support exists
  - add explicit hybrid validation benchmarks against the in-tree TGLF reference suite and power-balance benchmarks
  - evaluate whether the repo’s CGYRO/GENE-facing assets are mature enough for a second external escalation lane

## Commit Scope Guidance

- This is the second narrow transport-runtime batch on March 11.
- Stage only the five transport files above plus this in-repo session log for the commit.
- Do not push yet.
