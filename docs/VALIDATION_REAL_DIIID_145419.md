<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Fusion Core — real-data Grad-Shafranov validation record (DIII-D shot 145419) -->

# Real-data validation: DIII-D shot 145419 (EFIT reconstruction)

This document records the validation of the SI Grad-Shafranov machinery in
`scpn_fusion.core` against **real DIII-D data**, with every measured number, every failed
attempt, and exact reproduction instructions. Nothing here is synthetic-only or
cherry-picked; the honest negatives are part of the record.

## Reference and provenance

| item | value |
|---|---|
| Reference | `g145419.02100` — DIII-D shot 145419, t = 2100 ms |
| Content | EFIT equilibrium reconstruction, 129×129 grid, Ip = 1.508 MA |
| Source | openly redistributed by General Atomics inside the [`omas`](https://pypi.org/project/omas/) PyPI package (`omas/samples/g145419.02100`) |
| Nature of the reference | EFIT output is a **measurement-constrained model reconstruction**, not a raw measurement. Agreement below is agreement with EFIT, stated as such. |

No facility (MDSplus) access was used; the reference is public. `pip install omas` is
sufficient to obtain it.

## Reproduce

```bash
pip install -e ".[full]"        # or at minimum: pip install omas freeqdsk scipy jax
python validation/validate_real_diiid_145419.py
```

One script run regenerates every number and artefact below into
`artifacts/real_diiid_145419/` (ψ fields as `.npz`, metrics as
`real_145419_validation.json`, including the honest-negatives block).

## Results (three levels)

| level | test | result |
|---|---|---|
| 1 | **Operator satisfaction** (pure evaluation, no solve): does the real ψ with the g-file's own p′/FF′ satisfy our discretised GS operator? | deep-region residual **1.07 × 10⁻³ RMS** (relative to the Δ*ψ scale) |
| 2 | **Coil-free sub-domain reproduction**: Dirichlet = real ψ on an empirically verified coil-free rectangle, our source inside | deep RMS **0.108 %** of the ψ span (deep max 0.25 %, axis 0.06 %) |
| 3 | **Full-domain reproduction** (129², measured-external-source decomposition + Anderson) | deep RMS **0.717 %** of the ψ span (axis 0.28 %, global max 1.33 %) |

"Deep region" is ψ_N < 0.8 inside the confined plasma; the span is |ψ_axis − ψ_boundary|.
Normalisation levels in the reproduction runs are anchored to the reference values
(*reproduction* mode — this validates the operator and source model, it is not a blind
prediction; the blind-prediction lane is the free-boundary predictive solver, validated
separately against a FreeGS reference).

## Honest negatives and the debugging record (kept deliberately)

1. **A naive full-domain re-solve diverges to ≈ 26 % deep RMS.** Root cause, established
   empirically: the 129² g-file domain **contains PF-coil cross-sections** (661 vacuum cells
   with large |Δ*ψ| — inboard F-coils near R ≈ 0.85–0.88 m, divertor coils near
   R ≈ 2.26–2.50 m). A zero-source vacuum model is wrong in those cells and the elliptic
   inverse propagates the mismatch domain-wide. Any re-solve on a facility g-file grid
   should first map this external-current footprint.
2. **A physically corrected map can still converge to a wrong answer.** With the external
   source pinned to the measured Δ*ψ, plain relaxed Picard converges cleanly — small steps,
   no NaN — to an attractor ≈ 127 % away (the H-mode pedestal p″ makes the fixed-point map
   bistable). Anderson acceleration (depth 8) with Ip renormalisation selects the true
   branch in 26 iterations. *"Converged" is not "correct branch"* — fixed points are only
   trusted here against an independent reference.
3. **Rejected hypotheses are recorded, not deleted**: private-flux spurious current
   (measured ≈ 0 MA — not the cause); divertor-leg exclusion (did not change the wrong
   attractor — not the cause).
4. **Residual model boundary**: the 0.72 % (full-domain) vs 0.11 % (sub-domain) gap is the
   separatrix/pedestal-annulus mismatch between our ψ_N < 1 source cutoff and EFIT's edge
   current placement — a genuine fidelity boundary of the current source model, stated, not
   hidden.

## Conventions (stated explicitly)

- The g-file stores ψ descending from axis to boundary; the package convention is
  ψ-peaked-at-axis. The field and the profile derivatives are sign-flipped **together**
  (ψ → −ψ, p′ → −p′, FF′ → −FF′ — an exact GS symmetry). A full COCOS audit against a
  partner convention is a documented follow-up of the IMAS bridge, not silently claimed.
- geqdsk ψ is stored `[R, Z]`; the package uses `(NZ, NR)`. The transposition is covered by
  an orientation test that cannot self-cancel.

## Adjoint gradient validation (companion record)

The implicit-differentiation adjoint of the free-boundary predictive solver is validated
against warm-started central finite differences on its synthetic test case
(`artifacts/coilgrad_adjoint_fd_evidence.json` holds the raw numbers):

| quantity | agreement with FD |
|---|---|
| profile gradients ∂/∂p′, ∂/∂FF′ | < 2 × 10⁻³ (33²), ~2.4 × 10⁻⁵ (65², preconditioned) |
| coil-current gradients | **≈ 7 significant figures** (0.000 %–0.0002 % at 100–300 A steps) |

A historical "coil gradient ≈ 3 %" figure in earlier notes was traced to **finite-difference
truncation error** (a 3 kA step is a ~0.5 % coil perturbation, where the axis-flux response
is visibly nonlinear; central FD is then ~27 % off) — an artefact of the validation
methodology, not of the adjoint. The regression test
`tests/test_jax_free_boundary_predictive.py::test_coil_gradient_matches_finite_difference`
pins the corrected comparison. Methodological rule adopted: sweep the FD step and require
convergence toward the adjoint before recording any FD-based accuracy claim (truncation
error grows with the step; a genuine missing term does not).

## Scope and limitations

- Validation grid: one public shot/time slice (145419 / 2100 ms). Additional shots require
  facility data access under the applicable DOE/GA terms.
- All timings on the development host are load-contaminated and are **not** performance
  claims; performance benchmarking is a separate, dedicated-hardware exercise.
- EFIT-agreement is not measurement-agreement; forward modelling of raw magnetics is out of
  scope here.
