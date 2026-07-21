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
attempt, and exact reproduction instructions. The honest negatives are part of the record.

**Selection protocol** (stated, since a single shot cannot demonstrate the absence of
selection): shot 145419 / t = 2100 ms is the **only** real DIII-D equilibrium openly
redistributed in the `omas` package and hence the only real reference available to this
project without facility data access — it was used because it is the one that exists, not
chosen among candidates. Extending the validation grid requires facility access under the
applicable DOE/GA terms (in progress via the collaboration channel).

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
# exact reproduction (hash-pinned environment):
pip install -r requirements/full.txt && pip install -e . --no-deps
# or, quick-start (unpinned — versions may drift):
pip install -e ".[full]"        # at minimum: pip install omas freeqdsk scipy jax
python validation/validate_real_diiid_145419.py
```

One script run regenerates every number and artefact below into
`artifacts/real_diiid_145419/` (ψ fields as `.npz`, metrics as
`real_145419_validation.json`). That includes the **honest negatives — they are executable
lanes rerun on every invocation**, not archived constants — and a runtime provenance block
(reference SHA-256 + exact package versions of the run that produced the tracked JSON).

## Results (three levels)

| level | test | result |
|---|---|---|
| 1 | **Operator satisfaction** (pure evaluation, no solve): does the real ψ with the g-file's own p′/FF′ satisfy our discretised GS operator? | deep-region residual **1.07 × 10⁻³ RMS** (relative to the Δ*ψ scale) |
| 2 | **Coil-free sub-domain reproduction**: Dirichlet = real ψ on an empirically verified coil-free rectangle, our source inside | deep RMS **0.108 %** of the ψ span (deep max 0.25 %, axis 0.06 %) |
| 3 | **Full-domain reproduction** (129², measured-external-source decomposition, Anderson(m=8), warm start) | deep RMS **0.717 %** of the ψ span (axis 0.28 %, global max 1.33 %, 26 iterations) |
| 3b | **Cold start** (external-source-only field, zero plasma current) | lands in the **zero-plasma absorbing state** (deep RMS ≈ 127 %) — see the map-structure paragraph below |
| 3c | **Relaxed Picard, warm start** (ω = 0.5 and ω = 0.3, both executed every run) | reaches the **true branch** (deep RMS 0.63 % after 200 iterations, no early stop) — Anderson contributes acceleration (26 vs > 200 iterations), not branch selection |

"Deep region" is ψ_N < 0.8 inside the confined plasma; the span is |ψ_axis − ψ_boundary|.

**Disclosed prominently**: the reproduction lanes (2, 3, 3c) are **warm-started from the
EFIT ψ itself** and anchor their normalisation (ψ_axis, ψ_bnd, plasma-region Ip) to
reference values — they are fixed-point *consistency/reproduction* checks of the operator
and source model, **not** blind prediction and **not** independent branch selection. The
blind-prediction lane is the free-boundary predictive solver, validated separately against
a FreeGS reference from a genuine vacuum cold start; running it on this real shot would
require the PF coil currents, which g-files do not carry (that is the Rung-4 facility-data
gap, stated below).

**Fixed-point structure of the reproduction map** (measured 2026-07-22, regenerated on
every run): the map has two measured fixed points. Warm starts reach the true branch under
both Anderson and plain relaxed Picard. A plasma-free start cannot leave the **zero-plasma
absorbing state**: the ψ_N anchors are fixed reference values, so with no plasma current the
tanh LCFS cutoff is exactly saturated, zero model current flows, and the map is exactly
stationary (the cross-checks — Anderson at n_iter = 400 and Picard — are executed and
persisted on every run, not asserted from memory).

## Honest negatives and the debugging record (kept deliberately)

All negative lanes are **executable and regenerated on every script run** (the tracked JSON
holds the exact settings next to each number); development-time observations that they
supersede are kept as labelled archive entries, never silently replaced.

1. **Pretending the external-current cells are vacuum breaks the full domain.** The 129²
   g-file domain **contains PF-coil cross-sections** (661 vacuum cells with large |Δ*ψ| —
   inboard F-coils near R ≈ 0.85–0.88 m, divertor coils near R ≈ 2.26–2.50 m). The
   regenerated zero-external-source lane (Anderson, warm start) measures deep RMS **5.4 %**
   with a global max of **99 %** of the span; the first development observation of this
   failure mode (different, unaccelerated settings) measured ≈ 26 % deep RMS. Any re-solve
   on a facility g-file grid should first map this external-current footprint.
2. **A cleanly stationary iteration can still be 127 % wrong.** The development record noted
   a "relaxed Picard → ≈ 127 % wrong attractor" and attributed it to H-mode pedestal p″
   bistability. The regenerated experiments **retire that interpretation**: warm-started
   Picard reaches the true branch (result 3c), and the ≈ 127 % state is the **zero-plasma
   absorbing fixed point** (result 3b — the archived 1.27 matches it to three digits). The
   lesson stands in sharpened form: *"converged" is not "correct branch"* — a fixed point is
   only trusted against an independent reference — and the corrected identification of the
   wrong attractor is itself part of the record (data kept, reading corrected 2026-07-22).
3. **Rejected hypotheses are recorded, not deleted**: private-flux spurious current
   (measured ≈ 0 MA — not the cause); divertor-leg exclusion (did not change the wrong
   attractor — not the cause); H-mode pedestal p″ bistability (superseded by the absorbing-
   state identification above).
4. **Residual model boundary**: the 0.72 % (full-domain) vs 0.11 % (sub-domain) gap is the
   separatrix/pedestal-annulus mismatch between our ψ_N < 1 source cutoff and EFIT's edge
   current placement — a genuine fidelity boundary of the current source model, stated, not
   hidden.

## Conventions (stated explicitly)

- The g-file stores ψ descending from axis to boundary; the package convention is
  ψ-peaked-at-axis. The field and the profile derivatives are sign-flipped **together**
  (ψ → −ψ, p′ → −p′, FF′ → −FF′ — an exact GS symmetry).
- **COCOS audit (resolved)**: the solvers' native frame was derived and verified as
  **COCOS 3** — the Green's function carries ``1/2π`` over the Maxwell mutual-inductance
  form (ψ is flux *per radian*), and ``Ip > 0`` yields ψ peaked at the axis (σ_Bp = −1),
  matching the Sauter table entry shipped inside the `omas` package. The IMAS bridge
  (`scpn_fusion.core.imas_equilibrium_io`) therefore writes and reads through OMAS's own
  COCOS machinery (`omas_environment(cocosio=3)` → internal COCOS 11; measured transform
  ψ ↦ −2π·ψ), with the φ-handedness — unobservable to an axisymmetric 2-D solver —
  exposed as a parameter. Pinned by dedicated tests
  (`tests/test_imas_equilibrium_io.py`, COCOS section).
- geqdsk ψ is stored `[R, Z]`; the package uses `(NZ, NR)`. The transposition is covered by
  an orientation test that cannot self-cancel.

## Adjoint gradient validation (companion record)

The implicit-differentiation adjoint of the free-boundary predictive solver is validated
against warm-started central finite differences on its synthetic test case. The evidence
JSON (`artifacts/coilgrad_adjoint_fd_evidence.json`) is produced by a **committed,
re-runnable generator** (`validation/measure_coilgrad_adjoint_fd.py` — full-precision
values, strong and weak coil, 100 A–3 kA step sweep, hash-pinned environment) and its
`guarded_claim` field is derived from the worst small-step row of the actual run:

| quantity | agreement with FD (measured, this artefact) |
|---|---|
| profile gradients ∂/∂p′, ∂/∂FF′ | < 2 × 10⁻³ (33²), ~2.4 × 10⁻⁵ (65², preconditioned) |
| coil-current gradients | **≤ 1.8 × 10⁻⁵ relative (≈ 5 significant digits)** at 100–300 A steps, on both the tested strong and weak coil |

A historical "coil gradient ≈ 3 %" figure in earlier notes was traced to **finite-difference
truncation error** (a 3 kA step is a ~0.5 % coil perturbation, where the axis-flux response
is visibly nonlinear; central FD is then ~16–27 % off — the growing-with-step signature is
part of the committed sweep). This was an artefact of the validation methodology, not of
the adjoint; note the residual small-step disagreement (~10⁻⁵) mixes FD truncation with
finite forward-solve convergence, so it is an upper bound on the adjoint error, stated as
the guarded claim rather than extrapolated to "exact". The regression test
`tests/test_jax_free_boundary_predictive.py::test_coil_gradient_matches_finite_difference`
pins the comparison at eps = 300 A (rel < 10⁻³). Methodological rule adopted: sweep the FD
step and require convergence toward the adjoint before recording any FD-based accuracy
claim (truncation error grows with the step; a genuine missing term does not).

## Scope and limitations

- Validation grid: one public shot/time slice (145419 / 2100 ms). Additional shots require
  facility data access under the applicable DOE/GA terms.
- All timings on the development host are load-contaminated and are **not** performance
  claims; performance benchmarking is a separate, dedicated-hardware exercise.
- EFIT-agreement is not measurement-agreement; forward modelling of raw magnetics is out of
  scope here.
