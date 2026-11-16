<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — same-case FreeGSNKE comparison review package. -->

# Same-case free-boundary comparison with FreeGSNKE

Status: dated preprint of 16 November 2026, describing commit
9e2ea47c on main after release 4.1.0. The comparison code is not in
that release. The preprint has not been deposited or submitted.

The manuscript compares SCPN-Fusion-Core's predictive free-boundary solver with
FreeGSNKE 3.0.1 on the three static inverse problems that FreeGSNKE ships:

- forward, from FreeGSNKE's coil currents and profile;
- inverse, fitting coil currents to FreeGSNKE's isoflux and null-point targets,
  with each code's currents scored by both codes.

The manuscript is not named for, formatted for, or submitted to any specific
venue.

- Source: `manuscript.tex`
- Review PDF: `manuscript.pdf`
- Bibliography: `references.bib`
- Build procedure: `BUILD.md`
- Evidence generator: `generate_evidence_manifest.py`
- Evidence and custody manifest: `evidence/`

## Evidence boundary

- All three machine descriptions are synthetic configurations from the
  FreeGSNKE repository. No measured discharge enters the comparison.
- The forward agreement is a warm-start result. SCPN's cold start does not
  reach the diverted solution on the MAST-U-like case.
- The inverse fits use one perturbation seed per case. Neither code meets the
  ITER-like targets, which overdetermine the circuits.
- Coil currents are compared through target residuals under both codes. Where
  the targets leave current combinations free, as on the SPARC-like case,
  current sets 12.7 % apart both meet them.
- Grid convergence is checked on the MAST-U-like case only, at 65×129 and
  129×257: the frozen-support difference converges, the warm one barely moves.
- Run times are recorded only to describe the runs. No speed comparison is made.

## Custody

The two reports were produced at different commits:

- the forward report at 422e4c0b;
- the grid-convergence report at 220cd790;
- the inverse report from six runs at 44852fee, merged and tracked at
  86649a05. Its status is fail: on the MAST-U-like perturbed start,
  FreeGSNKE's refinement at SCPN's fitted currents did not converge.

Between them no forward-solver source changed.
`validation/benchmark_freegsnke_inverse.py` changed only by naming two constant
tables (the MAST-U-like current limits and Tikhonov weights), with the same
values. The custody manifest binds the solver and benchmark sources at the
package commit. The inverse report records the same digests for the sources it
used.

## Statements still open before any deposit

The author has not yet decided the following, and this package does not
contain them:

- the manuscript-content licence;
- a DOI;
- the funding and competing-interest statements.
