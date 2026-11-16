<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — TORAX model-intersection review package. -->

# Coupled transport model intersection with TORAX

Status: dated preprint of 16 November 2026, describing commit 1445bb14 on main
after release 4.1.0. The preprint has not been deposited or submitted.

The manuscript checks SCPN-Fusion-Core's four-state coupled transport step
(ion and electron temperature, electron density, poloidal flux) against
TORAX 1.4.3 on a declared model intersection, and checks the gradients of a
JAX version of the same step. It is not named for, formatted for, or
submitted to any specific venue.

- Source: `manuscript.tex`
- Review PDF: `manuscript.pdf`
- Bibliography: `references.bib`
- Build procedure: `BUILD.md`
- Evidence generator: `generate_evidence_manifest.py`
- Evidence and custody manifest: `evidence/`

## Evidence boundary

- The intersection is narrow: circular geometry, constant coefficients,
  Gaussian sources, collisional exchange, prescribed edge values.
- The case covers 20 ms in two steps of 10 ms on 16 cells.
- The bounded optimisation lowers its objective but does not identify the
  three source scales.
- No TORAX equivalence outside the intersection, no experimental data and no
  speed comparison.
- `validation/reports/torax_runtime_contract.json` is not cited: code it
  records changed after it was produced.

## Custody

The parity report (849b9b2c) and the gradient report (d84af539) record source
digests. Every one of them matches the package commit, so the manifest binds
those eleven sources at 1445bb14.

## Statements still open before any deposit

The author has not yet decided the following, and this package does not
contain them:

- the manuscript-content licence;
- a DOI;
- the funding and competing-interest statements.
