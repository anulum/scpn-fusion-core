<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — linear reduced-MHD review package. -->

# Linear resistive reduced-MHD eigenvalue solver

Status: dated preprint of 16 November 2026, describing commit f7a367a1 on main
after release 4.1.0. The preprint has not been deposited or submitted.

The manuscript tests the repository's linear resistive reduced-MHD eigenvalue
solver (Harris slab and periodic cylinder) against nine acceptance criteria
fixed before the run. Seven pass. The two misses are reported as misses.
It is not named for, formatted for, or submitted to any specific venue.

- Source: `manuscript.tex`
- Review PDF: `manuscript.pdf`
- Bibliography: `references.bib`
- Build procedure: `BUILD.md`
- Evidence generator: `generate_evidence_manifest.py`
- Evidence and custody manifest: `evidence/`

## Evidence boundary

- Slab and periodic-cylinder geometry only. There are no toroidal 1/R terms,
  no Grad–Shafranov equilibrium and no nonlinear physics.
- Two criteria are missed:
  - the cylinder 2/1 tearing slope (0.554);
  - the root of the Δ′ < 0 Harris sheet (7.9e-9 against a 1e-9 limit).

  Their causes are not established.
- The run time of the report describes one host.

## Custody

The acceptance report was run at c1bc7f63 and tracked at 44852fee. The five
sources it records are byte-identical at f7a367a1, where the manifest binds
them.

## Statements still open before any deposit

The author has not yet decided the following, and this package does not
contain them:

- the manuscript-content licence;
- a DOI;
- the funding and competing-interest statements.
