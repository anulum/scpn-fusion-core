<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — runaway kinetic Rust-kernel review package. -->

# Runaway kinetic operator with an equivalent Rust kernel

Status: dated preprint of 16 November 2026, describing commit 8bfedda5 on main
after release 4.1.0. The preprint has not been deposited or submitted.

The manuscript shows that the NumPy and Rust implementations of the
repository's finite-volume runaway-electron kinetic operator are equivalent:
- the distribution, every tendency and the runaway density are bit-identical;
- the moments agree to rounding.

It also reports a host-conditioned timing. It is not named for, formatted for,
or submitted to any specific venue.

- Source: `manuscript.tex`
- Review PDF: `manuscript.pdf`
- Bibliography: `references.bib`
- Build procedure: `BUILD.md`
- Evidence generator: `generate_evidence_manifest.py`
- Evidence and custody manifest: `evidence/`

## Evidence boundary

- The parity case has synthetic, index-varying coefficients; it is not a
  physical runaway scenario.
- No comparison with DREAM is made.
- The timing comes from one loaded workstation and one small case.

## Custody

The benchmark report was produced at 98b6d4d2 and tracked at d8a10e8f. The
twelve sources it records are byte-identical at 8bfedda5, where the manifest
binds them.

## Statements still open before any deposit

The author has not yet decided the following, and this package does not
contain them:

- the manuscript-content licence;
- a DOI;
- the funding and competing-interest statements.
