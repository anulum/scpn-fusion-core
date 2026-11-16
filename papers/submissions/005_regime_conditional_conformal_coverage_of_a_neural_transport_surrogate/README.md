<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — conformal-coverage review package. -->

# Regime-conditional conformal coverage of a neural transport surrogate

Status: dated preprint of 16 November 2026, describing commit 8191a1d3 on main
after release 4.1.0. The analysis modules are not in that release. The preprint
has not been deposited or submitted.

The manuscript asks whether the 90 % split-conformal certificate of the
repository's neural transport surrogate (trained on the QLKNN-10D database)
holds across transport regimes, and finds that it does not. Standard conformal
variants do not repair the gap, because the surrogate predicts most unstable
rows as stable. Class-conditional stability sets restore per-class coverage at
a cost in interval width. The manuscript is not named for, formatted for, or
submitted to any specific venue.

- Source: `manuscript.tex`
- Review PDF: `manuscript.pdf`
- Bibliography: `references.bib`
- Build procedure: `BUILD.md`
- Evidence generator: `generate_evidence_manifest.py`
- Evidence and custody manifest: `evidence/`

## Evidence boundary

- All coverage is marginal within each class and assumes exchangeability of
  the calibration and test rows. Nothing is claimed outside the database's
  distribution.
- The targets are QuaLiKiz-derived database values, not gyrokinetic or
  measured fluxes.
- The surrogate is the repository's own network, not QLKNN. Its training cost
  is not recorded with its weights, so the bias is not attributed to a training
  choice.
- Two Mondrian-by-prediction variants and the stability sets were designed
  after earlier results. Their settings were fixed before their own runs.
- The profile propagation is a frozen-coefficient sensitivity on one L-mode
  case.

## Custody

Each report records the commit that produced it. Since then the analysis code
changed only by making private helpers public. The variant report, produced
after the first such change, reproduces the regime report's certificate
coverage exactly. The manifest binds the five analysis modules at the package
commit.

## Statements still open before any deposit

The author has not yet decided the following, and this package does not
contain them:

- the manuscript-content licence;
- a DOI;
- the funding and competing-interest statements.
