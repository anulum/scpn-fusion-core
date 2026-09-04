<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — equilibrium-solver review package. -->

# Hybrid Rust–Python Grad–Shafranov equilibrium solver

Status: evidence-bounded review draft; not submitted.

The manuscript describes implemented equilibrium-solver components and the
strongest currently packaged real-data result.  It distinguishes a warm-start,
same-case DIII-D reproduction from blind prediction and retains the cold-start
failure as part of the result.

The second quantitative layer is a bounded SPARC pointwise gate: all 16 gated
rows use the neural backend and pass the declared NRMSE threshold, but only 8 of
16 pass the adapted source-convention contract.  Neither layer establishes
blind prediction, production free-boundary admission or experimental accuracy.

- Source: `manuscript.tex`
- Review PDF: `manuscript.pdf`
- Bibliography: `references.bib`
- Reproducible figures: `figures/`
- Frozen evidence: `evidence/`
- Build procedure: `BUILD.md`

The manufactured figures are illustrations, not solver executions.
Quantitative values come only from the packaged JSON evidence and are not a
claim of production-solver parity.  `evidence/evidence_manifest.json` records
exact custody, source and generator hashes, and the public claim boundary.
