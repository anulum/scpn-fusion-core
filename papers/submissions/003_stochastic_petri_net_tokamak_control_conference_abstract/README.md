<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — extended-abstract review package. -->

# Stochastic Petri net tokamak control extended abstract

Status: venue-neutral review draft; not submitted.

This concise manuscript summarizes the bounded equilibrium and reduced-order
control evidence.  It is not named for, formatted for, or submitted to any
specific venue.

- Source: `manuscript.tex`
- Review PDF: `manuscript.pdf`
- Bibliography: `references.bib`
- Build procedure: `BUILD.md`
- Evidence generator: `generate_evidence_manifest.py`
- Exact evidence and custody manifest: `evidence/`

The package is self-contained: it carries exact byte copies of the three JSON
records cited by the manuscript and records each companion revision, Git blob
identity and SHA-256 digest.  The generator validates the reported scalars,
source custody and implementation-source digests before rebuilding the manifest.

The evidence boundary is deliberately split:

- the production Petri compiler is source-level implementation evidence;
- the reduced-order vertical benchmark independently exercises PID,
  `LQRController` and `SpikingControllerPool`, not a compiled Petri artifact;
- the DIII-D result is a same-case warm-start reproduction, not blind
  prediction;
- the SPARC result is a 16/16 pointwise neural gate, while only 8/16 rows pass
  the adapted source contract; it is not free-boundary or experimental evidence.

No fresh scientific run was performed to assemble this review package.
