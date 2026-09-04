<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — extended-abstract exact evidence. -->

# Evidence

This directory makes the review package independent of companion-directory
layout while preserving exact custody.  The three scientific JSON files are
byte-for-byte copies from submissions `001` and `002`.  The generated manifest
records their source commits, Git blob identities, SHA-256 digests and roles.

`generate_evidence_manifest.py` verifies the reported results and the exact
implementation sources used to describe the Petri compiler and the independent
vertical controller.  It fails closed if evidence, source code or claim-critical
values drift.

The files do not establish experimental plasma control, hardware performance,
blind DIII-D prediction, free-boundary SPARC prediction, or an end-to-end
Petri-compiled vertical-control run.
