<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — conformal-coverage paper evidence. -->

# Evidence

The six JSON files here are tracked files of the repository:

- the certificate `validation/surrogate_conformal_certificates.json`;
- four reports under `validation/reports/qlknn_conformal_*.json`;
- the surrogate's training record `weights/neural_transport_qlknn.metrics.json`.

Each copy has the seven structured public header fields placed first; the
scientific payload is unchanged. The generated `evidence_manifest.json`
records each file's source commit, Git blob identity, SHA-256 digest and role,
and binds the five analysis modules.

`../generate_evidence_manifest.py` rebuilds the copies from the tracked
files. It recomputes every number the manuscript quotes, the derived
nonzero-flux coverages included, and it fails closed when one prints
differently.

The files concern one surrogate on one database. They give no guarantee
outside that database's distribution and make no claim about gyrokinetic or
measured fluxes.
