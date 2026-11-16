<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — linear reduced-MHD paper evidence. -->

# Evidence

`linear_mhd_reduced_model_acceptance.json` is the tracked report
`validation/reports/linear_mhd_reduced_model_acceptance.json`, and
`linear_mhd_miss_study.json` is the tracked report
`validation/reports/linear_mhd_miss_study.json` of the follow-up study. Each
copy has the seven structured public header fields placed first; the
scientific payload is unchanged. The generated `evidence_manifest.json`
records each report's source commit, Git blob identity, SHA-256 digest and
role, and binds the six sources the two reports record.

`../generate_evidence_manifest.py` rebuilds the copies from the tracked reports,
recomputes every number the manuscript quotes, including each criterion's pass
or miss, and fails closed when one prints differently.

The reports concern a slab and a periodic cylinder. It establishes nothing
about toroidal, shaped or nonlinear MHD.
