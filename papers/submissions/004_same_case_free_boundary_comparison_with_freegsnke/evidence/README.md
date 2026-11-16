<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — same-case FreeGSNKE comparison evidence. -->

# Evidence

The two JSON reports here are the tracked reports
`validation/reports/freegsnke_multimachine_parity.json` and
`validation/reports/freegsnke_inverse_fit.json`. Each copy has the seven
structured public header fields placed first; the scientific payload is
unchanged. The generated `evidence_manifest.json` records:

- each report's source commit;
- its Git blob identity and SHA-256 digest;
- its role in the paper;
- the solver and benchmark sources the manuscript describes.

`../generate_evidence_manifest.py` rebuilds both copies from the tracked
reports. It checks every value that the manuscript quotes, and it fails closed
when a report, a quoted value or a bound source differs from the record.

The reports compare two codes on three synthetic machine descriptions shipped
with FreeGSNKE 3.0.1. They do not compare either code with a measured
discharge. They make no facility, control, safety or speed claim.
