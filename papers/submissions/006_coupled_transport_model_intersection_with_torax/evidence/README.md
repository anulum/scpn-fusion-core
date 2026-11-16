<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN-FUSION-CORE — TORAX model-intersection paper evidence. -->

# Evidence

The three JSON files here are tracked files of the repository:

- the parity report `validation/reports/torax_coupled_transport_parity.json`;
- the gradient report `validation/reports/torax_differentiable_transport.json`;
- the TORAX reference run
  `validation/reference_data/torax/torax_coupled_transport_reference.json`.

Each copy has the seven structured public header fields placed first; the
scientific payload is unchanged. The generated `evidence_manifest.json`
records each file's source commit, Git blob identity, SHA-256 digest and role,
and binds the deck, the runner, the benchmarks and the solver modules.

`../generate_evidence_manifest.py` rebuilds the copies from the tracked files,
recomputes every number the manuscript quotes, and fails closed when one prints
differently.

The files concern one declared model intersection over 20 ms. They establish
no TORAX equivalence outside it, no experimental validity and no speed
comparison.
