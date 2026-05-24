# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Third-Party Security Audit Scope

# Third-party security audit scope

This repository still requires an external security review before claiming third-party audit completion. The audit scope should include:

- Rust crates under `scpn-fusion-rs/crates`, including PyO3 interfaces and any `unsafe` blocks.
- Python file loaders for JSON, G-EQDSK, NumPy archives, checkpoint payloads, and controller artefacts.
- Native library loading through `scpn_fusion.hpc.hpc_bridge`, including SHA-256 trust metadata and build sidecars.
- Streamlit dashboard response-header enforcement and deployment assumptions.
- CI dependency audit workflows, release provenance, and package publishing controls.

Required deliverables:

- Written findings with severity, affected path, exploitability, and remediation evidence.
- Re-run evidence for dependency audits, fuzz targets, and focused security regression tests.
- Explicit sign-off for the release tag or a tracked exception list for unresolved findings.

A dependency audit or internal code review is not a substitute for this external review.
