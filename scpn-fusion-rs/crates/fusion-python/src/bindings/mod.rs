// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
//! Per-domain PyO3 binding modules.
//!
//! Each submodule owns the `#[pyfunction]`/`#[pyclass]` bindings for one
//! responsibility of the SCPN Fusion Core Rust workspace; the crate root
//! (`lib.rs`) declares the `#[pymodule]` and registers the exported items.

pub(crate) mod control;
pub(crate) mod diagnostics;
pub(crate) mod mhd;
pub(crate) mod neural;
pub(crate) mod nuclear;
pub(crate) mod particles;
pub(crate) mod phase;
