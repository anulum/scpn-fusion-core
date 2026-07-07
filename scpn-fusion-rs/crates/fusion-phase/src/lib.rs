// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Phase-Dynamics Crate Root
//! Deterministic phase-dynamics kernels for the SCPN synchronisation lane:
//! the mean-field Kuramoto-Sakaguchi step and the multi-layer UPDE tick.
//! Both mirror the NumPy reference tier in `scpn_fusion.phase` and are
//! exposed to Python through `fusion-python` for fastest-first dispatch.

pub mod kuramoto;
pub mod upde;

pub use kuramoto::{kuramoto_step, lyapunov_v, order_parameter, wrap_phase, KuramotoStepResult};
pub use upde::{upde_run, upde_tick, UpdeRunResult, UpdeTickResult};
