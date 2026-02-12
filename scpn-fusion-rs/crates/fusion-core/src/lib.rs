//! Grad-Shafranov kernel and equilibrium solver.
//!
//! Stage 3: core kernel modules
//! Stage 4: ignition, transport, stability, RF heating

pub mod bfield;
pub mod ignition;
pub mod inverse;
pub mod jacobian;
pub mod kernel;
pub mod rf_heating;
pub mod source;
pub mod stability;
pub mod transport;
pub mod vacuum;
pub mod xpoint;
