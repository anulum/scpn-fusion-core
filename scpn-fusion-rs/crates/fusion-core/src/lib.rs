//! Grad-Shafranov kernel and equilibrium solver.
//!
//! Stage 3: core kernel modules
//! Stage 4: ignition, transport, stability, RF heating

pub mod amr_kernel;
pub mod bfield;
pub mod bout_interface;
pub mod ignition;
pub mod inverse;
pub mod jacobian;
pub mod jit;
pub mod kernel;
pub mod memory_transport;
pub mod mpi_domain;
pub mod particles;
pub mod pedestal;
pub mod rf_heating;
pub mod source;
pub mod stability;
pub mod transport;
pub mod vacuum;
pub mod vmec_interface;
pub mod xpoint;
