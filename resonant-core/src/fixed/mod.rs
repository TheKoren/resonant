//! Fixed-point arithmetic types for deterministic, FPU-free DSP.
//!
//! These types map a signed integer range onto the real interval \[-1.0, 1.0).
//! They are useful on targets without hardware floating-point or where
//! bit-exact reproducibility is required.

mod q15;

pub use q15::Q15;
