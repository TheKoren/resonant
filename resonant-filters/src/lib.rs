#![no_std]
#![warn(missing_docs)]

//! `resonant-filters` — FIR/IIR filters and design helpers.
//!
//! Provides stateful filter implementations with accessible internals,
//! suitable for both streaming audio and offline processing.

/// Biquad filter — second-order IIR section (direct form II transposed).
pub mod biquad;

pub use biquad::{Biquad, BiquadCoeffs, BiquadState};
