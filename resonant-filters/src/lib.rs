#![no_std]
#![warn(missing_docs)]

//! `resonant-filters` — FIR/IIR filters and design helpers.
//!
//! Provides stateful filter implementations with accessible internals,
//! suitable for both streaming audio and offline processing.
//!
//! The [`biquad`] module is always available (`no_alloc`). The [`fir`]
//! module requires the `alloc` feature (enabled by default).

/// Biquad filter — second-order IIR section (direct form II transposed).
pub mod biquad;

#[cfg(feature = "alloc")]
/// FIR filter with arbitrary coefficient slice.
pub mod fir;

/// Filter design helpers — Butterworth lowpass/highpass coefficient computation.
pub mod design;

pub use biquad::{Biquad, BiquadCoeffs, BiquadState};
#[cfg(feature = "alloc")]
pub use fir::Fir;
