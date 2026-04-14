//! Filter design helpers — compute biquad coefficients from audio parameters.
//!
//! Design arithmetic uses `f64` for precision; the resulting [`BiquadCoeffs`]
//! are `f32` for runtime efficiency.
//!
//! The Chebyshev functions ([`chebyshev1_lowpass`] etc.) require the `alloc`
//! feature and return a `Vec` of second-order sections to be cascaded.

mod butterworth;
mod chebyshev;
mod eq;

pub use butterworth::{butterworth_highpass, butterworth_lowpass};
#[cfg(feature = "alloc")]
pub use chebyshev::{
    chebyshev1_highpass, chebyshev1_lowpass, chebyshev2_highpass, chebyshev2_lowpass,
};
pub use eq::{allpass, notch, peaking_eq, shelving_high, shelving_low};
