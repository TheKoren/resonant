//! Filter design helpers — compute biquad coefficients from audio parameters.
//!
//! Design arithmetic uses `f64` for precision; the resulting [`BiquadCoeffs`]
//! are `f32` for runtime efficiency.
//!
//! All design functions return `Result<_, DesignError>`, making it possible to
//! distinguish the failure reason (invalid frequency, bad sample rate, etc.).
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

/// Errors returned by filter design functions.
///
/// Every design function in this module returns `Result<_, DesignError>` so
/// callers can distinguish the specific failure reason rather than receiving an
/// opaque `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DesignError {
    /// Cutoff or centre frequency is zero, negative, or ≥ Nyquist.
    FrequencyOutOfRange,
    /// Sample rate is zero or negative.
    SampleRateOutOfRange,
    /// Quality factor Q is zero or negative.
    QOutOfRange,
    /// Filter order is odd or outside the supported range (2–20 for Chebyshev).
    OrderOutOfRange,
    /// Passband ripple or stopband attenuation is zero or negative.
    RippleOutOfRange,
}

impl core::fmt::Display for DesignError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::FrequencyOutOfRange => write!(f, "frequency out of range"),
            Self::SampleRateOutOfRange => write!(f, "sample rate out of range"),
            Self::QOutOfRange => write!(f, "Q factor out of range"),
            Self::OrderOutOfRange => write!(f, "filter order out of range"),
            Self::RippleOutOfRange => write!(f, "ripple or stopband attenuation out of range"),
        }
    }
}
