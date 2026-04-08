//! Sample-rate conversion — integer decimation with anti-alias filtering.
//!
//! Decimation reduces the sample rate by an integer factor *M*: for every
//! *M* input samples, one output sample is produced. Before downsampling,
//! a lowpass anti-alias filter removes frequencies above the new Nyquist
//! limit to prevent aliasing.
//!
//! The implementation cascades two second-order Butterworth sections for a
//! fourth-order rolloff (~24 dB/octave), which provides adequate alias
//! rejection for most audio applications.
//!
//! # Examples
//!
//! ```
//! use resonant_filters::resample;
//!
//! // Decimate a 48 kHz signal down to 16 kHz (factor 3)
//! let input: Vec<f32> = (0..480).map(|i| (i as f32 * 0.1).sin()).collect();
//! let output = resample::decimate(&input, 3, 48000.0).unwrap();
//! assert_eq!(output.len(), 160);
//! ```

mod decimate;
mod polyphase;

pub use decimate::decimate;
pub use polyphase::PolyphaseResampler;
