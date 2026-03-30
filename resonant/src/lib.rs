#![warn(missing_docs)]

//! `resonant` — ergonomic audio DSP facade for Rust.
//!
//! Load an audio file and analyse it in one line:
//!
//! ```rust,ignore
//! use resonant::AudioFile;
//!
//! let audio = AudioFile::open("track.wav")?;
//! let bins = audio.with_window_size(4096).fft()?;
//! ```
//!
//! For streaming frame-by-frame analysis:
//!
//! ```rust,ignore
//! for frame in audio.with_window_size(1024).fft_stream()? {
//!     let bins = frame?;
//!     // process each frame...
//! }
//! ```
//!
//! # Re-exports
//!
//! This crate re-exports the most commonly needed types from the lower-level
//! resonant crates so that casual users can `use resonant::*` without pulling
//! in sub-crate details.

pub mod audio_file;
pub mod error;
pub mod frequency_bin;

// === Facade types ===
pub use audio_file::{AnalysisResult, AudioFile, FftFrameIter, WindowFn};
pub use error::AudioError;
pub use frequency_bin::FrequencyBin;

// === resonant-core re-exports ===

/// Type-state signal container with compile-time domain tracking.
pub use resonant_core::signal::{FreqDomain, Signal, TimeDomain};

/// Window functions (Hann, Hamming, Blackman, Rectangular, Bartlett).
pub use resonant_core::window;

// === resonant-fft re-exports ===

/// Extension traits for `.fft()`, `.ifft()`, and spectral helpers on `Signal`.
pub use resonant_fft::{SignalFftExt, SignalFreqExt, SignalIfftExt};

/// Complex number type used by FFT outputs.
pub use resonant_fft::Complex;

// === resonant-filters re-exports ===

/// Biquad (second-order IIR) filter and coefficients.
pub use resonant_filters::{Biquad, BiquadCoeffs};

/// Butterworth filter design helpers.
pub use resonant_filters::design;

// === resonant-analysis re-exports ===

/// Audio analysis error type.
pub use resonant_analysis::AnalysisError;

/// High-level analysis modules.
pub use resonant_analysis::{chroma, mfcc, onset, pitch, spectral, tempo};
