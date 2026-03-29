#![warn(missing_docs)]

//! `resonant-analysis` — high-level audio analysis features.
//!
//! Provides onset detection, beat tracking, pitch estimation, MFCCs,
//! chroma features, and spectral descriptors. All algorithms build on
//! the primitives in [`resonant-fft`] and [`resonant-core`].
//!
//! # Quick start
//!
//! ```
//! use resonant_analysis::spectral;
//!
//! let magnitudes = [0.1, 0.5, 1.0, 0.3];
//! let frequencies = [0.0, 500.0, 1000.0, 1500.0];
//!
//! let centroid = spectral::spectral_centroid(&magnitudes, &frequencies).unwrap();
//! let flatness = spectral::spectral_flatness(&magnitudes).unwrap();
//! ```

pub mod error;
/// Onset detection via spectral flux and adaptive thresholding.
pub mod onset;
/// Pitch estimation using the YIN algorithm.
pub mod pitch;
/// Spectral feature extraction: centroid, spread, flatness, rolloff.
pub mod spectral;

pub use error::AnalysisError;
