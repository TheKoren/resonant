#![warn(missing_docs)]

//! `resonant` — ergonomic audio DSP facade for Rust.
//!
//! Load an audio file and analyse it in one line:
//!
//! ```rust,ignore
//! use resonant::AudioFile;
//!
//! let audio = AudioFile::open("track.wav")?;
//! println!("Sample rate: {} Hz, duration: {:.1}s", audio.sample_rate(), audio.duration_secs());
//! ```

pub mod audio_file;
pub mod error;
pub mod frequency_bin;

pub use audio_file::{AudioFile, FftFrameIter};
pub use error::AudioError;
pub use frequency_bin::FrequencyBin;
