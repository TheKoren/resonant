#![cfg_attr(not(feature = "rustfft"), no_std)]
#![warn(missing_docs)]

//! `resonant-fft` — type-safe FFT, STFT, and DCT transforms.
//!
//! ## Backends
//!
//! - **`rustfft`** (default feature) — wraps the [`rustfft`] crate for
//!   arbitrary-size FFTs. Requires `std`.
//! - **`radix2`** — pure-core radix-2 Cooley-Tukey, `no_std`, `no_alloc`,
//!   power-of-two sizes only. Always available.
//!
//! ## Extension traits
//!
//! Import [`SignalFftExt`] to call `.fft()` on time-domain signals and
//! [`SignalIfftExt`] to call `.ifft()` on frequency-domain signals. These
//! enforce domain transitions at compile time.
//!
//! ```
//! use resonant_core::signal::Signal;
//! use resonant_fft::{SignalFftExt, SignalIfftExt};
//!
//! let sig = Signal::from_samples(vec![1.0_f32, 0.0, -1.0, 0.0]);
//! let freq = sig.fft().unwrap();
//! let time = freq.ifft().unwrap();
//! ```
//!
//! To use only the `no_std` fallback, disable default features:
//!
//! ```toml
//! resonant-fft = { version = "...", default-features = false }
//! ```

/// Pure-core radix-2 FFT implementation (power-of-two sizes, no allocation).
pub mod radix2;

#[cfg(feature = "rustfft")]
/// FFT backend using `rustfft` — supports arbitrary sizes.
pub mod rustfft_backend;

mod ext;

pub use ext::{SignalFftExt, SignalFreqExt, SignalIfftExt};
pub use num_complex::Complex;
pub use radix2::{fft, ifft};

/// Errors that can occur during FFT computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftError {
    /// Input length is not a power of two (radix-2), or does not match the
    /// planned length (rustfft plan).
    NotPowerOfTwo(usize),
    /// Input is empty.
    Empty,
}

impl core::fmt::Display for FftError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotPowerOfTwo(n) => write!(f, "FFT length {n} is not a power of two"),
            Self::Empty => write!(f, "FFT input is empty"),
        }
    }
}
