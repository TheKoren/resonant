#![cfg_attr(not(feature = "alloc"), no_std)]
#![warn(missing_docs)]
// Several modules import `num_traits::float::Float as _` to bring transcendental
// methods (.sin, .cos, .sqrt, .tanh) into scope on f32/f64 in no_std builds.
// The #[allow(unused_imports)] suppresses the lint because Rust considers it unused
// when the only use is implicit method resolution through the trait bound.

//! `resonant-fft` — type-safe FFT, STFT, and DCT transforms.
//!
//! ## Backends
//!
//! - **`rustfft`** (default feature) — wraps the [`rustfft`] crate for
//!   arbitrary-size FFTs. Requires `std` (implies `alloc`).
//! - **`radix2`** — pure-core radix-2 Cooley-Tukey, `no_std`, `no_alloc`,
//!   power-of-two sizes only. Always available.
//!
//! ## Feature flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `rustfft` | ✓ | Arbitrary-size FFT via `rustfft`; implies `alloc` |
//! | `alloc`   | via rustfft | Enables `SignalFftExt`, `SignalFreqExt`, STFT |
//!
//! ## Extension traits
//!
//! Import [`SignalFftExt`] to call `.fft()` on time-domain signals and
//! [`SignalIfftExt`] to call `.ifft()` on frequency-domain signals. These
//! enforce domain transitions at compile time. Requires the `alloc` feature.
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
//! ## no_std / no_alloc usage
//!
//! Disable default features to get only the radix-2 FFT and DCT, which
//! operate on caller-provided slices with no allocation:
//!
//! ```toml
//! resonant-fft = { version = "...", default-features = false }
//! ```

/// Discrete Cosine Transform — types II and III.
pub mod dct;

/// Pure-core radix-2 FFT implementation (power-of-two sizes, no allocation).
pub mod radix2;

/// Real-valued FFT — half the output bins, roughly half the cost.
pub mod rfft;

#[cfg(feature = "rustfft")]
/// FFT backend using `rustfft` — supports arbitrary sizes.
pub mod rustfft_backend;

#[cfg(feature = "alloc")]
mod ext;

#[cfg(feature = "alloc")]
pub(crate) mod simd;

/// Short-Time Fourier Transform with overlap-add reconstruction.
#[cfg(feature = "alloc")]
pub mod stft;

#[cfg(feature = "alloc")]
pub use ext::{SignalFftExt, SignalFreqExt, SignalIfftExt, SignalRfftExt};
pub use num_complex::Complex;
pub use radix2::{fft, ifft};
pub use rfft::{irfft, rfft};

/// Errors that can occur during FFT computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftError {
    /// Input length is not a power of two (radix-2), or does not match the
    /// planned length (rustfft plan).
    NotPowerOfTwo(usize),
    /// Input is empty.
    Empty,
    /// Input and output buffer lengths do not match.
    LengthMismatch {
        /// Length of the input buffer.
        input: usize,
        /// Length of the output buffer.
        output: usize,
    },
}

impl core::fmt::Display for FftError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotPowerOfTwo(n) => write!(f, "FFT length {n} is not a power of two"),
            Self::Empty => write!(f, "FFT input is empty"),
            Self::LengthMismatch { input, output } => {
                write!(f, "buffer length mismatch: input {input}, output {output}")
            }
        }
    }
}
