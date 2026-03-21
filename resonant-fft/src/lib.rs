#![no_std]
#![warn(missing_docs)]

//! `resonant-fft` — type-safe FFT, STFT, and DCT transforms.
//!
//! This crate provides a pure-core radix-2 Cooley-Tukey FFT that works on
//! `no_std` targets without allocation. Power-of-two sizes only.

/// Pure-core radix-2 FFT implementation (power-of-two sizes, no allocation).
pub mod radix2;

pub use num_complex::Complex;
pub use radix2::{fft, ifft};

/// Errors that can occur during FFT computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftError {
    /// Input length is not a power of two.
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
