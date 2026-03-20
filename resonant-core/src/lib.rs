#![no_std]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

/// Fixed-point arithmetic types (`Q15`, `Q31`).
pub mod fixed;
/// Fixed-capacity circular buffer for streaming DSP.
pub mod ring_buf;
/// Compile-time domain tracking for signals.
pub mod signal;
/// Heap-allocated sliding window for overlapping frame extraction.
#[cfg(feature = "alloc")]
pub mod sliding_window;
/// Window functions for spectral analysis.
pub mod window;

pub use fixed::Q15;
pub use ring_buf::RingBuf;
pub use signal::{Domain, FreqDomain, Signal, TimeDomain};
#[cfg(feature = "alloc")]
pub use sliding_window::SlidingWindow;
