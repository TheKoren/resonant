#![no_std]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

/// Fixed-point arithmetic types (`Q15`, `Q31`).
pub mod fixed;
/// Fixed-capacity circular buffer for streaming DSP.
pub mod ring_buf;
/// Heap-backed circular buffer for runtime-determined capacities.
#[cfg(feature = "alloc")]
pub mod ring_buf_heap;
/// Compile-time domain tracking for signals.
pub mod signal;
/// Arithmetic operators for [`Signal`] — element-wise, domain-preserving.
pub mod signal_ops;
/// Heap-allocated sliding window for overlapping frame extraction.
#[cfg(feature = "alloc")]
pub mod sliding_window;
/// Window functions for spectral analysis.
pub mod window;

pub(crate) mod simd;

pub use fixed::{Q15, Q31};
pub use ring_buf::RingBuf;
#[cfg(feature = "alloc")]
pub use ring_buf_heap::HeapRingBuf;
pub use signal::{Domain, FreqDomain, Signal, TimeDomain};
#[cfg(feature = "alloc")]
pub use sliding_window::SlidingWindow;
