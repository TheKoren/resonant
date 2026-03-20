#![no_std]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

/// Fixed-capacity circular buffer for streaming DSP.
pub mod ring_buf;
/// Compile-time domain tracking for signals.
pub mod signal;

pub use ring_buf::RingBuf;
pub use signal::{Domain, FreqDomain, Signal, TimeDomain};
