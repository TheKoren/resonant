#![no_std]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

/// Compile-time domain tracking for signals.
pub mod signal;

pub use signal::{Domain, FreqDomain, Signal, TimeDomain};
