#![warn(missing_docs)]

//! `resonant-stream` — streaming DSP pipeline.
//!
//! This crate provides a pull-based, synchronous processing pipeline for audio
//! DSP. Chunks of sample data flow through a chain of [`DspNode`]s, each
//! transforming the audio in place where possible to minimise allocations.
//!
//! # Architecture
//!
//! ```text
//! ┌───────────┐    ┌───────────┐    ┌───────────┐
//! │  Source    │───▶│  Node A   │───▶│  Node B   │───▶ output
//! │ (Chunk)   │    │ (filter)  │    │ (gain)    │
//! └───────────┘    └───────────┘    └───────────┘
//! ```
//!
//! - [`Chunk`] carries interleaved samples plus metadata (sample rate, channels).
//! - [`DspNode`] is the core trait — implement it to create custom processors.
//! - [`StreamError`] covers format mismatches and processing failures.
//!
//! # Quick start
//!
//! ```
//! use resonant_stream::{Chunk, DspNode, StreamError};
//!
//! // A simple gain node that scales all samples.
//! struct Gain(f32);
//!
//! impl DspNode for Gain {
//!     fn process(&mut self, mut input: Chunk) -> Result<Chunk, StreamError> {
//!         for s in input.data_mut() {
//!             *s *= self.0;
//!         }
//!         Ok(input)
//!     }
//!     fn reset(&mut self) {}
//! }
//!
//! let mut gain = Gain(0.5);
//! let chunk = Chunk::new(vec![1.0, -1.0, 0.5, -0.5], 44100, 1);
//! let out = gain.process(chunk).unwrap();
//! assert_eq!(out.data(), &[0.5, -0.5, 0.25, -0.25]);
//! ```

mod chunk;
mod error;
/// Operator-overloaded graph combinators for pipeline composition.
pub mod graph;
#[cfg(feature = "io")]
/// Hardware audio I/O nodes backed by [`cpal`](https://docs.rs/cpal).
pub mod io;
mod node;
/// Built-in processing nodes.
pub mod nodes;
mod pipeline;

pub use chunk::Chunk;
pub use error::StreamError;
pub use graph::{GraphExt, Parallel, Serial, Stack};
pub use node::DspNode;
pub use pipeline::{Pipeline, PipelineBuilder};
