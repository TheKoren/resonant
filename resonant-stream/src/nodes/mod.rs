//! Built-in processing nodes for common DSP operations.
//!
//! Each node implements [`DspNode`](crate::DspNode) and can be used standalone
//! or composed into a [`Pipeline`](crate::Pipeline).

mod fft;
mod filter;
mod gain;
mod mix;
mod resample;
mod tap;

pub use fft::{FftNode, FftOutput};
pub use filter::FilterNode;
pub use gain::GainNode;
pub use mix::MixNode;
pub use resample::ResampleNode;
pub use tap::TapNode;
