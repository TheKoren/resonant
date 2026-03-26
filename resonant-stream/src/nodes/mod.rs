//! Built-in processing nodes for common DSP operations.
//!
//! Each node implements [`DspNode`](crate::DspNode) and can be used standalone
//! or composed into a [`Pipeline`](crate::Pipeline).

mod filter;
mod gain;
mod tap;

pub use filter::FilterNode;
pub use gain::GainNode;
pub use tap::TapNode;
