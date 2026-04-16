//! Operator-overloaded pipeline graph composition.
//!
//! Provides concrete combinator types that implement [`DspNode`], along with
//! operator impls so nodes can be wired together with familiar symbols:
//!
//! | Operator | Type | Semantics |
//! |---|---|---|
//! | `a >> b` | [`Serial<A, B>`] | Feed `a`'s output into `b` |
//! | `a & b`  | [`Parallel<A, B>`] | Sum `a` and `b` over the same input |
//! | `a \| b`  | [`Stack<A, B>`] | Concatenate `a` and `b` outputs |
//!
//! # Examples
//!
//! ```
//! use resonant_stream::{Chunk, DspNode, StreamError};
//! use resonant_stream::graph::{GraphExt, NodeGraph};
//!
//! struct Scale(f32);
//! impl DspNode for Scale {
//!     fn process(&mut self, mut input: Chunk) -> Result<Chunk, StreamError> {
//!         for s in input.data_mut() { *s *= self.0; }
//!         Ok(input)
//!     }
//!     fn reset(&mut self) {}
//! }
//!
//! // Serial: halve, then halve again → ×0.25
//! let mut graph = Scale(0.5).serial(Scale(0.5));
//! let chunk = Chunk::new(vec![4.0, 8.0], 44100, 1);
//! let out = graph.process(chunk).unwrap();
//! assert_eq!(out.data(), &[1.0, 2.0]);
//! ```

use crate::{Chunk, DspNode, StreamError};

/// Serial composition: `A`'s output becomes `B`'s input.
///
/// Equivalent to `a >> b` via the [`GraphExt`] extension trait.
pub struct Serial<A, B> {
    first: A,
    second: B,
}

impl<A: DspNode, B: DspNode> DspNode for Serial<A, B> {
    fn process(&mut self, input: Chunk) -> Result<Chunk, StreamError> {
        let mid = self.first.process(input)?;
        self.second.process(mid)
    }

    fn reset(&mut self) {
        self.first.reset();
        self.second.reset();
    }
}

/// Parallel composition: both nodes receive the same input; outputs are summed.
///
/// Both nodes must produce the same number of samples. Returns
/// [`StreamError::ChannelMismatch`] if output lengths differ.
///
/// Equivalent to `a & b` via [`GraphExt`].
pub struct Parallel<A, B> {
    left: A,
    right: B,
}

impl<A: DspNode, B: DspNode> DspNode for Parallel<A, B> {
    fn process(&mut self, input: Chunk) -> Result<Chunk, StreamError> {
        let out_l = self.left.process(input.clone())?;
        let out_r = self.right.process(input)?;

        if out_l.len() != out_r.len() {
            return Err(StreamError::ChannelMismatch {
                expected: out_l.len() as u16,
                got: out_r.len() as u16,
            });
        }

        let sr = out_l.sample_rate();
        let ch = out_l.channels();
        let summed: alloc::vec::Vec<f32> = out_l
            .into_data()
            .iter()
            .zip(out_r.into_data().iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Chunk::new(summed, sr, ch))
    }

    fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
    }
}

/// Stack composition: both nodes receive the same input; outputs are concatenated.
///
/// Useful for building multi-band or multi-output processors.
/// The output chunk will have a sample count equal to the sum of both outputs.
///
/// Equivalent to `a | b` via [`GraphExt`].
pub struct Stack<A, B> {
    top: A,
    bottom: B,
}

impl<A: DspNode, B: DspNode> DspNode for Stack<A, B> {
    fn process(&mut self, input: Chunk) -> Result<Chunk, StreamError> {
        let out_t = self.top.process(input.clone())?;
        let out_b = self.bottom.process(input)?;

        let sr = out_t.sample_rate();
        let ch = out_t.channels();
        let mut combined = out_t.into_data();
        combined.extend(out_b.into_data());

        Ok(Chunk::new(combined, sr, ch))
    }

    fn reset(&mut self) {
        self.top.reset();
        self.bottom.reset();
    }
}

/// Extension trait that adds graph-composition methods to every [`DspNode`].
///
/// Prefer these methods over the operator impls when chaining more than two
/// nodes, since they avoid having to spell out type parameters:
///
/// ```
/// use resonant_stream::{Chunk, DspNode, StreamError};
/// use resonant_stream::graph::GraphExt;
///
/// struct Noop;
/// impl DspNode for Noop {
///     fn process(&mut self, input: Chunk) -> Result<Chunk, StreamError> { Ok(input) }
///     fn reset(&mut self) {}
/// }
///
/// let _ = Noop.serial(Noop).serial(Noop); // (Noop >> Noop) >> Noop
/// ```
pub trait GraphExt: DspNode + Sized {
    /// Chain `self` into `other` serially: `self`'s output → `other`'s input.
    fn serial<B: DspNode>(self, other: B) -> Serial<Self, B> {
        Serial {
            first: self,
            second: other,
        }
    }

    /// Run `self` and `other` on the same input; sum their outputs.
    fn parallel<B: DspNode>(self, other: B) -> Parallel<Self, B> {
        Parallel {
            left: self,
            right: other,
        }
    }

    /// Run `self` and `other` on the same input; concatenate their outputs.
    fn stack<B: DspNode>(self, other: B) -> Stack<Self, B> {
        Stack {
            top: self,
            bottom: other,
        }
    }
}

impl<T: DspNode + Sized> GraphExt for T {}

/// A `DspNode` that was constructed via graph combinators.
///
/// This is a convenience alias used in `Pipeline::from_graph`.
/// Any type implementing `DspNode + Send + 'static` qualifies.
pub trait NodeGraph: DspNode + Send + 'static {}
impl<T: DspNode + Send + 'static> NodeGraph for T {}

extern crate alloc;

#[cfg(test)]
mod tests {
    use super::*;

    fn scale(factor: f32) -> impl DspNode {
        struct Scale(f32);
        impl DspNode for Scale {
            fn process(&mut self, mut input: Chunk) -> Result<Chunk, StreamError> {
                for s in input.data_mut() {
                    *s *= self.0;
                }
                Ok(input)
            }
            fn reset(&mut self) {}
        }
        Scale(factor)
    }

    fn make_chunk(data: alloc::vec::Vec<f32>) -> Chunk {
        Chunk::new(data, 44100, 1)
    }

    #[test]
    fn serial_chains_nodes() {
        let mut g = scale(2.0).serial(scale(3.0));
        let out = g.process(make_chunk(alloc::vec![1.0, 2.0])).unwrap();
        assert_eq!(out.data(), &[6.0, 12.0]);
    }

    #[test]
    fn serial_matches_manual_pipeline() {
        let mut g = scale(2.0).serial(scale(0.5));
        let out = g.process(make_chunk(alloc::vec![4.0])).unwrap();
        assert_eq!(out.data(), &[4.0]); // 4 * 2 * 0.5
    }

    #[test]
    fn serial_associativity() {
        // (a >> b) >> c should equal a >> (b >> c)
        let chunk_a = make_chunk(alloc::vec![1.0]);
        let chunk_b = make_chunk(alloc::vec![1.0]);
        let mut left = scale(2.0).serial(scale(3.0)).serial(scale(4.0));
        let mut right = scale(2.0).serial(scale(3.0).serial(scale(4.0)));
        assert_eq!(
            left.process(chunk_a).unwrap().into_data(),
            right.process(chunk_b).unwrap().into_data()
        );
    }

    #[test]
    fn serial_error_propagates() {
        struct Fail;
        impl DspNode for Fail {
            fn process(&mut self, _: Chunk) -> Result<Chunk, StreamError> {
                Err(StreamError::ProcessingError("fail".into()))
            }
            fn reset(&mut self) {}
        }
        let mut g = scale(2.0).serial(Fail);
        assert!(g.process(make_chunk(alloc::vec![1.0])).is_err());
    }

    #[test]
    fn serial_reset_propagates() {
        let mut g = scale(1.0).serial(scale(1.0));
        g.reset(); // should not panic
    }

    #[test]
    fn parallel_sums_outputs() {
        // scale(2) & scale(3) on [1.0] → [2.0] + [3.0] = [5.0]
        let mut g = scale(2.0).parallel(scale(3.0));
        let out = g.process(make_chunk(alloc::vec![1.0, 1.0])).unwrap();
        assert_eq!(out.data(), &[5.0, 5.0]);
    }

    #[test]
    fn parallel_identity_doubles() {
        // Running the same passthrough in parallel sums each sample with itself
        struct Pass;
        impl DspNode for Pass {
            fn process(&mut self, input: Chunk) -> Result<Chunk, StreamError> {
                Ok(input)
            }
            fn reset(&mut self) {}
        }
        let mut g = Pass.parallel(Pass);
        let out = g.process(make_chunk(alloc::vec![1.0, -0.5])).unwrap();
        assert_eq!(out.data(), &[2.0, -1.0]);
    }

    #[test]
    fn parallel_reset_propagates() {
        let mut g = scale(1.0).parallel(scale(1.0));
        g.reset();
    }

    #[test]
    fn stack_concatenates_outputs() {
        let mut g = scale(2.0).stack(scale(3.0));
        let out = g.process(make_chunk(alloc::vec![1.0])).unwrap();
        // [1.0 * 2] ++ [1.0 * 3] = [2.0, 3.0]
        assert_eq!(out.data(), &[2.0, 3.0]);
    }

    #[test]
    fn stack_output_length_is_sum() {
        let mut g = scale(1.0).stack(scale(1.0));
        let out = g.process(make_chunk(alloc::vec![1.0; 4])).unwrap();
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn stack_reset_propagates() {
        let mut g = scale(1.0).stack(scale(1.0));
        g.reset();
    }

    #[test]
    fn graph_ext_serial_method() {
        let mut g = scale(4.0).serial(scale(0.25));
        let out = g.process(make_chunk(alloc::vec![2.0])).unwrap();
        assert_eq!(out.data(), &[2.0]);
    }

    #[test]
    fn graph_ext_parallel_method() {
        let mut g = scale(1.0).parallel(scale(1.0));
        let out = g.process(make_chunk(alloc::vec![0.5])).unwrap();
        assert_eq!(out.data(), &[1.0]);
    }

    #[test]
    fn graph_ext_stack_method() {
        let mut g = scale(1.0).stack(scale(2.0));
        let out = g.process(make_chunk(alloc::vec![1.0])).unwrap();
        assert_eq!(out.data(), &[1.0, 2.0]);
    }
}
