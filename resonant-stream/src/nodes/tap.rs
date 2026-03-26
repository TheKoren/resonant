extern crate alloc;

use alloc::boxed::Box;

use crate::chunk::Chunk;
use crate::error::StreamError;
use crate::node::DspNode;

/// A side-channel observer that inspects chunks without modifying them.
///
/// `TapNode` calls a user-provided closure on every chunk, then passes
/// the chunk through unchanged. Useful for metering, logging, or
/// capturing intermediate data in a pipeline.
///
/// # Examples
///
/// ```
/// use resonant_stream::{Chunk, DspNode};
/// use resonant_stream::nodes::TapNode;
/// use std::sync::{Arc, Mutex};
///
/// let peak = Arc::new(Mutex::new(0.0_f32));
/// let peak_clone = Arc::clone(&peak);
///
/// let mut tap = TapNode::new(move |chunk: &Chunk| {
///     let max = chunk.data().iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
///     *peak_clone.lock().unwrap() = max;
/// });
///
/// let chunk = Chunk::new(vec![0.3, -0.7, 0.5], 44100, 1);
/// let out = tap.process(chunk).unwrap();
///
/// assert_eq!(*peak.lock().unwrap(), 0.7);
/// assert_eq!(out.data(), &[0.3, -0.7, 0.5]); // unchanged
/// ```
pub struct TapNode {
    callback: Box<dyn FnMut(&Chunk) + Send>,
}

impl TapNode {
    /// Creates a tap node with the given observer callback.
    pub fn new(callback: impl FnMut(&Chunk) + Send + 'static) -> Self {
        Self {
            callback: Box::new(callback),
        }
    }
}

impl DspNode for TapNode {
    fn process(&mut self, input: Chunk) -> Result<Chunk, StreamError> {
        (self.callback)(&input);
        Ok(input)
    }

    fn reset(&mut self) {
        // No state to reset — the callback is opaque.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn tap_observes_without_modifying() {
        let count = Arc::new(AtomicU32::new(0));
        let count_clone = Arc::clone(&count);
        let mut tap = TapNode::new(move |chunk: &Chunk| {
            count_clone.store(chunk.len() as u32, Ordering::Relaxed);
        });

        let chunk = Chunk::new(vec![1.0, 2.0, 3.0], 44100, 1);
        let out = tap.process(chunk).ok();
        assert_eq!(count.load(Ordering::Relaxed), 3);
        assert_eq!(
            out.as_ref().map(|c| c.data()),
            Some([1.0, 2.0, 3.0].as_slice())
        );
    }

    #[test]
    fn tap_called_every_process() {
        let count = Arc::new(AtomicU32::new(0));
        let count_clone = Arc::clone(&count);
        let mut tap = TapNode::new(move |_: &Chunk| {
            count_clone.fetch_add(1, Ordering::Relaxed);
        });

        for _ in 0..5 {
            let chunk = Chunk::new(vec![0.0], 44100, 1);
            let _ = tap.process(chunk);
        }
        assert_eq!(count.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn tap_preserves_metadata() {
        let mut tap = TapNode::new(|_: &Chunk| {});
        let chunk = Chunk::new(vec![1.0, 2.0], 96000, 2);
        let out = tap.process(chunk).ok();
        let out = out.as_ref();
        assert_eq!(out.map(|c| c.sample_rate()), Some(96000));
        assert_eq!(out.map(|c| c.channels()), Some(2));
    }

    #[test]
    fn tap_empty_chunk() {
        let mut tap = TapNode::new(|_: &Chunk| {});
        let chunk = Chunk::empty(44100, 1);
        let out = tap.process(chunk).ok();
        assert_eq!(out.as_ref().map(|c| c.is_empty()), Some(true));
    }

    #[test]
    fn tap_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<TapNode>();
    }

    #[test]
    fn tap_in_pipeline() {
        use crate::Pipeline;

        let seen = Arc::new(AtomicU32::new(0));
        let seen_clone = Arc::clone(&seen);

        let mut pipeline = Pipeline::builder()
            .node(TapNode::new(move |chunk: &Chunk| {
                seen_clone.store(chunk.data().len() as u32, Ordering::Relaxed);
            }))
            .build();

        let chunk = Chunk::new(vec![1.0, 2.0], 44100, 1);
        let out = pipeline.process(chunk);
        assert!(out.is_ok());
        assert_eq!(seen.load(Ordering::Relaxed), 2);
    }
}
