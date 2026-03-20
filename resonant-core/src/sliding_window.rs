extern crate alloc;

use alloc::vec::Vec;

/// Heap-allocated sliding window that yields overlapping frames from a sample stream.
///
/// Collects samples via [`push`](SlidingWindow::push) or
/// [`push_slice`](SlidingWindow::push_slice). Once enough samples have been
/// collected, [`current_frame`](SlidingWindow::current_frame) returns the
/// active window. Call [`advance`](SlidingWindow::advance) to shift forward by
/// `hop_size` samples and prepare the next frame.
///
/// Requires the `alloc` feature.
///
/// # Examples
///
/// ```
/// use resonant_core::SlidingWindow;
///
/// let mut sw = SlidingWindow::new(4, 2);
/// sw.push_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
///
/// assert!(sw.is_ready());
/// assert_eq!(sw.current_frame(), Some([1.0, 2.0, 3.0, 4.0].as_slice()));
///
/// sw.advance();
/// assert!(!sw.is_ready());
///
/// sw.push_slice(&[5.0, 6.0]);
/// assert_eq!(sw.current_frame(), Some([3.0, 4.0, 5.0, 6.0].as_slice()));
/// ```
#[derive(Debug, Clone)]
pub struct SlidingWindow<T> {
    buf: Vec<T>,
    window_size: usize,
    hop_size: usize,
}

impl<T: Clone> SlidingWindow<T> {
    /// Creates a new sliding window.
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is zero or `hop_size` is zero.
    #[must_use]
    pub fn new(window_size: usize, hop_size: usize) -> Self {
        assert!(window_size > 0, "window_size must be > 0");
        assert!(hop_size > 0, "hop_size must be > 0");
        Self {
            buf: Vec::with_capacity(window_size),
            window_size,
            hop_size,
        }
    }

    /// Pushes a single sample into the window buffer.
    pub fn push(&mut self, sample: T) {
        self.buf.push(sample);
    }

    /// Pushes a slice of samples into the window buffer.
    pub fn push_slice(&mut self, samples: &[T]) {
        self.buf.extend_from_slice(samples);
    }

    /// Returns `true` when at least `window_size` samples are buffered.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.buf.len() >= self.window_size
    }

    /// Returns the current frame if enough samples are available.
    #[must_use]
    pub fn current_frame(&self) -> Option<&[T]> {
        if self.is_ready() {
            Some(&self.buf[..self.window_size])
        } else {
            None
        }
    }

    /// Advances the window by `hop_size`, discarding consumed samples.
    pub fn advance(&mut self) {
        if self.buf.len() >= self.hop_size {
            self.buf.drain(..self.hop_size);
        } else {
            self.buf.clear();
        }
    }

    /// Returns the configured window size.
    #[must_use]
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Returns the configured hop size.
    #[must_use]
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Returns the number of samples currently buffered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Returns `true` if no samples are buffered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Discards all buffered samples without changing configuration.
    pub fn clear(&mut self) {
        self.buf.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_window_is_empty_and_not_ready() {
        let sw: SlidingWindow<f32> = SlidingWindow::new(4, 2);
        assert!(sw.is_empty());
        assert!(!sw.is_ready());
        assert_eq!(sw.len(), 0);
        assert_eq!(sw.current_frame(), None);
    }

    #[test]
    fn push_single_samples_until_ready() {
        let mut sw = SlidingWindow::new(3, 1);
        sw.push(1.0_f32);
        sw.push(2.0);
        assert!(!sw.is_ready());
        sw.push(3.0);
        assert!(sw.is_ready());
        assert_eq!(sw.current_frame(), Some([1.0, 2.0, 3.0].as_slice()));
    }

    #[test]
    fn push_slice_fills_buffer() {
        let mut sw = SlidingWindow::new(4, 2);
        sw.push_slice(&[10.0_f32, 20.0, 30.0, 40.0]);
        assert!(sw.is_ready());
        assert_eq!(
            sw.current_frame(),
            Some([10.0, 20.0, 30.0, 40.0].as_slice())
        );
    }

    #[test]
    fn advance_shifts_by_hop_size() {
        let mut sw = SlidingWindow::new(4, 2);
        sw.push_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(sw.current_frame(), Some([1.0, 2.0, 3.0, 4.0].as_slice()));

        sw.advance();
        assert_eq!(sw.current_frame(), Some([3.0, 4.0, 5.0, 6.0].as_slice()));

        sw.advance();
        assert!(!sw.is_ready());
        assert_eq!(sw.len(), 2);
    }

    #[test]
    fn advance_when_fewer_than_hop_clears() {
        let mut sw = SlidingWindow::new(4, 8);
        sw.push_slice(&[1.0_f32, 2.0, 3.0]);
        sw.advance();
        assert!(sw.is_empty());
    }

    #[test]
    fn hop_equals_window_no_overlap() {
        let mut sw = SlidingWindow::new(3, 3);
        sw.push_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(sw.current_frame(), Some([1.0, 2.0, 3.0].as_slice()));
        sw.advance();
        assert_eq!(sw.current_frame(), Some([4.0, 5.0, 6.0].as_slice()));
    }

    #[test]
    fn clear_resets_buffer() {
        let mut sw = SlidingWindow::new(4, 2);
        sw.push_slice(&[1.0_f32, 2.0, 3.0]);
        sw.clear();
        assert!(sw.is_empty());
        assert!(!sw.is_ready());
    }

    #[test]
    fn accessors_return_config() {
        let sw: SlidingWindow<f32> = SlidingWindow::new(512, 128);
        assert_eq!(sw.window_size(), 512);
        assert_eq!(sw.hop_size(), 128);
    }

    #[test]
    #[should_panic(expected = "window_size must be > 0")]
    fn zero_window_size_panics() {
        let _sw: SlidingWindow<f32> = SlidingWindow::new(0, 1);
    }

    #[test]
    #[should_panic(expected = "hop_size must be > 0")]
    fn zero_hop_size_panics() {
        let _sw: SlidingWindow<f32> = SlidingWindow::new(4, 0);
    }

    #[test]
    fn works_with_integers() {
        let mut sw = SlidingWindow::new(2, 1);
        sw.push(10_i32);
        sw.push(20);
        assert_eq!(sw.current_frame(), Some([10, 20].as_slice()));
        sw.advance();
        sw.push(30);
        assert_eq!(sw.current_frame(), Some([20, 30].as_slice()));
    }
}
