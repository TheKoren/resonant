extern crate alloc;

use alloc::vec::Vec;

/// A buffer of audio samples with associated metadata.
///
/// `Chunk` is the unit of data that flows between [`DspNode`](crate::DspNode)s
/// in a pipeline. It owns its sample buffer so that nodes can process in-place
/// without extra allocations.
///
/// Samples are stored interleaved when `channels > 1`:
/// `[L0, R0, L1, R1, ...]` for stereo.
///
/// # Examples
///
/// ```
/// use resonant_stream::Chunk;
///
/// let chunk = Chunk::new(vec![0.0_f32; 1024], 44100, 1);
/// assert_eq!(chunk.frames(), 1024);
/// assert_eq!(chunk.sample_rate(), 44100);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    data: Vec<f32>,
    sample_rate: u32,
    channels: u16,
}

impl Chunk {
    /// Creates a new chunk from interleaved sample data.
    ///
    /// # Panics
    ///
    /// Panics if `channels` is zero or `sample_rate` is zero.
    #[must_use]
    pub fn new(data: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        assert!(channels > 0, "channels must be > 0");
        assert!(sample_rate > 0, "sample_rate must be > 0");
        Self {
            data,
            sample_rate,
            channels,
        }
    }

    /// Creates an empty chunk with the given format.
    ///
    /// # Panics
    ///
    /// Panics if `channels` is zero or `sample_rate` is zero.
    #[must_use]
    pub fn empty(sample_rate: u32, channels: u16) -> Self {
        Self::new(Vec::new(), sample_rate, channels)
    }

    /// Returns a reference to the interleaved sample data.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Returns a mutable reference to the interleaved sample data.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Consumes the chunk, returning the underlying sample buffer.
    #[inline]
    #[must_use]
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Replaces the sample data, reusing this chunk's metadata.
    #[inline]
    pub fn set_data(&mut self, data: Vec<f32>) {
        self.data = data;
    }

    /// The sample rate in Hz.
    #[inline]
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The number of interleaved channels.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// The number of sample frames (total samples / channels).
    #[inline]
    #[must_use]
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.data.len() / self.channels as usize
    }

    /// Returns `true` if the chunk contains no samples.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The total number of samples (frames * channels).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_mono_chunk() {
        let chunk = Chunk::new(vec![1.0, 2.0, 3.0, 4.0], 44100, 1);
        assert_eq!(chunk.frames(), 4);
        assert_eq!(chunk.len(), 4);
        assert_eq!(chunk.channels(), 1);
        assert_eq!(chunk.sample_rate(), 44100);
        assert!(!chunk.is_empty());
    }

    #[test]
    fn new_stereo_chunk() {
        let chunk = Chunk::new(vec![1.0, 2.0, 3.0, 4.0], 48000, 2);
        assert_eq!(chunk.frames(), 2);
        assert_eq!(chunk.len(), 4);
        assert_eq!(chunk.channels(), 2);
    }

    #[test]
    fn empty_chunk() {
        let chunk = Chunk::empty(44100, 1);
        assert!(chunk.is_empty());
        assert_eq!(chunk.frames(), 0);
        assert_eq!(chunk.len(), 0);
    }

    #[test]
    fn data_access() {
        let chunk = Chunk::new(vec![0.5, -0.5], 44100, 1);
        assert_eq!(chunk.data(), &[0.5, -0.5]);
    }

    #[test]
    fn data_mut_allows_modification() {
        let mut chunk = Chunk::new(vec![0.0; 4], 44100, 1);
        chunk.data_mut()[0] = 1.0;
        assert_eq!(chunk.data()[0], 1.0);
    }

    #[test]
    fn into_data_returns_buffer() {
        let chunk = Chunk::new(vec![1.0, 2.0], 44100, 1);
        let data = chunk.into_data();
        assert_eq!(data, vec![1.0, 2.0]);
    }

    #[test]
    fn set_data_replaces_buffer() {
        let mut chunk = Chunk::new(vec![1.0], 44100, 1);
        chunk.set_data(vec![2.0, 3.0]);
        assert_eq!(chunk.data(), &[2.0, 3.0]);
        assert_eq!(chunk.frames(), 2);
    }

    #[test]
    fn clone_and_eq() {
        let a = Chunk::new(vec![1.0, 2.0], 44100, 2);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    #[should_panic(expected = "channels must be > 0")]
    fn zero_channels_panics() {
        let _ = Chunk::new(vec![1.0], 44100, 0);
    }

    #[test]
    #[should_panic(expected = "sample_rate must be > 0")]
    fn zero_sample_rate_panics() {
        let _ = Chunk::new(vec![1.0], 0, 1);
    }
}
