//! Short-Time Fourier Transform with overlap-add reconstruction.
//!
//! Splits a signal into overlapping windowed frames, applies the FFT to each,
//! and can reconstruct the original via inverse FFT + overlap-add.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use num_complex::Complex;
use resonant_core::signal::{FreqDomain, Signal, TimeDomain};

use crate::FftError;

/// A single STFT frame in the frequency domain.
pub type StftFrame = Signal<Vec<Complex<f32>>, FreqDomain>;

/// Configures and runs the Short-Time Fourier Transform.
///
/// # Examples
///
/// ```
/// use resonant_core::signal::Signal;
/// use resonant_fft::stft::Stft;
///
/// let signal = Signal::from_samples(vec![0.0_f32; 256]);
/// let stft = Stft::builder(64, 32).build();
/// let frames = stft.analyze(&signal).unwrap();
/// let reconstructed = stft.synthesize(&frames).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Stft {
    window_size: usize,
    hop_size: usize,
    window_fn: Option<fn(&mut [f32])>,
}

/// Builder for [`Stft`] configuration.
#[derive(Debug, Clone)]
pub struct StftBuilder {
    window_size: usize,
    hop_size: usize,
    window_fn: Option<fn(&mut [f32])>,
}

impl Stft {
    /// Creates a builder with the given window and hop sizes.
    ///
    /// # Panics
    ///
    /// Panics if `window_size` or `hop_size` is zero, or if
    /// `hop_size > window_size`.
    #[must_use]
    pub fn builder(window_size: usize, hop_size: usize) -> StftBuilder {
        StftBuilder::new(window_size, hop_size)
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

    /// Splits a time-domain signal into overlapping windowed frames and
    /// computes the FFT of each.
    ///
    /// Returns a `Vec` of frequency-domain signals, one per frame.
    ///
    /// # Errors
    ///
    /// Returns [`FftError`] if the window size is invalid for the FFT backend.
    pub fn analyze<T: AsRef<[f32]>>(
        &self,
        signal: &Signal<T, TimeDomain>,
    ) -> Result<Vec<StftFrame>, FftError> {
        let samples = signal.data().as_ref();
        if samples.is_empty() {
            return Err(FftError::Empty);
        }

        let mut frames = Vec::new();
        let mut offset = 0;

        while offset + self.window_size <= samples.len() {
            let frame = self.analyze_frame(&samples[offset..offset + self.window_size])?;
            frames.push(frame);
            offset += self.hop_size;
        }

        Ok(frames)
    }

    /// Reconstructs a time-domain signal from STFT frames via overlap-add.
    ///
    /// # Errors
    ///
    /// Returns [`FftError`] if the IFFT fails on any frame.
    pub fn synthesize(
        &self,
        frames: &[StftFrame],
    ) -> Result<Signal<Vec<f32>, TimeDomain>, FftError> {
        if frames.is_empty() {
            return Ok(Signal::new(Vec::new()));
        }

        let output_len = (frames.len() - 1) * self.hop_size + self.window_size;
        let mut output = vec![0.0_f32; output_len];

        for (i, frame) in frames.iter().enumerate() {
            let time_frame = self.synthesize_frame(frame)?;
            let offset = i * self.hop_size;
            for (j, &sample) in time_frame.iter().enumerate() {
                output[offset + j] += sample;
            }
        }

        Ok(Signal::new(output))
    }

    /// Analyzes a single frame: apply window, then FFT.
    fn analyze_frame(&self, frame: &[f32]) -> Result<StftFrame, FftError> {
        let mut windowed = vec![0.0_f32; self.window_size];
        windowed.copy_from_slice(frame);

        if let Some(wfn) = self.window_fn {
            wfn(&mut windowed);
        }

        let mut buf: Vec<Complex<f32>> = windowed.iter().map(|&s| Complex::new(s, 0.0)).collect();

        crate::ext::run_fft_forward(&mut buf)?;
        Ok(Signal::new(buf))
    }

    /// Synthesizes a single frame: IFFT, then apply synthesis window.
    fn synthesize_frame(&self, frame: &StftFrame) -> Result<Vec<f32>, FftError> {
        let mut buf = frame.data().clone();
        crate::ext::run_fft_inverse(&mut buf)?;
        let mut real: Vec<f32> = buf.iter().map(|c| c.re).collect();

        if let Some(wfn) = self.window_fn {
            wfn(&mut real);
        }

        Ok(real)
    }
}

impl StftBuilder {
    /// Creates a new builder.
    ///
    /// # Panics
    ///
    /// Panics if `window_size` or `hop_size` is zero, or if
    /// `hop_size > window_size`.
    #[must_use]
    pub fn new(window_size: usize, hop_size: usize) -> Self {
        assert!(window_size > 0, "window_size must be > 0");
        assert!(hop_size > 0, "hop_size must be > 0");
        assert!(hop_size <= window_size, "hop_size must be <= window_size");
        Self {
            window_size,
            hop_size,
            window_fn: None,
        }
    }

    /// Sets the window function applied to each frame during analysis
    /// and synthesis.
    ///
    /// Use functions from `resonant_core::window` (e.g. `window::hann`).
    /// If not set, no windowing is applied (rectangular window).
    #[must_use]
    pub fn window_fn(mut self, f: fn(&mut [f32])) -> Self {
        self.window_fn = Some(f);
        self
    }

    /// Builds the [`Stft`] configuration.
    #[must_use]
    pub fn build(self) -> Stft {
        Stft {
            window_size: self.window_size,
            hop_size: self.hop_size,
            window_fn: self.window_fn,
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;

    #[test]
    fn analyze_produces_correct_frame_count() {
        // 256 samples, window=64, hop=32 → (256-64)/32 + 1 = 7 frames
        let sig = Signal::from_samples(std::vec![0.0_f32; 256]);
        let stft = Stft::builder(64, 32).build();
        let frames = stft.analyze(&sig).unwrap();
        assert_eq!(frames.len(), 7);
    }

    #[test]
    fn analyze_frame_size_matches_window() {
        let sig = Signal::from_samples(std::vec![1.0_f32; 128]);
        let stft = Stft::builder(32, 16).build();
        let frames = stft.analyze(&sig).unwrap();
        for f in &frames {
            assert_eq!(f.data().len(), 32);
        }
    }

    #[test]
    fn roundtrip_no_window() {
        // Without windowing, overlap-add with hop=window gives exact reconstruction
        let original: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
        let sig = Signal::from_samples(original.clone());
        let stft = Stft::builder(16, 16).build();
        let frames = stft.analyze(&sig).unwrap();
        let reconstructed = stft.synthesize(&frames).unwrap();
        for (a, b) in reconstructed.data().iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-3, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn analyze_empty_returns_error() {
        let sig = Signal::from_samples(std::vec![0.0_f32; 0]);
        let stft = Stft::builder(16, 8).build();
        assert_eq!(stft.analyze(&sig), Err(FftError::Empty));
    }

    #[test]
    fn analyze_shorter_than_window_returns_empty() {
        let sig = Signal::from_samples(std::vec![1.0_f32; 8]);
        let stft = Stft::builder(16, 8).build();
        // Not an error — just no complete frames
        // Actually, this returns empty because 8 < 16
        // Wait, we return Err for empty signal but Ok(empty vec) for too-short
        let frames = stft.analyze(&sig).unwrap();
        assert!(frames.is_empty());
    }

    #[test]
    fn synthesize_empty_frames() {
        let stft = Stft::builder(16, 8).build();
        let result = stft.synthesize(&[]).unwrap();
        assert!(result.data().is_empty());
    }

    #[test]
    fn accessors() {
        let stft = Stft::builder(512, 128).build();
        assert_eq!(stft.window_size(), 512);
        assert_eq!(stft.hop_size(), 128);
    }

    #[test]
    fn with_window_function() {
        let sig = Signal::from_samples(std::vec![1.0_f32; 64]);
        let stft = Stft::builder(16, 8)
            .window_fn(resonant_core::window::hann)
            .build();
        let frames = stft.analyze(&sig).unwrap();
        assert!(!frames.is_empty());
    }

    #[test]
    #[should_panic(expected = "window_size must be > 0")]
    fn zero_window_panics() {
        let _ = Stft::builder(0, 1);
    }

    #[test]
    #[should_panic(expected = "hop_size must be > 0")]
    fn zero_hop_panics() {
        let _ = Stft::builder(16, 0);
    }

    #[test]
    #[should_panic(expected = "hop_size must be <= window_size")]
    fn hop_larger_than_window_panics() {
        let _ = Stft::builder(16, 32);
    }

    #[test]
    fn single_frame_roundtrip() {
        let original: Vec<f32> = (0..16).map(|i| (i as f32) / 16.0).collect();
        let sig = Signal::from_samples(original.clone());
        let stft = Stft::builder(16, 16).build();
        let frames = stft.analyze(&sig).unwrap();
        assert_eq!(frames.len(), 1);
        let reconstructed = stft.synthesize(&frames).unwrap();
        for (a, b) in reconstructed.data().iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
    }
}
