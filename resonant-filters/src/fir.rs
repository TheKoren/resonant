//! FIR (Finite Impulse Response) filter with arbitrary coefficients.
//!
//! Convolves input samples with a fixed coefficient set. The filter is
//! stateful — it holds a delay line matching the coefficient length so
//! it can be used in a streaming context.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

/// A stateful FIR filter.
///
/// Stores the coefficient vector and an internal delay line. Feed samples
/// through [`process_sample`](Fir::process_sample) or
/// [`process_buf`](Fir::process_buf).
///
/// # Examples
///
/// ```
/// use resonant_filters::fir::Fir;
///
/// // Simple 3-tap averaging filter
/// let mut f = Fir::new(vec![1.0 / 3.0; 3]);
/// let y = f.process_sample(1.0);
/// assert!((y - 1.0 / 3.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct Fir {
    coeffs: Vec<f32>,
    delay: Vec<f32>,
    pos: usize,
    // Linearized delay line reused across process_buf calls to avoid
    // per-call heap allocation on the hot audio path.
    scratch: Vec<f32>,
}

impl PartialEq for Fir {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs && self.delay == other.delay && self.pos == other.pos
    }
}

impl Fir {
    /// Creates a FIR filter with the given coefficients.
    ///
    /// The delay line is initialised to zeros. The number of taps equals
    /// `coeffs.len()`.
    ///
    /// # Panics
    ///
    /// Panics if `coeffs` is empty.
    #[must_use]
    pub fn new(coeffs: Vec<f32>) -> Self {
        assert!(!coeffs.is_empty(), "FIR filter requires at least one tap");
        let len = coeffs.len();
        Self {
            coeffs,
            delay: vec![0.0; len],
            pos: 0,
            scratch: vec![0.0; len],
        }
    }

    /// Processes a single input sample, returning the filtered output.
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        self.delay[self.pos] = input;
        let mut output = 0.0;
        let len = self.coeffs.len();

        // Convolve: newest sample at pos, oldest at pos+1 (wrapped)
        let mut idx = self.pos;
        for coeff in &self.coeffs {
            output += coeff * self.delay[idx];
            if idx == 0 {
                idx = len - 1;
            } else {
                idx -= 1;
            }
        }

        self.pos = (self.pos + 1) % len;
        output
    }

    /// Filters an entire buffer in-place.
    ///
    /// For each sample, the circular delay line is linearized into a
    /// contiguous scratch buffer so the convolution can use a SIMD dot
    /// product. The scratch buffer is stored in the struct and reused across
    /// calls, so this method does not allocate.
    pub fn process_buf(&mut self, buf: &mut [f32]) {
        let len = self.coeffs.len();

        for sample in buf.iter_mut() {
            self.delay[self.pos] = *sample;

            // Linearize: newest sample first, oldest last.
            // Layout: [pos, pos-1, ..., 0, len-1, len-2, ..., pos+1]
            let after = self.pos + 1;
            // First part: delay[0..=pos] reversed
            self.scratch[..after].copy_from_slice(&self.delay[..after]);
            self.scratch[..after].reverse();
            // Second part: delay[pos+1..len] reversed
            if after < len {
                self.scratch[after..].copy_from_slice(&self.delay[after..]);
                self.scratch[after..].reverse();
            }

            *sample = crate::simd::dot_product(&self.coeffs, &self.scratch);
            self.pos = (self.pos + 1) % len;
        }
    }

    /// Returns the filter coefficients.
    #[must_use]
    pub fn coeffs(&self) -> &[f32] {
        &self.coeffs
    }

    /// Returns the number of taps (coefficient length).
    #[must_use]
    #[inline]
    pub fn num_taps(&self) -> usize {
        self.coeffs.len()
    }

    /// Returns the current delay line contents.
    #[must_use]
    pub fn delay(&self) -> &[f32] {
        &self.delay
    }

    /// Resets the delay line to zeros.
    pub fn reset(&mut self) {
        self.delay.fill(0.0);
        self.pos = 0;
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;

    #[test]
    fn single_tap_is_scalar_multiply() {
        let mut f = Fir::new(std::vec![0.5]);
        assert!((f.process_sample(1.0) - 0.5).abs() < 1e-6);
        assert!((f.process_sample(2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn two_tap_delay() {
        // h = [1, 0.5] → y[n] = x[n] + 0.5*x[n-1]
        let mut f = Fir::new(std::vec![1.0, 0.5]);
        assert!((f.process_sample(1.0) - 1.0).abs() < 1e-6); // 1*1 + 0.5*0
        assert!((f.process_sample(0.0) - 0.5).abs() < 1e-6); // 1*0 + 0.5*1
        assert!(f.process_sample(0.0).abs() < 1e-6); // 1*0 + 0.5*0
    }

    #[test]
    fn impulse_response_matches_coeffs() {
        let coeffs = std::vec![0.3, 0.5, 0.2];
        let mut f = Fir::new(coeffs.clone());

        // Feed impulse
        let y0 = f.process_sample(1.0);
        let y1 = f.process_sample(0.0);
        let y2 = f.process_sample(0.0);
        let y3 = f.process_sample(0.0);

        assert!((y0 - 0.3).abs() < 1e-6);
        assert!((y1 - 0.5).abs() < 1e-6);
        assert!((y2 - 0.2).abs() < 1e-6);
        assert!(y3.abs() < 1e-6); // FIR: finite response
    }

    #[test]
    fn process_buf_matches_sample_by_sample() {
        let coeffs = std::vec![0.4, 0.3, 0.2, 0.1];
        let input = [1.0_f32, 0.5, -0.3, 0.7, -0.1];

        let mut f1 = Fir::new(coeffs.clone());
        let expected: Vec<f32> = input.iter().map(|&x| f1.process_sample(x)).collect();

        let mut f2 = Fir::new(coeffs);
        let mut buf = input;
        f2.process_buf(&mut buf);

        for (a, b) in buf.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut f = Fir::new(std::vec![1.0, 0.5]);
        f.process_sample(1.0);
        f.reset();
        // After reset, delay line should be all zeros
        assert!(f.delay().iter().all(|&d| d == 0.0));
        assert!((f.process_sample(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn accessors() {
        let f = Fir::new(std::vec![0.25; 8]);
        assert_eq!(f.num_taps(), 8);
        assert_eq!(f.coeffs().len(), 8);
        assert_eq!(f.delay().len(), 8);
    }

    #[test]
    fn averaging_filter() {
        // 4-tap box filter: each coeff = 0.25
        let mut f = Fir::new(std::vec![0.25; 4]);
        // Feed constant 1.0 — after 4 samples, output should be 1.0
        for _ in 0..3 {
            f.process_sample(1.0);
        }
        let y = f.process_sample(1.0);
        assert!((y - 1.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "at least one tap")]
    fn empty_coeffs_panics() {
        let _ = Fir::new(std::vec![]);
    }
}
