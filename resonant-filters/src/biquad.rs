//! Biquad filter — direct form II transposed.
//!
//! A second-order IIR section. Cascading multiple biquads builds higher-order
//! filters. The coefficients follow the standard convention:
//!
//! ```text
//! H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)
//! ```

/// Biquad filter coefficients.
///
/// Normalised so the denominator leading coefficient (a0) is 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiquadCoeffs {
    /// Numerator coefficients.
    pub b0: f32,
    /// Numerator z⁻¹ coefficient.
    pub b1: f32,
    /// Numerator z⁻² coefficient.
    pub b2: f32,
    /// Denominator z⁻¹ coefficient (a0 = 1 implied).
    pub a1: f32,
    /// Denominator z⁻² coefficient.
    pub a2: f32,
}

/// Internal delay state for a biquad section.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BiquadState {
    /// First delay element.
    pub s1: f32,
    /// Second delay element.
    pub s2: f32,
}

/// A stateful biquad filter (direct form II transposed).
///
/// Create with coefficients, then feed samples through
/// [`process_sample`](Biquad::process_sample) or
/// [`process_buf`](Biquad::process_buf). The filter retains its delay state
/// across calls, making it suitable for streaming.
///
/// # Examples
///
/// ```
/// use resonant_filters::biquad::{Biquad, BiquadCoeffs};
///
/// // Unity pass-through: b0=1, all others 0
/// let coeffs = BiquadCoeffs { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 };
/// let mut filter = Biquad::new(coeffs);
///
/// assert!((filter.process_sample(0.5) - 0.5).abs() < 1e-6);
/// assert!((filter.process_sample(-0.3) - (-0.3)).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Biquad {
    coeffs: BiquadCoeffs,
    state: BiquadState,
}

impl Biquad {
    /// Creates a biquad filter with the given coefficients and zeroed state.
    #[must_use]
    #[inline]
    pub fn new(coeffs: BiquadCoeffs) -> Self {
        Self {
            coeffs,
            state: BiquadState::default(),
        }
    }

    /// Processes a single input sample, returning the filtered output.
    ///
    /// Uses direct form II transposed for numerical stability.
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let c = &self.coeffs;
        let output = c.b0 * input + self.state.s1;
        self.state.s1 = c.b1 * input - c.a1 * output + self.state.s2;
        self.state.s2 = c.b2 * input - c.a2 * output;
        output
    }

    /// Filters an entire buffer in-place.
    pub fn process_buf(&mut self, buf: &mut [f32]) {
        for sample in buf.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Returns a reference to the current filter coefficients.
    #[must_use]
    #[inline]
    pub fn coeffs(&self) -> &BiquadCoeffs {
        &self.coeffs
    }

    /// Returns a reference to the current delay state.
    #[must_use]
    #[inline]
    pub fn state(&self) -> &BiquadState {
        &self.state
    }

    /// Replaces the filter coefficients, preserving the delay state.
    ///
    /// Useful for real-time parameter sweeps where resetting state would
    /// cause a click.
    #[inline]
    pub fn set_coeffs(&mut self, coeffs: BiquadCoeffs) {
        self.coeffs = coeffs;
    }

    /// Resets the delay state to zero.
    #[inline]
    pub fn reset(&mut self) {
        self.state = BiquadState::default();
    }
}

impl BiquadCoeffs {
    /// Unity pass-through (output = input).
    pub const PASSTHROUGH: Self = Self {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    };
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec::Vec;

    use super::*;

    #[test]
    fn passthrough_echoes_input() {
        let mut f = Biquad::new(BiquadCoeffs::PASSTHROUGH);
        for i in 0..10 {
            let x = i as f32 * 0.1;
            assert!((f.process_sample(x) - x).abs() < 1e-6);
        }
    }

    #[test]
    fn process_buf_matches_sample_by_sample() {
        let coeffs = BiquadCoeffs {
            b0: 0.5,
            b1: 0.3,
            b2: 0.1,
            a1: -0.2,
            a2: 0.05,
        };
        let input = [1.0_f32, 0.5, -0.3, 0.7, -0.1, 0.0, 0.4, -0.8];

        // sample-by-sample
        let mut f1 = Biquad::new(coeffs);
        let expected: Vec<f32> = input.iter().map(|&x| f1.process_sample(x)).collect();

        // buffer mode
        let mut f2 = Biquad::new(coeffs);
        let mut buf = input;
        f2.process_buf(&mut buf);

        for (a, b) in buf.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn reset_zeroes_state() {
        let mut f = Biquad::new(BiquadCoeffs {
            b0: 0.5,
            b1: 0.3,
            b2: 0.0,
            a1: -0.1,
            a2: 0.0,
        });
        f.process_sample(1.0);
        assert_ne!(f.state().s1, 0.0);
        f.reset();
        assert_eq!(f.state().s1, 0.0);
        assert_eq!(f.state().s2, 0.0);
    }

    #[test]
    fn set_coeffs_preserves_state() {
        let mut f = Biquad::new(BiquadCoeffs::PASSTHROUGH);
        f.process_sample(1.0);
        let state_before = *f.state();

        let new_coeffs = BiquadCoeffs {
            b0: 0.5,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        f.set_coeffs(new_coeffs);
        assert_eq!(*f.state(), state_before);
        assert_eq!(*f.coeffs(), new_coeffs);
    }

    #[test]
    fn impulse_response_first_tap_is_b0() {
        let coeffs = BiquadCoeffs {
            b0: 0.7,
            b1: 0.2,
            b2: 0.1,
            a1: -0.3,
            a2: 0.1,
        };
        let mut f = Biquad::new(coeffs);
        let y0 = f.process_sample(1.0);
        assert!((y0 - 0.7).abs() < 1e-6);
    }

    #[test]
    fn dc_gain_passthrough() {
        // DC gain = (b0+b1+b2) / (1+a1+a2)
        let coeffs = BiquadCoeffs {
            b0: 0.25,
            b1: 0.5,
            b2: 0.25,
            a1: -0.5,
            a2: 0.1,
        };
        let dc_gain = (coeffs.b0 + coeffs.b1 + coeffs.b2) / (1.0 + coeffs.a1 + coeffs.a2);

        let mut f = Biquad::new(coeffs);
        // Feed DC (constant 1.0) for enough samples to settle
        let mut output = 0.0;
        for _ in 0..1000 {
            output = f.process_sample(1.0);
        }
        assert!(
            (output - dc_gain).abs() < 1e-3,
            "expected DC gain {dc_gain}, got {output}"
        );
    }

    #[test]
    fn state_accessible() {
        let mut f = Biquad::new(BiquadCoeffs::PASSTHROUGH);
        f.process_sample(1.0);
        // State should be readable
        let _s = f.state();
    }

    #[test]
    fn zero_input_produces_decay() {
        let coeffs = BiquadCoeffs {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: -0.9,
            a2: 0.0,
        };
        let mut f = Biquad::new(coeffs);
        f.process_sample(1.0);

        // Feed zeros — output should decay toward 0
        let mut prev = f.process_sample(0.0).abs();
        for _ in 0..100 {
            let y = f.process_sample(0.0).abs();
            assert!(y <= prev + 1e-6);
            prev = y;
        }
        assert!(prev < 1e-3);
    }
}
