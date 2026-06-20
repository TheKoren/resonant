#[allow(unused_imports)]
use num_traits::float::Float as _;

use crate::BiquadCoeffs;

/// Biquad filter with a tanh waveshaper applied to the output and state updates.
///
/// `drive = 0.0` gives linear (identical to [`Biquad`](crate::Biquad)).
/// `drive = 1.0` gives full tanh saturation: output is bounded by `(-1, 1)`.
///
/// # Examples
///
/// ```
/// use resonant_filters::{design, nonlinear::SaturatingBiquad};
///
/// let coeffs = design::butterworth_lowpass(1000.0, 44100.0).unwrap();
/// let mut f = SaturatingBiquad::new(coeffs, 0.5);
/// let y = f.process_sample(1.0);
/// assert!(y.is_finite());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaturatingBiquad {
    pub(super) coeffs: BiquadCoeffs,
    pub(super) s1: f32,
    pub(super) s2: f32,
    pub(super) drive: f32,
}

impl SaturatingBiquad {
    /// Creates a new saturating biquad with zeroed state.
    #[must_use]
    #[inline]
    pub fn new(coeffs: BiquadCoeffs, drive: f32) -> Self {
        Self {
            coeffs,
            s1: 0.0,
            s2: 0.0,
            drive: drive.clamp(0.0, 1.0),
        }
    }

    /// Processes a single sample.
    ///
    /// Applies the waveshaper `(1 − drive)·x + drive·tanh(x)` to the
    /// combined output signal and both state-update paths.
    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let c = &self.coeffs;
        let pre = c.b0 * x + self.s1;
        let output = self.shape(pre);
        self.s1 = self.shape(c.b1 * x - c.a1 * output + self.s2);
        self.s2 = self.shape(c.b2 * x - c.a2 * output);
        output
    }

    /// Filters an entire buffer in-place.
    pub fn process_buf(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            *s = self.process_sample(*s);
        }
    }

    /// Returns the current drive amount in `[0.0, 1.0]`.
    #[inline]
    #[must_use]
    pub fn drive(&self) -> f32 {
        self.drive
    }

    /// Sets the drive amount, clamping to `[0.0, 1.0]`.
    #[inline]
    pub fn set_drive(&mut self, drive: f32) {
        self.drive = drive.clamp(0.0, 1.0);
    }

    /// Replaces the coefficients without resetting state.
    #[inline]
    pub fn set_coeffs(&mut self, coeffs: BiquadCoeffs) {
        self.coeffs = coeffs;
    }

    /// Resets the delay state to zero.
    #[inline]
    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }

    #[inline]
    fn shape(&self, x: f32) -> f32 {
        x * (1.0 - self.drive) + x.tanh() * self.drive
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{design, Biquad, BiquadCoeffs};

    #[test]
    fn drive_zero_is_linear() {
        let coeffs = BiquadCoeffs {
            b0: 0.5,
            b1: 0.3,
            b2: 0.1,
            a1: -0.2,
            a2: 0.05,
        };
        let mut sat = SaturatingBiquad::new(coeffs, 0.0);
        let mut lin = Biquad::new(coeffs);
        for i in 0..20 {
            let x = i as f32 * 0.1 - 1.0;
            let ys = sat.process_sample(x);
            let yl = lin.process_sample(x);
            assert!((ys - yl).abs() < 1e-5, "mismatch at {x}: {ys} vs {yl}");
        }
    }

    #[test]
    fn drive_one_bounds_output() {
        let mut f = SaturatingBiquad::new(BiquadCoeffs::PASSTHROUGH, 1.0);
        for _ in 0..200 {
            let y = f.process_sample(1000.0);
            assert!(y.abs() <= 1.0 + 1e-4, "output out of bounds: {y}");
        }
    }

    #[test]
    fn drive_one_negative_large_input() {
        let mut f = SaturatingBiquad::new(BiquadCoeffs::PASSTHROUGH, 1.0);
        for _ in 0..200 {
            let y = f.process_sample(-500.0);
            assert!(y.abs() <= 1.0 + 1e-4, "output out of bounds: {y}");
        }
    }

    #[test]
    fn small_signal_matches_linear() {
        let coeffs = design::butterworth_lowpass(1000.0, 44100.0).unwrap();
        let mut sat = SaturatingBiquad::new(coeffs, 0.8);
        let mut lin = Biquad::new(coeffs);
        for i in 0..50 {
            let x = (i as f32 * 0.01).sin() * 0.001;
            let ys = sat.process_sample(x);
            let yl = lin.process_sample(x);
            assert!(
                (ys - yl).abs() < 1e-4,
                "small-signal mismatch: {ys} vs {yl}"
            );
        }
    }

    #[test]
    fn reset_zeroes_state() {
        let mut f = SaturatingBiquad::new(BiquadCoeffs::PASSTHROUGH, 0.5);
        f.process_sample(1.0);
        f.reset();
        assert_eq!(f.s1, 0.0);
        assert_eq!(f.s2, 0.0);
    }

    #[test]
    fn drive_getter_matches_constructed_value() {
        let f = SaturatingBiquad::new(BiquadCoeffs::PASSTHROUGH, 0.75);
        assert_eq!(f.drive(), 0.75);
    }

    #[test]
    fn set_drive_clamps_above_one() {
        let mut f = SaturatingBiquad::new(BiquadCoeffs::PASSTHROUGH, 0.5);
        f.set_drive(2.0);
        assert_eq!(f.drive(), 1.0);
    }

    #[test]
    fn set_drive_clamps_below_zero() {
        let mut f = SaturatingBiquad::new(BiquadCoeffs::PASSTHROUGH, 0.5);
        f.set_drive(-1.0);
        assert_eq!(f.drive(), 0.0);
    }

    #[test]
    fn set_drive_accepts_valid_range() {
        let mut f = SaturatingBiquad::new(BiquadCoeffs::PASSTHROUGH, 0.0);
        f.set_drive(0.3);
        assert_eq!(f.drive(), 0.3);
    }

    #[test]
    fn new_clamps_drive_above_one() {
        let f = SaturatingBiquad::new(BiquadCoeffs::PASSTHROUGH, 5.0);
        assert_eq!(f.drive(), 1.0);
    }

    #[test]
    fn finite_output() {
        let coeffs = design::butterworth_lowpass(500.0, 44100.0).unwrap();
        let mut f = SaturatingBiquad::new(coeffs, 1.0);
        for i in 0..1000 {
            let y = f.process_sample((i as f32 * 0.1).sin());
            assert!(y.is_finite(), "NaN/inf at sample {i}");
        }
    }
}
