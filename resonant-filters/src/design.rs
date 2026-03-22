//! Filter design helpers — compute biquad coefficients from audio parameters.
//!
//! Design arithmetic uses `f64` for precision; the resulting [`BiquadCoeffs`]
//! are `f32` for runtime efficiency.

use core::f64::consts::PI;

#[allow(unused_imports)]
use num_traits::float::Float as _;

use crate::BiquadCoeffs;

/// Designs a second-order Butterworth lowpass filter.
///
/// Returns biquad coefficients normalised for direct form II transposed.
///
/// # Arguments
///
/// * `cutoff_hz` — cutoff frequency in Hz (must be < `sample_rate / 2`)
/// * `sample_rate` — sample rate in Hz
///
/// # Errors
///
/// Returns `None` if `cutoff_hz` is zero, negative, or ≥ Nyquist.
///
/// # Examples
///
/// ```
/// use resonant_filters::design;
/// use resonant_filters::Biquad;
///
/// let coeffs = design::butterworth_lowpass(1000.0, 44100.0).unwrap();
/// let mut filter = Biquad::new(coeffs);
/// let y = filter.process_sample(1.0);
/// assert!(y.is_finite());
/// ```
#[must_use]
pub fn butterworth_lowpass(cutoff_hz: f64, sample_rate: f64) -> Option<BiquadCoeffs> {
    let nyquist = sample_rate / 2.0;
    if cutoff_hz <= 0.0 || cutoff_hz >= nyquist || sample_rate <= 0.0 {
        return None;
    }

    // Bilinear transform pre-warp
    let wc = 2.0 * PI * cutoff_hz / sample_rate;
    let k = (wc / 2.0).tan();
    let k2 = k * k;
    let sqrt2 = core::f64::consts::SQRT_2;
    let norm = 1.0 / (1.0 + sqrt2 * k + k2);

    let b0 = k2 * norm;
    let b1 = 2.0 * b0;
    let b2 = b0;
    let a1 = 2.0 * (k2 - 1.0) * norm;
    let a2 = (1.0 - sqrt2 * k + k2) * norm;

    Some(BiquadCoeffs {
        b0: b0 as f32,
        b1: b1 as f32,
        b2: b2 as f32,
        a1: a1 as f32,
        a2: a2 as f32,
    })
}

/// Designs a second-order Butterworth highpass filter.
///
/// Returns biquad coefficients normalised for direct form II transposed.
///
/// # Arguments
///
/// * `cutoff_hz` — cutoff frequency in Hz (must be < `sample_rate / 2`)
/// * `sample_rate` — sample rate in Hz
///
/// # Errors
///
/// Returns `None` if `cutoff_hz` is zero, negative, or ≥ Nyquist.
///
/// # Examples
///
/// ```
/// use resonant_filters::design;
/// use resonant_filters::Biquad;
///
/// let coeffs = design::butterworth_highpass(1000.0, 44100.0).unwrap();
/// let mut filter = Biquad::new(coeffs);
/// let y = filter.process_sample(1.0);
/// assert!(y.is_finite());
/// ```
#[must_use]
pub fn butterworth_highpass(cutoff_hz: f64, sample_rate: f64) -> Option<BiquadCoeffs> {
    let nyquist = sample_rate / 2.0;
    if cutoff_hz <= 0.0 || cutoff_hz >= nyquist || sample_rate <= 0.0 {
        return None;
    }

    let wc = 2.0 * PI * cutoff_hz / sample_rate;
    let k = (wc / 2.0).tan();
    let k2 = k * k;
    let sqrt2 = core::f64::consts::SQRT_2;
    let norm = 1.0 / (1.0 + sqrt2 * k + k2);

    let b0 = norm;
    let b1 = -2.0 * norm;
    let b2 = norm;
    let a1 = 2.0 * (k2 - 1.0) * norm;
    let a2 = (1.0 - sqrt2 * k + k2) * norm;

    Some(BiquadCoeffs {
        b0: b0 as f32,
        b1: b1 as f32,
        b2: b2 as f32,
        a1: a1 as f32,
        a2: a2 as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Biquad;

    const SR: f64 = 44100.0;

    #[test]
    fn lowpass_returns_some_for_valid_params() {
        assert!(butterworth_lowpass(1000.0, SR).is_some());
    }

    #[test]
    fn lowpass_none_for_zero_cutoff() {
        assert!(butterworth_lowpass(0.0, SR).is_none());
    }

    #[test]
    fn lowpass_none_for_negative_cutoff() {
        assert!(butterworth_lowpass(-100.0, SR).is_none());
    }

    #[test]
    fn lowpass_none_at_nyquist() {
        assert!(butterworth_lowpass(SR / 2.0, SR).is_none());
    }

    #[test]
    fn lowpass_none_above_nyquist() {
        assert!(butterworth_lowpass(SR, SR).is_none());
    }

    #[test]
    fn lowpass_none_for_zero_sample_rate() {
        assert!(butterworth_lowpass(100.0, 0.0).is_none());
    }

    #[test]
    fn lowpass_dc_gain_is_unity() {
        // DC gain = (b0+b1+b2) / (1+a1+a2) should be ~1.0 for a lowpass
        let c = butterworth_lowpass(1000.0, SR).unwrap();
        let dc = (c.b0 + c.b1 + c.b2) / (1.0 + c.a1 + c.a2);
        assert!((dc - 1.0).abs() < 1e-4, "DC gain = {dc}");
    }

    #[test]
    fn lowpass_attenuates_high_frequency() {
        let c = butterworth_lowpass(1000.0, SR).unwrap();
        let mut f = Biquad::new(c);

        // Generate a high-frequency sinusoid (10kHz)
        let freq = 10000.0_f32;
        let mut max_out = 0.0_f32;
        for i in 0..4000 {
            let x = (2.0 * core::f32::consts::PI * freq * i as f32 / SR as f32).sin();
            let y = f.process_sample(x);
            if i > 1000 {
                // skip transient
                max_out = max_out.max(y.abs());
            }
        }
        // Should be significantly attenuated
        assert!(max_out < 0.2, "high-freq output peak = {max_out}");
    }

    #[test]
    fn lowpass_passes_low_frequency() {
        let c = butterworth_lowpass(5000.0, SR).unwrap();
        let mut f = Biquad::new(c);

        // Generate a low-frequency sinusoid (100Hz)
        let freq = 100.0_f32;
        let mut max_out = 0.0_f32;
        for i in 0..4000 {
            let x = (2.0 * core::f32::consts::PI * freq * i as f32 / SR as f32).sin();
            let y = f.process_sample(x);
            if i > 1000 {
                max_out = max_out.max(y.abs());
            }
        }
        // Should pass through near unity
        assert!(max_out > 0.9, "low-freq output peak = {max_out}");
    }

    #[test]
    fn highpass_returns_some_for_valid_params() {
        assert!(butterworth_highpass(1000.0, SR).is_some());
    }

    #[test]
    fn highpass_none_for_invalid_params() {
        assert!(butterworth_highpass(0.0, SR).is_none());
        assert!(butterworth_highpass(-100.0, SR).is_none());
        assert!(butterworth_highpass(SR / 2.0, SR).is_none());
    }

    #[test]
    fn highpass_dc_gain_is_zero() {
        let c = butterworth_highpass(1000.0, SR).unwrap();
        let dc = (c.b0 + c.b1 + c.b2) / (1.0 + c.a1 + c.a2);
        assert!(dc.abs() < 1e-4, "DC gain = {dc}");
    }

    #[test]
    fn highpass_passes_high_frequency() {
        let c = butterworth_highpass(1000.0, SR).unwrap();
        let mut f = Biquad::new(c);

        let freq = 10000.0_f32;
        let mut max_out = 0.0_f32;
        for i in 0..4000 {
            let x = (2.0 * core::f32::consts::PI * freq * i as f32 / SR as f32).sin();
            let y = f.process_sample(x);
            if i > 1000 {
                max_out = max_out.max(y.abs());
            }
        }
        assert!(max_out > 0.9, "high-freq output peak = {max_out}");
    }

    #[test]
    fn highpass_attenuates_low_frequency() {
        let c = butterworth_highpass(5000.0, SR).unwrap();
        let mut f = Biquad::new(c);

        let freq = 100.0_f32;
        let mut max_out = 0.0_f32;
        for i in 0..4000 {
            let x = (2.0 * core::f32::consts::PI * freq * i as f32 / SR as f32).sin();
            let y = f.process_sample(x);
            if i > 1000 {
                max_out = max_out.max(y.abs());
            }
        }
        assert!(max_out < 0.1, "low-freq output peak = {max_out}");
    }

    #[test]
    fn coefficients_are_finite() {
        let lp = butterworth_lowpass(5000.0, SR).unwrap();
        assert!(lp.b0.is_finite());
        assert!(lp.b1.is_finite());
        assert!(lp.b2.is_finite());
        assert!(lp.a1.is_finite());
        assert!(lp.a2.is_finite());

        let hp = butterworth_highpass(5000.0, SR).unwrap();
        assert!(hp.b0.is_finite());
        assert!(hp.b1.is_finite());
        assert!(hp.b2.is_finite());
        assert!(hp.a1.is_finite());
        assert!(hp.a2.is_finite());
    }
}
