#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use core::f64::consts::PI;

#[allow(unused_imports)]
use num_traits::float::Float as _;

use crate::BiquadCoeffs;

#[cfg(feature = "alloc")]
use crate::response::FilterError;

#[cfg(feature = "alloc")]
fn validate_cheby(
    order: usize,
    param_db: f64,
    cutoff_hz: f64,
    sample_rate: f64,
) -> Result<(), FilterError> {
    if !(2..=20).contains(&order) || order % 2 != 0 {
        return Err(FilterError::InvalidParameter);
    }
    if param_db <= 0.0 || sample_rate <= 0.0 {
        return Err(FilterError::InvalidParameter);
    }
    if cutoff_hz <= 0.0 || cutoff_hz >= sample_rate / 2.0 {
        return Err(FilterError::InvalidParameter);
    }
    Ok(())
}

/// Designs an Nth-order Chebyshev Type I lowpass filter.
///
/// Returns a cascade of `order / 2` second-order sections. Apply them in
/// sequence: `for section in sections { y = Biquad::new(section).process_sample(x); }`.
///
/// # Parameters
///
/// * `order` — filter order; must be even, 2–20
/// * `ripple_db` — passband ripple in dB (e.g. `0.5` or `3.0`)
/// * `cutoff_hz` — passband edge frequency in Hz (gain = −`ripple_db` dB here)
/// * `sample_rate` — sample rate in Hz
///
/// # Errors
///
/// Returns [`FilterError::InvalidParameter`] if `order` is odd or outside
/// \[2, 20\], if `ripple_db` ≤ 0, or if `cutoff_hz` is not in `(0, Nyquist)`.
///
/// # Examples
///
/// ```
/// use resonant_filters::design;
/// use resonant_filters::Biquad;
///
/// let sections = design::chebyshev1_lowpass(4, 3.0, 1000.0, 44100.0).unwrap();
/// let mut filters: Vec<_> = sections.iter().map(|&c| Biquad::new(c)).collect();
/// let mut y = 1.0_f32;
/// for f in &mut filters { y = f.process_sample(y); }
/// assert!(y.is_finite());
/// ```
#[cfg(feature = "alloc")]
pub fn chebyshev1_lowpass(
    order: usize,
    ripple_db: f64,
    cutoff_hz: f64,
    sample_rate: f64,
) -> Result<Vec<BiquadCoeffs>, FilterError> {
    validate_cheby(order, ripple_db, cutoff_hz, sample_rate)?;

    let eps_sq = 10.0_f64.powf(ripple_db / 10.0) - 1.0;
    let gamma = (1.0 / eps_sq.sqrt()).asinh() / order as f64;
    let omega_p = (PI * cutoff_hz / sample_rate).tan();
    // Distribute the gain correction for even-N Chebyshev I across all sections
    // so that the maximum passband gain equals 1.0 and DC = 1/sqrt(1+ε²).
    let gc = (1.0 + eps_sq).powf(-1.0 / order as f64);

    let half = order / 2;
    let mut out = Vec::with_capacity(half);
    for k in 1..=half {
        let theta = PI * (2 * k - 1) as f64 / (2 * order) as f64;
        let sigma = gamma.sinh() * theta.sin(); // > 0; analog pole Re = -omega_p*sigma
        let omega = gamma.cosh() * theta.cos();
        let c = omega_p * omega_p * (sigma * sigma + omega * omega);
        let d = 2.0 * omega_p * sigma; // -2 * Re(scaled pole)
        let norm = 1.0 + d + c;
        out.push(BiquadCoeffs {
            b0: (gc * c / norm) as f32,
            b1: (gc * 2.0 * c / norm) as f32,
            b2: (gc * c / norm) as f32,
            a1: (2.0 * (c - 1.0) / norm) as f32,
            a2: ((1.0 - d + c) / norm) as f32,
        });
    }
    Ok(out)
}

/// Designs an Nth-order Chebyshev Type I highpass filter.
///
/// Returns a cascade of `order / 2` second-order sections.
///
/// # Parameters
///
/// * `order` — filter order; must be even, 2–20
/// * `ripple_db` — passband ripple in dB
/// * `cutoff_hz` — passband edge frequency in Hz
/// * `sample_rate` — sample rate in Hz
///
/// # Errors
///
/// Same conditions as [`chebyshev1_lowpass`].
#[cfg(feature = "alloc")]
pub fn chebyshev1_highpass(
    order: usize,
    ripple_db: f64,
    cutoff_hz: f64,
    sample_rate: f64,
) -> Result<Vec<BiquadCoeffs>, FilterError> {
    validate_cheby(order, ripple_db, cutoff_hz, sample_rate)?;

    let eps_sq = 10.0_f64.powf(ripple_db / 10.0) - 1.0;
    let gamma = (1.0 / eps_sq.sqrt()).asinh() / order as f64;
    let omega_p = (PI * cutoff_hz / sample_rate).tan();
    let gc = (1.0 + eps_sq).powf(-1.0 / order as f64);

    let half = order / 2;
    let mut out = Vec::with_capacity(half);
    for k in 1..=half {
        let theta = PI * (2 * k - 1) as f64 / (2 * order) as f64;
        let sigma = gamma.sinh() * theta.sin();
        let omega = gamma.cosh() * theta.cos();
        let omega0_sq = sigma * sigma + omega * omega;
        // LP-to-HP: scaled HP pole = omega_p / (normalized LP pole)
        // Re(HP pole) = -omega_p * sigma / omega0_sq
        let c = omega_p * omega_p / omega0_sq; // |HP pole|²
        let d = 2.0 * omega_p * sigma / omega0_sq; // -2 * Re(HP pole)
        let norm = 1.0 + d + c;
        out.push(BiquadCoeffs {
            b0: (gc / norm) as f32,
            b1: (-gc * 2.0 / norm) as f32,
            b2: (gc / norm) as f32,
            a1: (2.0 * (c - 1.0) / norm) as f32,
            a2: ((1.0 - d + c) / norm) as f32,
        });
    }
    Ok(out)
}

/// Designs an Nth-order Chebyshev Type II lowpass filter.
///
/// Type II has a maximally flat passband and equiripple stopband.
/// `cutoff_hz` is the **stopband** edge: below it the passband is flat;
/// at it the attenuation reaches exactly `stopband_db`.
///
/// Returns a cascade of `order / 2` second-order sections.
///
/// # Parameters
///
/// * `order` — filter order; must be even, 2–20
/// * `stopband_db` — minimum stopband attenuation in dB (e.g. `40.0` or `60.0`)
/// * `cutoff_hz` — stopband edge frequency in Hz
/// * `sample_rate` — sample rate in Hz
///
/// # Errors
///
/// Returns [`FilterError::InvalidParameter`] for out-of-range arguments.
#[cfg(feature = "alloc")]
pub fn chebyshev2_lowpass(
    order: usize,
    stopband_db: f64,
    cutoff_hz: f64,
    sample_rate: f64,
) -> Result<Vec<BiquadCoeffs>, FilterError> {
    validate_cheby(order, stopband_db, cutoff_hz, sample_rate)?;

    let eps_sq = 10.0_f64.powf(stopband_db / 10.0) - 1.0;
    // Type II prototype uses asinh(ε_s)/N, not asinh(1/ε_s)/N like Type I.
    let gamma = eps_sq.sqrt().asinh() / order as f64;
    let omega_s = (PI * cutoff_hz / sample_rate).tan();

    let half = order / 2;
    let mut out = Vec::with_capacity(half);
    for k in 1..=half {
        let theta = PI * (2 * k - 1) as f64 / (2 * order) as f64;
        let cos_theta = theta.cos(); // > 0 for all k, even N

        // Q_k: denominator of the inverted Type I LP pole. Note the sin/cos
        // roles are swapped compared to the Type I formula.
        let q = gamma.sinh() * gamma.sinh() * cos_theta * cos_theta
            + gamma.cosh() * gamma.cosh() * theta.sin() * theta.sin();

        // Type II LP poles: inverse of the pre-inversion poles.
        // Re(pole) = -sinh(γ)*cos(θ)/Q_k  (negative ✓)
        let c = omega_s * omega_s / q; // |scaled pole|²
        let d = 2.0 * omega_s * gamma.sinh() * cos_theta / q; // -2*Re(scaled pole)

        // Zeros at ±j * omega_s / cos(theta) — above the stopband edge.
        let e = omega_s * omega_s / (cos_theta * cos_theta);

        // gain = cos²(θ)/Q so each section has unity DC gain.
        let gain = cos_theta * cos_theta / q;
        let norm = 1.0 + d + c;
        out.push(BiquadCoeffs {
            b0: (gain * (1.0 + e) / norm) as f32,
            b1: (gain * 2.0 * (e - 1.0) / norm) as f32,
            b2: (gain * (1.0 + e) / norm) as f32,
            a1: (2.0 * (c - 1.0) / norm) as f32,
            a2: ((1.0 - d + c) / norm) as f32,
        });
    }
    Ok(out)
}

/// Designs an Nth-order Chebyshev Type II highpass filter.
///
/// `cutoff_hz` is the **stopband** edge: above it the passband is flat;
/// below it the stopband has equiripple attenuation of at least `stopband_db`.
///
/// Returns a cascade of `order / 2` second-order sections.
///
/// # Parameters
///
/// * `order` — filter order; must be even, 2–20
/// * `stopband_db` — minimum stopband attenuation in dB
/// * `cutoff_hz` — stopband edge frequency in Hz
/// * `sample_rate` — sample rate in Hz
///
/// # Errors
///
/// Returns [`FilterError::InvalidParameter`] for out-of-range arguments.
#[cfg(feature = "alloc")]
pub fn chebyshev2_highpass(
    order: usize,
    stopband_db: f64,
    cutoff_hz: f64,
    sample_rate: f64,
) -> Result<Vec<BiquadCoeffs>, FilterError> {
    validate_cheby(order, stopband_db, cutoff_hz, sample_rate)?;

    let eps_sq = 10.0_f64.powf(stopband_db / 10.0) - 1.0;
    // Same gamma formula as Type II LP.
    let gamma = eps_sq.sqrt().asinh() / order as f64;
    let omega_s = (PI * cutoff_hz / sample_rate).tan();

    let half = order / 2;
    let mut out = Vec::with_capacity(half);
    for k in 1..=half {
        let theta = PI * (2 * k - 1) as f64 / (2 * order) as f64;
        let cos_theta = theta.cos();

        // Same Q_k as Type II LP (sin/cos roles swapped vs Type I).
        let q = gamma.sinh() * gamma.sinh() * cos_theta * cos_theta
            + gamma.cosh() * gamma.cosh() * theta.sin() * theta.sin();

        // LP-to-HP transform of Type II LP: multiply poles by omega_s.
        // Re(HP pole) = omega_s * sinh(γ)*cos(θ) (stays negative after sign flip)
        // |HP pole|² = omega_s² * Q_k  (product, not quotient like LP)
        let c = omega_s * omega_s * q; // |HP pole|²
        let d = 2.0 * omega_s * gamma.sinh() * cos_theta; // -2 * Re(HP pole)

        // HP zeros at ±j * omega_s * cos(theta) (LP zeros inverted by LP→HP)
        let e = omega_s * omega_s * cos_theta * cos_theta; // HP zero freq²

        // gain = 1.0 → Nyquist gain = 1.0 per section
        let norm = 1.0 + d + c;
        out.push(BiquadCoeffs {
            b0: ((1.0 + e) / norm) as f32,
            b1: (2.0 * (e - 1.0) / norm) as f32,
            b2: ((1.0 + e) / norm) as f32,
            a1: (2.0 * (c - 1.0) / norm) as f32,
            a2: ((1.0 - d + c) / norm) as f32,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Biquad;

    const SR: f64 = 44100.0;

    #[cfg(feature = "alloc")]
    fn cascade_gain(sections: &[BiquadCoeffs], freq_hz: f32, sample_rate: f32) -> f32 {
        extern crate alloc;
        let mut filters: alloc::vec::Vec<Biquad> =
            sections.iter().map(|&c| Biquad::new(c)).collect();
        let phase_inc = 2.0 * core::f32::consts::PI * freq_hz / sample_rate;
        let mut peak = 0.0_f32;
        for i in 0..8000_usize {
            let x = (phase_inc * i as f32).sin();
            let mut y = x;
            for f in &mut filters {
                y = f.process_sample(y);
            }
            if i >= 4000 {
                peak = peak.max(y.abs());
            }
        }
        peak
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby1_lp_invalid_order() {
        assert!(chebyshev1_lowpass(0, 3.0, 1000.0, SR).is_err());
        assert!(chebyshev1_lowpass(1, 3.0, 1000.0, SR).is_err()); // odd
        assert!(chebyshev1_lowpass(3, 3.0, 1000.0, SR).is_err()); // odd
        assert!(chebyshev1_lowpass(22, 3.0, 1000.0, SR).is_err()); // > 20
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby1_lp_invalid_params() {
        assert!(chebyshev1_lowpass(4, 0.0, 1000.0, SR).is_err()); // ripple = 0
        assert!(chebyshev1_lowpass(4, 3.0, 0.0, SR).is_err()); // cutoff = 0
        assert!(chebyshev1_lowpass(4, 3.0, SR / 2.0, SR).is_err()); // at Nyquist
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby1_lp_section_count() {
        let s2 = chebyshev1_lowpass(2, 3.0, 1000.0, SR).unwrap();
        assert_eq!(s2.len(), 1);
        let s4 = chebyshev1_lowpass(4, 3.0, 1000.0, SR).unwrap();
        assert_eq!(s4.len(), 2);
        let s8 = chebyshev1_lowpass(8, 3.0, 1000.0, SR).unwrap();
        assert_eq!(s8.len(), 4);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby1_lp_coeffs_finite() {
        let sections = chebyshev1_lowpass(4, 3.0, 1000.0, SR).unwrap();
        for s in &sections {
            assert!(s.b0.is_finite() && s.b1.is_finite() && s.b2.is_finite());
            assert!(s.a1.is_finite() && s.a2.is_finite());
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby1_lp_attenuation_at_cutoff() {
        // For ripple_db = 3.0 dB, gain at fc = 1/sqrt(1 + eps²) ≈ −3.01 dB.
        // Tolerance: within 0.5 dB.
        let fc = 1000.0_f32;
        let sections = chebyshev1_lowpass(4, 3.0, fc as f64, SR as f64).unwrap();
        let gain = cascade_gain(&sections, fc, SR as f32);
        let gain_db = 20.0 * gain.log10();
        assert!(
            gain_db > -3.5 && gain_db < -2.5,
            "gain at cutoff = {gain_db:.2} dB, expected ≈ −3 dB"
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby1_lp_stopband_one_octave() {
        // One octave above fc: 4th-order → well over 24 dB attenuation.
        let fc = 1000.0_f32;
        let sections = chebyshev1_lowpass(4, 3.0, fc as f64, SR as f64).unwrap();
        let gain = cascade_gain(&sections, fc * 2.0, SR as f32);
        let gain_db = 20.0 * gain.log10();
        assert!(
            gain_db < -24.0,
            "gain at 2×fc = {gain_db:.2} dB, expected < −24 dB"
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby1_lp_passband_max_does_not_exceed_ripple() {
        let fc = 2000.0_f32;
        let sections = chebyshev1_lowpass(4, 1.0, fc as f64, SR as f64).unwrap();
        let gain = cascade_gain(&sections, fc * 0.1, SR as f32); // 10% of fc
        let gain_db = 20.0 * gain.log10();
        assert!(
            gain_db <= 0.1,
            "mid-passband gain = {gain_db:.2} dB, expected ≤ 0 dB"
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby1_hp_invalid_params() {
        assert!(chebyshev1_highpass(3, 3.0, 1000.0, SR).is_err()); // odd order
        assert!(chebyshev1_highpass(4, 0.0, 1000.0, SR).is_err());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby1_hp_attenuation_at_cutoff() {
        let fc = 8000.0_f32;
        let sections = chebyshev1_highpass(4, 3.0, fc as f64, SR as f64).unwrap();
        let gain = cascade_gain(&sections, fc, SR as f32);
        let gain_db = 20.0 * gain.log10();
        assert!(
            gain_db > -3.5 && gain_db < -2.5,
            "HP gain at cutoff = {gain_db:.2} dB, expected ≈ −3 dB"
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby1_hp_stopband_one_octave_below() {
        let fc = 8000.0_f32;
        let sections = chebyshev1_highpass(4, 3.0, fc as f64, SR as f64).unwrap();
        let gain = cascade_gain(&sections, fc / 2.0, SR as f32);
        let gain_db = 20.0 * gain.log10();
        assert!(
            gain_db < -24.0,
            "HP gain at fc/2 = {gain_db:.2} dB, expected < −24 dB"
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby1_hp_passes_high_frequency() {
        let fc = 1000.0_f32;
        let sections = chebyshev1_highpass(4, 3.0, fc as f64, SR as f64).unwrap();
        let gain = cascade_gain(&sections, 10000.0, SR as f32);
        assert!(gain > 0.5, "HP passband gain = {gain:.3}");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby2_lp_invalid_params() {
        assert!(chebyshev2_lowpass(3, 40.0, 1000.0, SR).is_err()); // odd
        assert!(chebyshev2_lowpass(4, 0.0, 1000.0, SR).is_err());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby2_lp_section_count() {
        assert_eq!(chebyshev2_lowpass(4, 40.0, 1000.0, SR).unwrap().len(), 2);
        assert_eq!(chebyshev2_lowpass(6, 60.0, 1000.0, SR).unwrap().len(), 3);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby2_lp_coeffs_finite() {
        let sections = chebyshev2_lowpass(4, 40.0, 1000.0, SR).unwrap();
        for s in &sections {
            assert!(s.b0.is_finite() && s.b1.is_finite() && s.b2.is_finite());
            assert!(s.a1.is_finite() && s.a2.is_finite());
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby2_lp_passes_low_frequency() {
        let fc = 5000.0_f32;
        let sections = chebyshev2_lowpass(4, 40.0, fc as f64, SR as f64).unwrap();
        let gain = cascade_gain(&sections, 200.0, SR as f32);
        assert!(
            gain > 0.9,
            "Type II LP low-freq gain = {gain:.4}, expected ≈ 1.0"
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby2_lp_stopband_at_two_x_cutoff() {
        let fc = 2000.0_f32;
        let stopband_db = 40.0_f32;
        let sections = chebyshev2_lowpass(4, stopband_db as f64, fc as f64, SR as f64).unwrap();
        let gain = cascade_gain(&sections, fc * 2.0, SR as f32);
        let gain_db = 20.0 * gain.log10();
        assert!(
            gain_db < -(stopband_db - 1.0),
            "Type II LP gain at 2×fc = {gain_db:.2} dB, expected < −{} dB",
            stopband_db - 1.0
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby2_hp_invalid_params() {
        assert!(chebyshev2_highpass(3, 40.0, 5000.0, SR).is_err()); // odd
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby2_hp_passes_high_frequency() {
        let fc = 2000.0_f32;
        let sections = chebyshev2_highpass(4, 40.0, fc as f64, SR as f64).unwrap();
        let gain = cascade_gain(&sections, 15000.0, SR as f32);
        assert!(
            gain > 0.9,
            "Type II HP high-freq gain = {gain:.4}, expected ≈ 1.0"
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn cheby2_hp_stopband_below_cutoff() {
        let fc = 4000.0_f32;
        let stopband_db = 40.0_f32;
        let sections = chebyshev2_highpass(4, stopband_db as f64, fc as f64, SR as f64).unwrap();
        let gain = cascade_gain(&sections, fc / 4.0, SR as f32);
        let gain_db = 20.0 * gain.log10();
        assert!(
            gain_db < -(stopband_db - 1.0),
            "Type II HP gain at fc/4 = {gain_db:.2} dB, expected < −{} dB",
            stopband_db - 1.0
        );
    }
}
