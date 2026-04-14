//! Filter design helpers — compute biquad coefficients from audio parameters.
//!
//! Design arithmetic uses `f64` for precision; the resulting [`BiquadCoeffs`]
//! are `f32` for runtime efficiency.
//!
//! The Chebyshev functions ([`chebyshev1_lowpass`] etc.) require the `alloc`
//! feature and return a `Vec` of second-order sections to be cascaded.

use core::f64::consts::PI;

#[allow(unused_imports)]
use num_traits::float::Float as _;

use crate::BiquadCoeffs;

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use crate::response::FilterError;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

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

// ── Chebyshev design (alloc) ─────────────────────────────────────────────────

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

/// Validates parameters shared by all single-section EQ helpers.
fn validate_eq(freq_hz: f64, sample_rate: f64) -> Option<()> {
    if freq_hz <= 0.0 || sample_rate <= 0.0 || freq_hz >= sample_rate / 2.0 {
        None
    } else {
        Some(())
    }
}

/// Validates the Q factor for EQ helpers.
fn validate_q(q: f64) -> Option<()> {
    if q > 0.0 {
        Some(())
    } else {
        None
    }
}

/// Designs a low-shelf biquad: boosts or cuts frequencies below `cutoff_hz` by `gain_db`.
///
/// Uses the Audio EQ Cookbook formulas (R. Bristow-Johnson).
///
/// # Arguments
///
/// * `gain_db` — shelf gain in dB; positive boosts, negative cuts
/// * `cutoff_hz` — shelf midpoint frequency in Hz
/// * `sample_rate` — sample rate in Hz
///
/// # Errors
///
/// Returns `None` if `cutoff_hz` is out of range or `sample_rate` ≤ 0.
///
/// # Examples
///
/// ```
/// use resonant_filters::design;
/// let c = design::shelving_low(6.0, 200.0, 44100.0).unwrap();
/// assert!(c.b0.is_finite());
/// ```
pub fn shelving_low(gain_db: f32, cutoff_hz: f32, sample_rate: f32) -> Option<BiquadCoeffs> {
    let freq = cutoff_hz as f64;
    let sr = sample_rate as f64;
    validate_eq(freq, sr)?;

    let a = 10.0_f64.powf(gain_db as f64 / 40.0); // sqrt(10^(dB/20))
    let w0 = 2.0 * PI * freq / sr;
    let cos_w0 = w0.cos();
    // S = 1 (shelf slope parameter; 1 = maximum slope)
    // S = 1 (maximum shelf slope), so (A + 1/A)*(1/S − 1) + 2 = 2.
    let alpha = w0.sin() / 2.0 * 2.0_f64.sqrt();

    let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha);
    let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha);
    let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha;
    let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha;

    Some(BiquadCoeffs {
        b0: (b0 / a0) as f32,
        b1: (b1 / a0) as f32,
        b2: (b2 / a0) as f32,
        a1: (a1 / a0) as f32,
        a2: (a2 / a0) as f32,
    })
}

/// Designs a high-shelf biquad: boosts or cuts frequencies above `cutoff_hz` by `gain_db`.
///
/// Uses the Audio EQ Cookbook formulas (R. Bristow-Johnson).
///
/// # Arguments
///
/// * `gain_db` — shelf gain in dB; positive boosts, negative cuts
/// * `cutoff_hz` — shelf midpoint frequency in Hz
/// * `sample_rate` — sample rate in Hz
///
/// # Errors
///
/// Returns `None` if `cutoff_hz` is out of range or `sample_rate` ≤ 0.
///
/// # Examples
///
/// ```
/// use resonant_filters::design;
/// let c = design::shelving_high(6.0, 8000.0, 44100.0).unwrap();
/// assert!(c.b0.is_finite());
/// ```
pub fn shelving_high(gain_db: f32, cutoff_hz: f32, sample_rate: f32) -> Option<BiquadCoeffs> {
    let freq = cutoff_hz as f64;
    let sr = sample_rate as f64;
    validate_eq(freq, sr)?;

    let a = 10.0_f64.powf(gain_db as f64 / 40.0);
    let w0 = 2.0 * PI * freq / sr;
    let cos_w0 = w0.cos();
    // S = 1 (maximum shelf slope), so (A + 1/A)*(1/S − 1) + 2 = 2.
    let alpha = w0.sin() / 2.0 * 2.0_f64.sqrt();

    let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha);
    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha;

    Some(BiquadCoeffs {
        b0: (b0 / a0) as f32,
        b1: (b1 / a0) as f32,
        b2: (b2 / a0) as f32,
        a1: (a1 / a0) as f32,
        a2: (a2 / a0) as f32,
    })
}

/// Designs a peaking EQ biquad: boosts or cuts a band centred at `center_hz`.
///
/// Uses the Audio EQ Cookbook formulas (R. Bristow-Johnson).
///
/// # Arguments
///
/// * `gain_db` — peak gain in dB; positive boosts, negative cuts
/// * `center_hz` — centre frequency in Hz
/// * `q` — quality factor (bandwidth control); must be > 0
/// * `sample_rate` — sample rate in Hz
///
/// # Errors
///
/// Returns `None` if any parameter is out of range.
///
/// # Examples
///
/// ```
/// use resonant_filters::design;
/// let c = design::peaking_eq(6.0, 1000.0, 1.0, 44100.0).unwrap();
/// assert!(c.b0.is_finite());
/// ```
pub fn peaking_eq(gain_db: f32, center_hz: f32, q: f32, sample_rate: f32) -> Option<BiquadCoeffs> {
    let freq = center_hz as f64;
    let sr = sample_rate as f64;
    let qq = q as f64;
    validate_eq(freq, sr)?;
    validate_q(qq)?;

    let a = 10.0_f64.powf(gain_db as f64 / 40.0);
    let w0 = 2.0 * PI * freq / sr;
    let alpha = w0.sin() / (2.0 * qq);

    let b0 = 1.0 + alpha * a;
    let b1 = -2.0 * w0.cos();
    let b2 = 1.0 - alpha * a;
    let a0 = 1.0 + alpha / a;
    let a1 = -2.0 * w0.cos();
    let a2 = 1.0 - alpha / a;

    Some(BiquadCoeffs {
        b0: (b0 / a0) as f32,
        b1: (b1 / a0) as f32,
        b2: (b2 / a0) as f32,
        a1: (a1 / a0) as f32,
        a2: (a2 / a0) as f32,
    })
}

/// Designs a notch (band-reject) biquad centred at `center_hz`.
///
/// Uses the Audio EQ Cookbook formulas (R. Bristow-Johnson).
///
/// # Arguments
///
/// * `center_hz` — notch frequency in Hz
/// * `q` — quality factor; higher values give a narrower notch
/// * `sample_rate` — sample rate in Hz
///
/// # Errors
///
/// Returns `None` if any parameter is out of range.
///
/// # Examples
///
/// ```
/// use resonant_filters::design;
/// let c = design::notch(1000.0, 10.0, 44100.0).unwrap();
/// assert!(c.b0.is_finite());
/// ```
pub fn notch(center_hz: f32, q: f32, sample_rate: f32) -> Option<BiquadCoeffs> {
    let freq = center_hz as f64;
    let sr = sample_rate as f64;
    let qq = q as f64;
    validate_eq(freq, sr)?;
    validate_q(qq)?;

    let w0 = 2.0 * PI * freq / sr;
    let alpha = w0.sin() / (2.0 * qq);

    let b0 = 1.0;
    let b1 = -2.0 * w0.cos();
    let b2 = 1.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * w0.cos();
    let a2 = 1.0 - alpha;

    Some(BiquadCoeffs {
        b0: (b0 / a0) as f32,
        b1: (b1 / a0) as f32,
        b2: (b2 / a0) as f32,
        a1: (a1 / a0) as f32,
        a2: (a2 / a0) as f32,
    })
}

/// Designs an allpass biquad: unity magnitude at all frequencies, frequency-dependent phase.
///
/// Uses the Audio EQ Cookbook formulas (R. Bristow-Johnson).
///
/// # Arguments
///
/// * `center_hz` — frequency at which phase shift is −180° (90° shift above and below)
/// * `q` — quality factor controlling the rate of phase transition
/// * `sample_rate` — sample rate in Hz
///
/// # Errors
///
/// Returns `None` if any parameter is out of range.
///
/// # Examples
///
/// ```
/// use resonant_filters::design;
/// let c = design::allpass(1000.0, 0.707, 44100.0).unwrap();
/// assert!(c.b0.is_finite());
/// ```
pub fn allpass(center_hz: f32, q: f32, sample_rate: f32) -> Option<BiquadCoeffs> {
    let freq = center_hz as f64;
    let sr = sample_rate as f64;
    let qq = q as f64;
    validate_eq(freq, sr)?;
    validate_q(qq)?;

    let w0 = 2.0 * PI * freq / sr;
    let alpha = w0.sin() / (2.0 * qq);

    let b0 = 1.0 - alpha;
    let b1 = -2.0 * w0.cos();
    let b2 = 1.0 + alpha;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * w0.cos();
    let a2 = 1.0 - alpha;

    Some(BiquadCoeffs {
        b0: (b0 / a0) as f32,
        b1: (b1 / a0) as f32,
        b2: (b2 / a0) as f32,
        a1: (a1 / a0) as f32,
        a2: (a2 / a0) as f32,
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

    // ── Chebyshev helpers ────────────────────────────────────────────────────

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

    // ── Chebyshev I LP ───────────────────────────────────────────────────────

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
        // Maximum passband gain = 1.0 (0 dB) by construction; check at a
        // mid-passband frequency.
        let fc = 2000.0_f32;
        let sections = chebyshev1_lowpass(4, 1.0, fc as f64, SR as f64).unwrap();
        let gain = cascade_gain(&sections, fc * 0.1, SR as f32); // 10% of fc
        let gain_db = 20.0 * gain.log10();
        assert!(
            gain_db <= 0.1,
            "mid-passband gain = {gain_db:.2} dB, expected ≤ 0 dB"
        );
    }

    // ── Chebyshev I HP ───────────────────────────────────────────────────────

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
        // One octave below fc: should be well attenuated.
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
        // Should be within the passband ripple band
        assert!(gain > 0.5, "HP passband gain = {gain:.3}");
    }

    // ── Chebyshev II LP ──────────────────────────────────────────────────────

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
        // Passband is flat near DC; gain should be close to 1.0.
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
        // At 2× the stopband edge, attenuation must exceed stopband_db − 1 dB.
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

    // ── Chebyshev II HP ──────────────────────────────────────────────────────

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
        // Well below the stopband edge, attenuation must exceed stopband_db − 1 dB.
        let fc = 4000.0_f32;
        let stopband_db = 40.0_f32;
        let sections = chebyshev2_highpass(4, stopband_db as f64, fc as f64, SR as f64).unwrap();
        let gain = cascade_gain(&sections, fc / 4.0, SR as f32); // 1/4 of fc
        let gain_db = 20.0 * gain.log10();
        assert!(
            gain_db < -(stopband_db - 1.0),
            "Type II HP gain at fc/4 = {gain_db:.2} dB, expected < −{} dB",
            stopband_db - 1.0
        );
    }

    // ── EQ helper ────────────────────────────────────────────────────────────
    // Evaluates |H(e^jω)| analytically for a single biquad section.
    fn biquad_mag(c: &BiquadCoeffs, freq_hz: f32, sample_rate: f32) -> f32 {
        use core::f32::consts::PI as PI32;
        let w = 2.0 * PI32 * freq_hz / sample_rate;
        let (sin_w, cos_w) = (w.sin(), w.cos());
        let (sin_2w, cos_2w) = ((2.0 * w).sin(), (2.0 * w).cos());
        // numerator: b0 + b1*e^{-jω} + b2*e^{-2jω}
        let br = c.b0 + c.b1 * cos_w + c.b2 * cos_2w;
        let bi = -(c.b1 * sin_w + c.b2 * sin_2w);
        // denominator: 1 + a1*e^{-jω} + a2*e^{-2jω}
        let ar = 1.0 + c.a1 * cos_w + c.a2 * cos_2w;
        let ai = -(c.a1 * sin_w + c.a2 * sin_2w);
        ((br * br + bi * bi) / (ar * ar + ai * ai)).sqrt()
    }

    fn biquad_mag_db(c: &BiquadCoeffs, freq_hz: f32, sr: f32) -> f32 {
        20.0 * biquad_mag(c, freq_hz, sr).log10()
    }

    // DC and Nyquist helpers (avoid log10(0)).
    fn biquad_mag_dc(c: &BiquadCoeffs) -> f32 {
        // z=1: all z^{-k}=1
        let num = c.b0 + c.b1 + c.b2;
        let den = 1.0 + c.a1 + c.a2;
        (num / den).abs()
    }

    fn biquad_mag_nyquist(c: &BiquadCoeffs) -> f32 {
        // z=-1: z^{-k}=(-1)^k
        let num = c.b0 - c.b1 + c.b2;
        let den = 1.0 - c.a1 + c.a2;
        (num / den).abs()
    }

    // ── shelving_low tests ────────────────────────────────────────────────────

    #[test]
    fn shelf_low_dc_gain_matches_gain_db() {
        let gain_db = 12.0_f32;
        let c = shelving_low(gain_db, 100.0, SR as f32).unwrap();
        let dc_db = 20.0 * biquad_mag_dc(&c).log10();
        assert!(
            (dc_db - gain_db).abs() < 0.1,
            "low shelf DC gain = {dc_db:.2} dB, expected {gain_db} dB"
        );
    }

    #[test]
    fn shelf_low_cut_dc_gain_matches() {
        let gain_db = -6.0_f32;
        let c = shelving_low(gain_db, 200.0, SR as f32).unwrap();
        let dc_db = 20.0 * biquad_mag_dc(&c).log10();
        assert!(
            (dc_db - gain_db).abs() < 0.1,
            "low shelf cut DC = {dc_db:.2} dB, expected {gain_db}"
        );
    }

    #[test]
    fn shelf_low_invalid_params() {
        assert!(shelving_low(6.0, 0.0, SR as f32).is_none());
        assert!(shelving_low(6.0, SR as f32 / 2.0, SR as f32).is_none());
        assert!(shelving_low(6.0, 1000.0, 0.0).is_none());
    }

    // ── shelving_high tests ───────────────────────────────────────────────────

    #[test]
    fn shelf_high_nyquist_gain_matches_gain_db() {
        let gain_db = 6.0_f32;
        let c = shelving_high(gain_db, 8000.0, SR as f32).unwrap();
        let nyq_db = 20.0 * biquad_mag_nyquist(&c).log10();
        assert!(
            (nyq_db - gain_db).abs() < 0.1,
            "high shelf Nyquist = {nyq_db:.2} dB, expected {gain_db}"
        );
    }

    #[test]
    fn shelf_high_dc_is_unity() {
        let c = shelving_high(12.0, 8000.0, SR as f32).unwrap();
        let dc_db = 20.0 * biquad_mag_dc(&c).log10();
        assert!(
            dc_db.abs() < 0.1,
            "high shelf DC = {dc_db:.2} dB, expected ~0 dB"
        );
    }

    #[test]
    fn shelf_high_invalid_params() {
        assert!(shelving_high(6.0, 0.0, SR as f32).is_none());
        assert!(shelving_high(6.0, SR as f32 / 2.0, SR as f32).is_none());
    }

    // ── peaking_eq tests ──────────────────────────────────────────────────────

    #[test]
    fn peaking_eq_boost_at_center() {
        let gain_db = 6.0_f32;
        let fc = 1000.0_f32;
        let c = peaking_eq(gain_db, fc, 1.0, SR as f32).unwrap();
        let measured = biquad_mag_db(&c, fc, SR as f32);
        assert!(
            (measured - gain_db).abs() < 0.15,
            "peaking boost at fc = {measured:.2} dB, expected {gain_db}"
        );
    }

    #[test]
    fn peaking_eq_dc_is_unity() {
        let c = peaking_eq(6.0, 1000.0, 1.0, SR as f32).unwrap();
        let dc_db = 20.0 * biquad_mag_dc(&c).log10();
        assert!(dc_db.abs() < 0.05, "peaking DC = {dc_db:.3} dB, expected 0");
    }

    #[test]
    fn peaking_eq_nyquist_is_unity() {
        let c = peaking_eq(6.0, 1000.0, 1.0, SR as f32).unwrap();
        let nyq_db = 20.0 * biquad_mag_nyquist(&c).log10();
        assert!(
            nyq_db.abs() < 0.05,
            "peaking Nyquist = {nyq_db:.3} dB, expected 0"
        );
    }

    #[test]
    fn peaking_eq_cut_at_center() {
        let gain_db = -12.0_f32;
        let fc = 2000.0_f32;
        let c = peaking_eq(gain_db, fc, 2.0, SR as f32).unwrap();
        let measured = biquad_mag_db(&c, fc, SR as f32);
        assert!(
            (measured - gain_db).abs() < 0.2,
            "peaking cut at fc = {measured:.2} dB, expected {gain_db}"
        );
    }

    #[test]
    fn peaking_eq_invalid_params() {
        assert!(peaking_eq(6.0, 0.0, 1.0, SR as f32).is_none()); // zero freq
        assert!(peaking_eq(6.0, 1000.0, 0.0, SR as f32).is_none()); // zero Q
        assert!(peaking_eq(6.0, 1000.0, 1.0, 0.0).is_none()); // zero sr
    }

    // ── notch tests ───────────────────────────────────────────────────────────

    #[test]
    fn notch_deep_attenuation_at_center() {
        let fc = 1000.0_f32;
        let c = notch(fc, 10.0, SR as f32).unwrap();
        let mag = biquad_mag(&c, fc, SR as f32);
        assert!(
            mag < 0.01,
            "notch magnitude at fc = {mag:.5}, expected < 0.01"
        );
    }

    #[test]
    fn notch_dc_is_unity() {
        let c = notch(1000.0, 10.0, SR as f32).unwrap();
        let dc = biquad_mag_dc(&c);
        assert!((dc - 1.0).abs() < 0.001, "notch DC = {dc:.4}, expected 1.0");
    }

    #[test]
    fn notch_nyquist_is_unity() {
        let c = notch(1000.0, 10.0, SR as f32).unwrap();
        let nyq = biquad_mag_nyquist(&c);
        assert!(
            (nyq - 1.0).abs() < 0.001,
            "notch Nyquist = {nyq:.4}, expected 1.0"
        );
    }

    #[test]
    fn notch_invalid_params() {
        assert!(notch(0.0, 1.0, SR as f32).is_none());
        assert!(notch(1000.0, 0.0, SR as f32).is_none());
        assert!(notch(1000.0, 1.0, 0.0).is_none());
    }

    // ── allpass tests ─────────────────────────────────────────────────────────

    #[test]
    fn allpass_unity_magnitude_at_multiple_frequencies() {
        let c = allpass(1000.0, 0.707, SR as f32).unwrap();
        for freq in [100.0_f32, 500.0, 1000.0, 4000.0, 10000.0, 20000.0] {
            if freq < SR as f32 / 2.0 {
                let mag = biquad_mag(&c, freq, SR as f32);
                assert!(
                    (mag - 1.0).abs() < 0.001,
                    "allpass |H| at {freq} Hz = {mag:.5}, expected 1.0"
                );
            }
        }
    }

    #[test]
    fn allpass_dc_is_unity() {
        let c = allpass(1000.0, 0.707, SR as f32).unwrap();
        let dc = biquad_mag_dc(&c);
        assert!(
            (dc - 1.0).abs() < 0.001,
            "allpass DC = {dc:.4}, expected 1.0"
        );
    }

    #[test]
    fn allpass_nyquist_is_unity() {
        let c = allpass(1000.0, 0.707, SR as f32).unwrap();
        let nyq = biquad_mag_nyquist(&c);
        assert!(
            (nyq - 1.0).abs() < 0.001,
            "allpass Nyquist = {nyq:.4}, expected 1.0"
        );
    }

    #[test]
    fn allpass_invalid_params() {
        assert!(allpass(0.0, 1.0, SR as f32).is_none());
        assert!(allpass(1000.0, -1.0, SR as f32).is_none());
        assert!(allpass(1000.0, 1.0, 0.0).is_none());
    }
}
