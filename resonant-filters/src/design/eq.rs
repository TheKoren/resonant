use core::f64::consts::PI;

#[allow(unused_imports)]
use num_traits::float::Float as _;

use crate::BiquadCoeffs;

use super::DesignError;

fn validate_eq(freq_hz: f64, sample_rate: f64) -> Result<(), DesignError> {
    if sample_rate <= 0.0 {
        return Err(DesignError::SampleRateOutOfRange);
    }
    if freq_hz <= 0.0 || freq_hz >= sample_rate / 2.0 {
        return Err(DesignError::FrequencyOutOfRange);
    }
    Ok(())
}

fn validate_q(q: f64) -> Result<(), DesignError> {
    if q > 0.0 {
        Ok(())
    } else {
        Err(DesignError::QOutOfRange)
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
/// Returns [`DesignError::SampleRateOutOfRange`] if `sample_rate` ≤ 0.
/// Returns [`DesignError::FrequencyOutOfRange`] if `cutoff_hz` is zero, negative, or ≥ Nyquist.
///
/// # Examples
///
/// ```
/// use resonant_filters::design;
/// let c = design::shelving_low(6.0, 200.0, 44100.0).unwrap();
/// assert!(c.b0.is_finite());
/// ```
pub fn shelving_low(
    gain_db: f32,
    cutoff_hz: f32,
    sample_rate: f32,
) -> Result<BiquadCoeffs, DesignError> {
    let freq = cutoff_hz as f64;
    let sr = sample_rate as f64;
    validate_eq(freq, sr)?;

    let a = 10.0_f64.powf(gain_db as f64 / 40.0); // sqrt(10^(dB/20))
    let w0 = 2.0 * PI * freq / sr;
    let cos_w0 = w0.cos();
    // S = 1 (maximum shelf slope), so (A + 1/A)*(1/S − 1) + 2 = 2.
    let alpha = w0.sin() / 2.0 * 2.0_f64.sqrt();

    let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha);
    let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha);
    let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha;
    let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha;

    Ok(BiquadCoeffs {
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
/// Returns [`DesignError::SampleRateOutOfRange`] if `sample_rate` ≤ 0.
/// Returns [`DesignError::FrequencyOutOfRange`] if `cutoff_hz` is zero, negative, or ≥ Nyquist.
///
/// # Examples
///
/// ```
/// use resonant_filters::design;
/// let c = design::shelving_high(6.0, 8000.0, 44100.0).unwrap();
/// assert!(c.b0.is_finite());
/// ```
pub fn shelving_high(
    gain_db: f32,
    cutoff_hz: f32,
    sample_rate: f32,
) -> Result<BiquadCoeffs, DesignError> {
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

    Ok(BiquadCoeffs {
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
/// Returns [`DesignError::SampleRateOutOfRange`] if `sample_rate` ≤ 0.
/// Returns [`DesignError::FrequencyOutOfRange`] if `center_hz` is zero, negative, or ≥ Nyquist.
/// Returns [`DesignError::QOutOfRange`] if `q` ≤ 0.
///
/// # Examples
///
/// ```
/// use resonant_filters::design;
/// let c = design::peaking_eq(6.0, 1000.0, 1.0, 44100.0).unwrap();
/// assert!(c.b0.is_finite());
/// ```
pub fn peaking_eq(
    gain_db: f32,
    center_hz: f32,
    q: f32,
    sample_rate: f32,
) -> Result<BiquadCoeffs, DesignError> {
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

    Ok(BiquadCoeffs {
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
/// Returns [`DesignError::SampleRateOutOfRange`] if `sample_rate` ≤ 0.
/// Returns [`DesignError::FrequencyOutOfRange`] if `center_hz` is zero, negative, or ≥ Nyquist.
/// Returns [`DesignError::QOutOfRange`] if `q` ≤ 0.
///
/// # Examples
///
/// ```
/// use resonant_filters::design;
/// let c = design::notch(1000.0, 10.0, 44100.0).unwrap();
/// assert!(c.b0.is_finite());
/// ```
pub fn notch(center_hz: f32, q: f32, sample_rate: f32) -> Result<BiquadCoeffs, DesignError> {
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

    Ok(BiquadCoeffs {
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
/// Returns [`DesignError::SampleRateOutOfRange`] if `sample_rate` ≤ 0.
/// Returns [`DesignError::FrequencyOutOfRange`] if `center_hz` is zero, negative, or ≥ Nyquist.
/// Returns [`DesignError::QOutOfRange`] if `q` ≤ 0.
///
/// # Examples
///
/// ```
/// use resonant_filters::design;
/// let c = design::allpass(1000.0, 0.707, 44100.0).unwrap();
/// assert!(c.b0.is_finite());
/// ```
pub fn allpass(center_hz: f32, q: f32, sample_rate: f32) -> Result<BiquadCoeffs, DesignError> {
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

    Ok(BiquadCoeffs {
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

    const SR: f64 = 44100.0;

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
        assert_eq!(
            shelving_low(6.0, 0.0, SR as f32),
            Err(DesignError::FrequencyOutOfRange)
        );
        assert_eq!(
            shelving_low(6.0, SR as f32 / 2.0, SR as f32),
            Err(DesignError::FrequencyOutOfRange)
        );
        assert_eq!(
            shelving_low(6.0, 1000.0, 0.0),
            Err(DesignError::SampleRateOutOfRange)
        );
    }

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
        assert_eq!(
            shelving_high(6.0, 0.0, SR as f32),
            Err(DesignError::FrequencyOutOfRange)
        );
        assert_eq!(
            shelving_high(6.0, SR as f32 / 2.0, SR as f32),
            Err(DesignError::FrequencyOutOfRange)
        );
    }

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
        assert_eq!(
            peaking_eq(6.0, 0.0, 1.0, SR as f32),
            Err(DesignError::FrequencyOutOfRange)
        );
        assert_eq!(
            peaking_eq(6.0, 1000.0, 0.0, SR as f32),
            Err(DesignError::QOutOfRange)
        );
        assert_eq!(
            peaking_eq(6.0, 1000.0, 1.0, 0.0),
            Err(DesignError::SampleRateOutOfRange)
        );
    }

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
        assert_eq!(
            notch(0.0, 1.0, SR as f32),
            Err(DesignError::FrequencyOutOfRange)
        );
        assert_eq!(notch(1000.0, 0.0, SR as f32), Err(DesignError::QOutOfRange));
        assert_eq!(
            notch(1000.0, 1.0, 0.0),
            Err(DesignError::SampleRateOutOfRange)
        );
    }

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
        assert_eq!(
            allpass(0.0, 1.0, SR as f32),
            Err(DesignError::FrequencyOutOfRange)
        );
        assert_eq!(
            allpass(1000.0, -1.0, SR as f32),
            Err(DesignError::QOutOfRange)
        );
        assert_eq!(
            allpass(1000.0, 1.0, 0.0),
            Err(DesignError::SampleRateOutOfRange)
        );
    }
}
