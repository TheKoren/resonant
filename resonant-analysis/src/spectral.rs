//! Spectral feature extraction: centroid, spread, flatness, rolloff.
//!
//! All functions operate on a pre-computed magnitude spectrum (positive
//! frequencies only). They do **not** perform an FFT — feed them the
//! output of [`SignalFreqExt::magnitude()`](resonant_fft::SignalFreqExt).

use crate::error::AnalysisError;

/// Spectral centroid: the weighted mean frequency of the spectrum.
///
/// Returns the "centre of mass" of the magnitude spectrum, in Hz.
/// A bright sound has a high centroid; a dull sound has a low one.
///
/// # Errors
///
/// Returns [`AnalysisError::EmptyInput`] if `magnitudes` is empty, or
/// [`AnalysisError::InvalidParameter`] if the slice lengths differ.
///
/// # Examples
///
/// ```
/// use resonant_analysis::spectral;
///
/// // Single peak at 1000 Hz
/// let mags = [0.0, 0.0, 1.0, 0.0];
/// let freqs = [0.0, 500.0, 1000.0, 1500.0];
/// let c = spectral::spectral_centroid(&mags, &freqs).unwrap();
/// assert!((c - 1000.0).abs() < 1e-4);
/// ```
#[must_use = "returns the centroid value"]
pub fn spectral_centroid(magnitudes: &[f32], frequencies: &[f32]) -> Result<f32, AnalysisError> {
    spectral_centroid_and_spread(magnitudes, frequencies).map(|(c, _)| c)
}

/// Spectral spread: the standard deviation of frequency around the centroid.
///
/// Measures how "wide" the spectrum is. A pure tone has near-zero spread;
/// noise has high spread.
///
/// # Errors
///
/// Returns [`AnalysisError::EmptyInput`] if `magnitudes` is empty, or
/// [`AnalysisError::InvalidParameter`] if the slice lengths differ.
///
/// # Examples
///
/// ```
/// use resonant_analysis::spectral;
///
/// // Single peak → spread ≈ 0
/// let mags = [0.0, 0.0, 1.0, 0.0];
/// let freqs = [0.0, 500.0, 1000.0, 1500.0];
/// let s = spectral::spectral_spread(&mags, &freqs).unwrap();
/// assert!(s < 1.0);
/// ```
#[must_use = "returns the spread value"]
pub fn spectral_spread(magnitudes: &[f32], frequencies: &[f32]) -> Result<f32, AnalysisError> {
    spectral_centroid_and_spread(magnitudes, frequencies).map(|(_, s)| s)
}

/// Spectral centroid and spread computed in a single pass.
///
/// Equivalent to calling [`spectral_centroid`] and [`spectral_spread`] separately
/// but traverses the slices only once. Prefer this when both values are needed.
///
/// Returns `(centroid_hz, spread_hz)`.
///
/// # Errors
///
/// Returns [`AnalysisError::EmptyInput`] if `magnitudes` is empty, or
/// [`AnalysisError::InvalidParameter`] if the slice lengths differ.
///
/// # Examples
///
/// ```
/// use resonant_analysis::spectral;
///
/// let mags = [0.0, 0.0, 1.0, 0.0];
/// let freqs = [0.0, 500.0, 1000.0, 1500.0];
/// let (c, s) = spectral::spectral_centroid_and_spread(&mags, &freqs).unwrap();
/// assert!((c - 1000.0).abs() < 1e-4);
/// assert!(s < 1.0);
/// ```
#[must_use = "returns (centroid_hz, spread_hz)"]
pub fn spectral_centroid_and_spread(
    magnitudes: &[f32],
    frequencies: &[f32],
) -> Result<(f32, f32), AnalysisError> {
    validate_inputs(magnitudes, frequencies)?;

    let total: f32 = magnitudes.iter().sum();
    if total <= f32::EPSILON {
        return Ok((0.0, 0.0));
    }

    // Accumulate weighted f and f² in one pass.
    // Var[f] = E[f²] − E[f]² avoids a second traversal once centroid is known.
    let (weighted_f, weighted_f2) = magnitudes
        .iter()
        .zip(frequencies)
        .fold((0.0_f32, 0.0_f32), |(wf, wf2), (m, f)| {
            (wf + m * f, wf2 + m * f * f)
        });

    let centroid = weighted_f / total;
    // Clamp to zero: floating-point rounding can produce a tiny negative variance.
    let variance = (weighted_f2 / total - centroid * centroid).max(0.0);

    Ok((centroid, variance.sqrt()))
}

/// Spectral flatness: ratio of geometric mean to arithmetic mean of magnitudes.
///
/// Ranges from 0.0 (tonal/peaked) to 1.0 (flat/noise-like).
/// Computed in the log domain for numerical stability.
///
/// # Errors
///
/// Returns [`AnalysisError::EmptyInput`] if `magnitudes` is empty.
///
/// # Examples
///
/// ```
/// use resonant_analysis::spectral;
///
/// // Flat spectrum → flatness ≈ 1.0
/// let mags = [1.0, 1.0, 1.0, 1.0];
/// let f = spectral::spectral_flatness(&mags).unwrap();
/// assert!((f - 1.0).abs() < 1e-4);
/// ```
#[must_use = "returns the flatness value"]
pub fn spectral_flatness(magnitudes: &[f32]) -> Result<f32, AnalysisError> {
    if magnitudes.is_empty() {
        return Err(AnalysisError::EmptyInput);
    }

    let n = magnitudes.len() as f32;
    let arith_mean: f32 = magnitudes.iter().sum::<f32>() / n;

    if arith_mean <= f32::EPSILON {
        return Ok(0.0);
    }

    // Geometric mean via log domain: exp(mean(log(x)))
    // Clamp to epsilon to avoid log(0)
    let log_sum: f32 = magnitudes.iter().map(|&m| (m.max(f32::EPSILON)).ln()).sum();
    let geo_mean = (log_sum / n).exp();

    Ok(geo_mean / arith_mean)
}

/// Spectral rolloff: the frequency below which `percent` of the total
/// spectral energy is contained.
///
/// `percent` should be in the range `0.0..=1.0` (e.g. `0.85` for the
/// 85th percentile). Returns the rolloff frequency in Hz.
///
/// # Errors
///
/// Returns [`AnalysisError::EmptyInput`] if `magnitudes` is empty,
/// [`AnalysisError::InvalidParameter`] if slice lengths differ or
/// `percent` is outside `0.0..=1.0`.
///
/// # Examples
///
/// ```
/// use resonant_analysis::spectral;
///
/// // All energy in the first bin
/// let mags = [1.0, 0.0, 0.0, 0.0];
/// let freqs = [100.0, 200.0, 300.0, 400.0];
/// let r = spectral::spectral_rolloff(&mags, &freqs, 0.85).unwrap();
/// assert!((r - 100.0).abs() < 1e-4);
/// ```
#[must_use = "returns the rolloff frequency"]
pub fn spectral_rolloff(
    magnitudes: &[f32],
    frequencies: &[f32],
    percent: f32,
) -> Result<f32, AnalysisError> {
    validate_inputs(magnitudes, frequencies)?;

    if !(0.0..=1.0).contains(&percent) {
        return Err(AnalysisError::InvalidParameter {
            name: "percent",
            reason: "must be between 0.0 and 1.0",
        });
    }

    let total: f32 = magnitudes.iter().sum();
    let threshold = total * percent;
    let mut cumsum = 0.0_f32;

    for (m, f) in magnitudes.iter().zip(frequencies) {
        cumsum += m;
        if cumsum >= threshold {
            return Ok(*f);
        }
    }

    // If we get here, return the last frequency
    Ok(*frequencies.last().unwrap_or(&0.0))
}

/// Validates that inputs are non-empty and lengths match.
fn validate_inputs(magnitudes: &[f32], frequencies: &[f32]) -> Result<(), AnalysisError> {
    if magnitudes.is_empty() {
        return Err(AnalysisError::EmptyInput);
    }
    if magnitudes.len() != frequencies.len() {
        return Err(AnalysisError::InvalidParameter {
            name: "frequencies",
            reason: "length must match magnitudes",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FREQS: [f32; 4] = [0.0, 500.0, 1000.0, 1500.0];

    #[test]
    fn centroid_single_peak() {
        let mags = [0.0, 0.0, 1.0, 0.0];
        let c = spectral_centroid(&mags, &FREQS).ok();
        assert!(c.is_some_and(|v| (v - 1000.0).abs() < 1e-4));
    }

    #[test]
    fn centroid_flat_spectrum() {
        let mags = [1.0, 1.0, 1.0, 1.0];
        let c = spectral_centroid(&mags, &FREQS).ok();
        // Mean of 0, 500, 1000, 1500 = 750
        assert!(c.is_some_and(|v| (v - 750.0).abs() < 1e-4));
    }

    #[test]
    fn centroid_empty() {
        let err = spectral_centroid(&[], &[]);
        assert_eq!(err, Err(AnalysisError::EmptyInput));
    }

    #[test]
    fn centroid_length_mismatch() {
        let err = spectral_centroid(&[1.0], &[1.0, 2.0]);
        assert!(matches!(err, Err(AnalysisError::InvalidParameter { .. })));
    }

    #[test]
    fn centroid_silence() {
        let mags = [0.0, 0.0, 0.0, 0.0];
        let c = spectral_centroid(&mags, &FREQS).ok();
        assert!(c.is_some_and(|v| v == 0.0));
    }

    #[test]
    fn spread_single_peak() {
        let mags = [0.0, 0.0, 1.0, 0.0];
        let s = spectral_spread(&mags, &FREQS).ok();
        assert!(s.is_some_and(|v| v < 1.0));
    }

    #[test]
    fn spread_flat_wider_than_peak() {
        let flat = [1.0, 1.0, 1.0, 1.0];
        let peak = [0.0, 0.0, 1.0, 0.0];
        let s_flat = spectral_spread(&flat, &FREQS).ok().unwrap_or(0.0);
        let s_peak = spectral_spread(&peak, &FREQS).ok().unwrap_or(f32::MAX);
        assert!(s_flat > s_peak);
    }

    #[test]
    fn spread_empty() {
        assert_eq!(spectral_spread(&[], &[]), Err(AnalysisError::EmptyInput));
    }

    #[test]
    fn spread_silence() {
        let mags = [0.0; 4];
        let s = spectral_spread(&mags, &FREQS).ok();
        assert!(s.is_some_and(|v| v == 0.0));
    }

    #[test]
    fn flatness_flat_spectrum() {
        let mags = [1.0, 1.0, 1.0, 1.0];
        let f = spectral_flatness(&mags).ok();
        assert!(f.is_some_and(|v| (v - 1.0).abs() < 1e-4));
    }

    #[test]
    fn flatness_peaked_spectrum() {
        let mags = [0.0, 0.0, 100.0, 0.0];
        let f = spectral_flatness(&mags).ok();
        // Heavily peaked → close to 0
        assert!(f.is_some_and(|v| v < 0.1));
    }

    #[test]
    fn flatness_empty() {
        assert_eq!(spectral_flatness(&[]), Err(AnalysisError::EmptyInput));
    }

    #[test]
    fn flatness_silence() {
        let mags = [0.0; 4];
        let f = spectral_flatness(&mags).ok();
        assert!(f.is_some_and(|v| v == 0.0));
    }

    #[test]
    fn flatness_in_range() {
        let mags = [0.5, 1.0, 0.2, 0.8];
        let f = spectral_flatness(&mags).ok();
        assert!(f.is_some_and(|v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn rolloff_energy_in_first_bin() {
        let mags = [1.0, 0.0, 0.0, 0.0];
        let r = spectral_rolloff(&mags, &FREQS, 0.85).ok();
        assert!(r.is_some_and(|v| (v - 0.0).abs() < 1e-4));
    }

    #[test]
    fn rolloff_flat_spectrum() {
        let mags = [1.0, 1.0, 1.0, 1.0];
        // 85% of 4.0 = 3.4 → need bins 0..3 (cumsum 3.0 < 3.4, cumsum 4.0 >= 3.4 at bin 3)
        let r = spectral_rolloff(&mags, &FREQS, 0.85).ok();
        assert!(r.is_some_and(|v| (v - 1500.0).abs() < 1e-4));
    }

    #[test]
    fn rolloff_empty() {
        assert_eq!(
            spectral_rolloff(&[], &[], 0.85),
            Err(AnalysisError::EmptyInput)
        );
    }

    #[test]
    fn rolloff_invalid_percent_high() {
        let mags = [1.0];
        let freqs = [100.0];
        let err = spectral_rolloff(&mags, &freqs, 1.5);
        assert!(matches!(err, Err(AnalysisError::InvalidParameter { .. })));
    }

    #[test]
    fn rolloff_invalid_percent_negative() {
        let mags = [1.0];
        let freqs = [100.0];
        let err = spectral_rolloff(&mags, &freqs, -0.1);
        assert!(matches!(err, Err(AnalysisError::InvalidParameter { .. })));
    }

    #[test]
    fn rolloff_zero_percent() {
        let mags = [1.0, 1.0, 1.0];
        let freqs = [100.0, 200.0, 300.0];
        let r = spectral_rolloff(&mags, &freqs, 0.0).ok();
        // 0% threshold → first bin
        assert!(r.is_some_and(|v| (v - 100.0).abs() < 1e-4));
    }

    #[test]
    fn rolloff_length_mismatch() {
        let err = spectral_rolloff(&[1.0], &[1.0, 2.0], 0.5);
        assert!(matches!(err, Err(AnalysisError::InvalidParameter { .. })));
    }

    #[test]
    fn centroid_and_spread_empty() {
        assert_eq!(
            spectral_centroid_and_spread(&[], &[]),
            Err(AnalysisError::EmptyInput)
        );
    }

    #[test]
    fn centroid_and_spread_length_mismatch() {
        let err = spectral_centroid_and_spread(&[1.0], &[1.0, 2.0]);
        assert!(matches!(err, Err(AnalysisError::InvalidParameter { .. })));
    }

    #[test]
    fn centroid_and_spread_single_peak() {
        // Single peak at 1000 Hz: centroid = 1000, spread = 0.
        let mags = [0.0, 0.0, 1.0, 0.0];
        let (c, s) = spectral_centroid_and_spread(&mags, &FREQS).unwrap();
        assert!((c - 1000.0).abs() < 1e-4, "centroid {c} ≠ 1000");
        assert!(s < 1e-4, "spread {s} should be ~0 for a single peak");
    }

    #[test]
    fn centroid_and_spread_silence() {
        let mags = [0.0_f32; 4];
        let (c, s) = spectral_centroid_and_spread(&mags, &FREQS).unwrap();
        assert!(c.is_finite() && c == 0.0);
        assert!(s.is_finite() && s == 0.0);
    }

    #[test]
    fn centroid_and_spread_matches_individual_functions() {
        // Both a flat and a peaked spectrum; verify combined result equals
        // calling the individual functions separately.
        let cases: &[(&[f32], &[f32])] = &[
            (&[1.0, 1.0, 1.0, 1.0], &FREQS),
            (&[0.0, 0.5, 1.0, 0.5], &FREQS),
            (&[1.0, 0.0, 0.0, 0.0], &FREQS),
        ];
        for (mags, freqs) in cases {
            let (c, s) = spectral_centroid_and_spread(mags, freqs).unwrap();
            let expected_c = spectral_centroid(mags, freqs).unwrap();
            let expected_s = spectral_spread(mags, freqs).unwrap();
            // Tolerance: single-pass variance formula vs two-pass; ±0.5 Hz is generous.
            assert!(
                (c - expected_c).abs() < 0.5,
                "centroid mismatch: {c} vs {expected_c}"
            );
            assert!(
                (s - expected_s).abs() < 0.5,
                "spread mismatch: {s} vs {expected_s}"
            );
        }
    }
}
