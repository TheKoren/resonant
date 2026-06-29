//! Mel scale utilities and filterbank construction.
//!
//! Internal helpers used by the MFCC extractor. Not part of the public API.

use alloc::vec;
use alloc::vec::Vec;

/// Converts frequency in Hz to mel scale.
pub(crate) fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Converts mel value to frequency in Hz.
pub(crate) fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// A triangular mel filterbank: `num_bands` filters spanning [0, sample_rate/2].
///
/// Returns a `Vec<Vec<f32>>` where each inner vec has length `fft_size / 2 + 1`
/// (one weight per FFT magnitude bin).
pub(crate) fn build_mel_filterbank(
    num_bands: usize,
    fft_size: usize,
    sample_rate: f32,
) -> Vec<Vec<f32>> {
    let num_bins = fft_size / 2 + 1;
    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(sample_rate / 2.0);

    // num_bands + 2 equally spaced points on the mel scale
    let num_points = num_bands + 2;
    let mel_points: Vec<f32> = (0..num_points)
        .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (num_points - 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz points to FFT bin indices (fractional)
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&hz| hz * fft_size as f32 / sample_rate)
        .collect();

    let mut filterbank = Vec::with_capacity(num_bands);
    for band in 0..num_bands {
        let left = bin_points[band];
        let center = bin_points[band + 1];
        let right = bin_points[band + 2];

        let mut weights = vec![0.0_f32; num_bins];
        for (bin, weight) in weights.iter_mut().enumerate() {
            let b = bin as f32;
            if b > left && b < center && center > left {
                *weight = (b - left) / (center - left);
            } else if b >= center && b < right && right > center {
                *weight = (right - b) / (right - center);
            }
        }
        filterbank.push(weights);
    }

    filterbank
}

/// Applies the mel filterbank to a magnitude spectrum, producing mel-band energies.
pub(crate) fn apply_mel_filterbank(magnitudes: &[f32], filterbank: &[Vec<f32>]) -> Vec<f32> {
    filterbank
        .iter()
        .map(|weights| {
            let len = magnitudes.len().min(weights.len());
            magnitudes[..len]
                .iter()
                .zip(&weights[..len])
                .map(|(m, w)| m * m * w) // power spectrum weighted
                .sum()
        })
        .collect()
}

/// Converts mel-band energies to log scale, with a floor to avoid log(0).
pub(crate) fn log_mel_energy(energies: &[f32]) -> Vec<f32> {
    const FLOOR: f32 = 1e-10;
    energies.iter().map(|&e| (e.max(FLOOR)).ln()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 44100.0;

    #[test]
    fn hz_to_mel_known_values() {
        // 0 Hz → 0 mel
        assert!((hz_to_mel(0.0)).abs() < 1e-4);
        // 1000 Hz → ~1000 mel (by design of the mel scale formula)
        let mel_1k = hz_to_mel(1000.0);
        assert!(
            (mel_1k - 1000.0).abs() < 100.0,
            "1000 Hz should be ~1000 mel, got {mel_1k}"
        );
    }

    #[test]
    fn mel_hz_round_trip() {
        for &hz in &[0.0, 100.0, 440.0, 1000.0, 8000.0, 22050.0] {
            let recovered = mel_to_hz(hz_to_mel(hz));
            assert!(
                (recovered - hz).abs() < 0.1,
                "round-trip failed for {hz} Hz: got {recovered}"
            );
        }
    }

    #[test]
    fn filterbank_shape() {
        let fb = build_mel_filterbank(26, 1024, SR);
        assert_eq!(fb.len(), 26);
        for band in &fb {
            assert_eq!(band.len(), 513); // 1024/2 + 1
                                         // All weights non-negative
            for &w in band {
                assert!(w >= 0.0, "negative filterbank weight: {w}");
            }
        }
    }

    #[test]
    fn filterbank_triangular_peaks_at_most_one() {
        let fb = build_mel_filterbank(26, 1024, SR);
        for band in &fb {
            for &w in band {
                assert!(w <= 1.0 + 1e-6, "weight exceeds 1.0: {w}");
            }
        }
    }

    #[test]
    fn log_mel_energy_floor() {
        let energies = vec![0.0, 1e-20, 1.0];
        let log_e = log_mel_energy(&energies);
        for &v in &log_e {
            assert!(v.is_finite(), "log_mel_energy produced non-finite: {v}");
        }
        // log(1.0) = 0
        assert!((log_e[2]).abs() < 1e-4);
    }

    #[test]
    fn apply_filterbank_basic() {
        // Single-band filterbank that weights bin 1 fully
        let filterbank = vec![vec![0.0, 1.0, 0.0]];
        let magnitudes = [0.5, 2.0, 0.3];
        let result = apply_mel_filterbank(&magnitudes, &filterbank);
        // Power of bin 1 * weight 1.0 = 2.0² * 1.0 = 4.0
        assert!((result[0] - 4.0).abs() < 1e-4);
    }
}
