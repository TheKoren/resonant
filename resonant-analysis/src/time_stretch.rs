//! Time-stretching and pitch-shifting via STFT phase vocoder.
//!
//! [`time_stretch`] combines phase-vocoder time-stretching with polyphase
//! resampling to provide independent control over duration and pitch.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use core::f32::consts::{LN_2, PI};

use resonant_fft::phase_vocoder::PhaseVocoder;
use resonant_fft::{fft, ifft, Complex};
use resonant_filters::resample::PolyphaseResampler;

use crate::error::AnalysisError;

/// Time-stretch or pitch-shift a mono signal.
///
/// `stretch_factor > 1.0` slows playback; `< 1.0` speeds up. `pitch_semitones`
/// shifts pitch in semitones (positive = up, negative = down); `0.0` leaves
/// pitch unchanged. Both controls are independent.
///
/// Internally: the phase vocoder stretches by `stretch_factor / pitch_ratio`,
/// then polyphase resampling introduces the pitch shift — the standard technique
/// for independently controlling duration and pitch.
///
/// # Errors
///
/// Returns [`AnalysisError::EmptyInput`] if `samples` is empty.
/// Returns [`AnalysisError::InvalidParameter`] if `sample_rate` or
/// `stretch_factor` is not positive.
pub fn time_stretch(
    samples: &[f32],
    sample_rate: f32,
    stretch_factor: f32,
    pitch_semitones: f32,
) -> Result<Vec<f32>, AnalysisError> {
    if samples.is_empty() {
        return Err(AnalysisError::EmptyInput);
    }
    if sample_rate <= 0.0 {
        return Err(AnalysisError::InvalidParameter {
            name: "sample_rate",
            reason: "must be positive",
        });
    }
    if stretch_factor <= 0.0 {
        return Err(AnalysisError::InvalidParameter {
            name: "stretch_factor",
            reason: "must be positive",
        });
    }

    // 2^(semitones/12) — the frequency ratio for the requested pitch shift.
    let pitch_ratio = (pitch_semitones / 12.0 * LN_2).exp();

    // Phase vocoder stretches by stretch_factor × pitch_ratio; polyphase
    // resampling then divides the sample count by pitch_ratio to restore
    // the target duration while introducing the frequency shift.
    let pv_output = run_phase_vocoder(samples, stretch_factor * pitch_ratio)?;

    if (pitch_ratio - 1.0).abs() < 1e-4 {
        return Ok(pv_output);
    }

    resample_pitch(pv_output, pitch_ratio)
}

/// Applies STFT-based phase vocoder time-stretching.
fn run_phase_vocoder(samples: &[f32], stretch_factor: f32) -> Result<Vec<f32>, AnalysisError> {
    let n = samples.len();
    let fft_size = choose_fft_size(n);
    let hop_a = fft_size / 4; // 75% overlap

    let mut pv = PhaseVocoder::new(fft_size, hop_a, stretch_factor)?;
    let hop_s = pv.hop_synthesis();
    let num_bins = fft_size / 2 + 1;

    let hann = hann_window(fft_size);

    let n_frames = n.saturating_sub(fft_size) / hop_a + 1;
    let out_len = n_frames * hop_s + fft_size;
    let mut output = vec![0.0_f32; out_len];
    let mut norm_buf = vec![0.0_f32; out_len];

    let mut frame_buf = vec![Complex::new(0.0_f32, 0.0_f32); fft_size];
    let mut mags = vec![0.0_f32; num_bins];
    let mut phases = vec![0.0_f32; num_bins];
    let mut out_mags = vec![0.0_f32; num_bins];
    let mut out_phases = vec![0.0_f32; num_bins];

    let mut in_pos = 0;
    let mut out_pos = 0;

    while in_pos + fft_size <= n {
        for (i, c) in frame_buf.iter_mut().enumerate() {
            c.re = samples[in_pos + i] * hann[i];
            c.im = 0.0;
        }

        fft(&mut frame_buf)?;

        for k in 0..num_bins {
            mags[k] = frame_buf[k].norm();
            phases[k] = frame_buf[k].arg();
        }

        pv.process_frame(&mags, &phases, &mut out_mags, &mut out_phases);

        // Reconstruct a conjugate-symmetric spectrum for real-valued IFFT output.
        for k in 0..num_bins {
            let (sin_p, cos_p) = out_phases[k].sin_cos();
            frame_buf[k] = Complex::new(out_mags[k] * cos_p, out_mags[k] * sin_p);
        }
        // DC and Nyquist bins must be purely real.
        frame_buf[0].im = 0.0;
        frame_buf[num_bins - 1].im = 0.0;
        // Mirror the positive-frequency bins into the upper half.
        for k in 1..num_bins - 1 {
            frame_buf[fft_size - k] = frame_buf[k].conj();
        }

        ifft(&mut frame_buf)?;

        // Overlap-add with synthesis Hann window; track per-sample normalization.
        for j in 0..fft_size {
            let w = hann[j];
            output[out_pos + j] += frame_buf[j].re * w;
            norm_buf[out_pos + j] += w * w;
        }

        in_pos += hop_a;
        out_pos += hop_s;
    }

    // Divide out the window power accumulation (handles non-uniform edge overlap).
    for (o, &norm) in output.iter_mut().zip(norm_buf.iter()) {
        if norm > 1e-8 {
            *o /= norm;
        }
    }

    // Trim to the expected output length.
    let expected = (n as f32 * stretch_factor).round() as usize;
    output.truncate(expected.min(output.len()));
    Ok(output)
}

/// Downsamples `samples` by `pitch_ratio` using a polyphase sinc filter.
///
/// Downsampling by `pitch_ratio` compresses the signal in time, raising the
/// apparent pitch by `pitch_ratio` when played at the original sample rate.
fn resample_pitch(samples: Vec<f32>, pitch_ratio: f32) -> Result<Vec<f32>, AnalysisError> {
    // down/up ≈ pitch_ratio with denominator Q.
    const Q: usize = 1000;
    let down = (pitch_ratio * Q as f32).round() as usize;
    if down == 0 {
        return Err(AnalysisError::InvalidParameter {
            name: "pitch_semitones",
            reason: "results in zero output rate",
        });
    }
    // up=Q, down=round(pitch_ratio*Q): output has input.len() * Q / down samples.
    let mut resampler =
        PolyphaseResampler::new(Q, down).ok_or(AnalysisError::InvalidParameter {
            name: "pitch_semitones",
            reason: "resampler construction failed for the requested pitch ratio",
        })?;
    Ok(resampler.process(&samples))
}

/// Returns the largest power of two ≤ half the signal length, clamped to [4, 2048].
fn choose_fft_size(signal_len: usize) -> usize {
    let max_fft = (signal_len / 2).clamp(4, 2048);
    // ilog2 rounds down, so 1 << ilog2(x) is the largest power of two ≤ x.
    1usize << max_fft.ilog2()
}

/// Hann window of length `n`.
fn hann_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0_f32; n];
    }
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / denom).cos())
        .collect()
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use crate::pitch::YinEstimator;

    fn sine_wave(freq_hz: f32, sample_rate: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn empty_input_returns_error() {
        assert!(matches!(
            time_stretch(&[], 44100.0, 1.0, 0.0),
            Err(AnalysisError::EmptyInput)
        ));
    }

    #[test]
    fn invalid_sample_rate_returns_error() {
        let sig = std::vec![0.0_f32; 256];
        assert!(matches!(
            time_stretch(&sig, 0.0, 1.0, 0.0),
            Err(AnalysisError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn invalid_stretch_factor_returns_error() {
        let sig = std::vec![0.0_f32; 256];
        assert!(matches!(
            time_stretch(&sig, 44100.0, 0.0, 0.0),
            Err(AnalysisError::InvalidParameter { .. })
        ));
        assert!(matches!(
            time_stretch(&sig, 44100.0, -1.0, 0.0),
            Err(AnalysisError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn passthrough_preserves_length_and_content() {
        let sr = 44100.0_f32;
        let freq = 440.0_f32;
        // 8192 samples → 13 frames at fft_size=2048, hop=512.
        let sig = sine_wave(freq, sr, 8192);
        let out =
            time_stretch(&sig, sr, 1.0, 0.0).unwrap_or_else(|e| panic!("passthrough failed: {e}"));

        assert_eq!(out.len(), sig.len(), "passthrough must not change length");

        // Interior samples (skip first and last 2 frames worth) should match.
        let fft_size = choose_fft_size(sig.len());
        let margin = fft_size * 2;
        if margin * 2 < sig.len() {
            let max_err = sig[margin..sig.len() - margin]
                .iter()
                .zip(out[margin..sig.len() - margin].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, |acc, x| acc.max(x));
            assert!(
                max_err < 0.05,
                "passthrough interior error {max_err:.4} > 0.05"
            );
        }
    }

    #[test]
    fn double_stretch_doubles_length() {
        let sr = 44100.0_f32;
        // Use 8192 samples so there are enough analysis frames (~13) for the
        // OLA output to approach the expected 2× length.
        let sig = sine_wave(440.0, sr, 8192);
        let out =
            time_stretch(&sig, sr, 2.0, 0.0).unwrap_or_else(|e| panic!("2× stretch failed: {e}"));

        let expected = sig.len() * 2;
        let ratio = out.len() as f32 / expected as f32;
        assert!(
            (0.9..=1.1).contains(&ratio),
            "2× stretch: output/expected length ratio {ratio:.3} outside [0.9, 1.1]"
        );
    }

    #[test]
    fn half_stretch_halves_length() {
        let sr = 44100.0_f32;
        let sig = sine_wave(440.0, sr, 8192);
        let out =
            time_stretch(&sig, sr, 0.5, 0.0).unwrap_or_else(|e| panic!("0.5× stretch failed: {e}"));

        let expected = sig.len() / 2;
        let ratio = out.len() as f32 / expected as f32;
        assert!(
            (0.9..=1.1).contains(&ratio),
            "0.5× stretch: output/expected length ratio {ratio:.3} outside [0.9, 1.1]"
        );
    }

    #[test]
    fn pitch_shift_up_one_octave() {
        let sr = 44100.0_f32;
        let freq = 440.0_f32;
        let sig = sine_wave(freq, sr, 44100); // 1 second

        let out = time_stretch(&sig, sr, 1.0, 12.0)
            .unwrap_or_else(|e| panic!("+12 semitones failed: {e}"));

        // Duration should be approximately unchanged.
        let len_ratio = out.len() as f32 / sig.len() as f32;
        assert!(
            (0.8..=1.2).contains(&len_ratio),
            "+12 semitones: length ratio {len_ratio:.3} outside [0.8, 1.2]"
        );

        // Pitch should be doubled: 440 Hz → 880 Hz.
        let est = YinEstimator::new(sr)
            .with_frequency_range(200.0, 2000.0)
            .estimate(&out)
            .ok();
        let detected = est.and_then(|e| e.frequency_hz);
        assert!(
            detected.is_some_and(|f| (f - freq * 2.0).abs() < 80.0),
            "+12 semitones: expected ~880 Hz, got {detected:?}"
        );
    }

    #[test]
    fn pitch_shift_down_one_octave() {
        let sr = 44100.0_f32;
        let freq = 440.0_f32;
        let sig = sine_wave(freq, sr, 44100); // 1 second

        let out = time_stretch(&sig, sr, 1.0, -12.0)
            .unwrap_or_else(|e| panic!("-12 semitones failed: {e}"));

        // Pitch should be halved: 440 Hz → 220 Hz.
        let est = YinEstimator::new(sr)
            .with_frequency_range(50.0, 500.0)
            .estimate(&out)
            .ok();
        let detected = est.and_then(|e| e.frequency_hz);
        assert!(
            detected.is_some_and(|f| (f - freq * 0.5).abs() < 40.0),
            "-12 semitones: expected ~220 Hz, got {detected:?}"
        );
    }

    #[test]
    fn hann_window_shape() {
        let w = hann_window(16);
        assert!(w[0].abs() < 1e-6, "first sample must be ~0");
        assert!(w[15].abs() < 1e-6, "last sample must be ~0");
        // For n=16 the maximum splits between indices 7 and 8 (centre at 7.5).
        let peak = w[7].max(w[8]);
        assert!(peak > 0.97, "Hann peak {peak:.4} should be > 0.97");
    }

    #[test]
    fn choose_fft_size_is_power_of_two_and_bounded() {
        for n in [16, 100, 512, 1024, 4096, 44100, 100000] {
            let fft_size = choose_fft_size(n);
            assert!(
                fft_size.is_power_of_two(),
                "fft_size {fft_size} is not a power of two"
            );
            assert!(fft_size >= 4, "fft_size {fft_size} < 4");
            assert!(fft_size <= 2048, "fft_size {fft_size} > 2048");
        }
    }
}
