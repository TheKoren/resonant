//! Simple loudness analysis: RMS, peak, and crest factor.
//!
//! All methods operate on a mono `f32` sample slice and return scalar
//! measurements. dBFS values use the convention that a full-scale sine
//! peak of 1.0 = 0 dBFS. Silence is clamped to −120 dBFS.

/// Minimum dBFS value returned for silence or near-silence.
const SILENCE_DB: f32 = -120.0;

/// Loudness analyser for mono audio buffers.
///
/// Stateless — all methods take a sample slice and return a measurement.
/// Construct once and reuse; there is no internal state to reset.
///
/// # Examples
///
/// ```
/// use resonant_analysis::loudness::LoudnessAnalyser;
/// use core::f32::consts::PI;
///
/// let sr = 44100_usize;
/// let samples: Vec<f32> = (0..sr)
///     .map(|i| (2.0 * PI * 440.0 * i as f32 / sr as f32).sin())
///     .collect();
///
/// let la = LoudnessAnalyser::new();
/// let rms_db = la.rms_db(&samples);
/// assert!((rms_db - (-3.01)).abs() < 0.1, "expected ≈ −3.01 dBFS, got {rms_db:.3}");
/// ```
pub struct LoudnessAnalyser;

impl LoudnessAnalyser {
    /// Creates a new analyser.
    #[must_use]
    pub fn new() -> Self {
        LoudnessAnalyser
    }

    /// Root mean square level. Returns 0.0 for empty input.
    #[must_use]
    pub fn rms(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
        (sum_sq / samples.len() as f32).sqrt()
    }

    /// Absolute peak sample value. Returns 0.0 for empty input.
    #[must_use]
    pub fn peak(&self, samples: &[f32]) -> f32 {
        samples.iter().map(|s| s.abs()).fold(0.0_f32, f32::max)
    }

    /// Peak level in dBFS. Returns [`SILENCE_DB`] (−120) for silence or empty input.
    #[must_use]
    pub fn peak_db(&self, samples: &[f32]) -> f32 {
        let p = self.peak(samples);
        lin_to_db(p)
    }

    /// RMS level in dBFS. Returns [`SILENCE_DB`] (−120) for silence or empty input.
    #[must_use]
    pub fn rms_db(&self, samples: &[f32]) -> f32 {
        lin_to_db(self.rms(samples))
    }

    /// Crest factor in dB: `peak_db − rms_db`.
    ///
    /// Indicates dynamic range. A square wave has a crest factor of 0 dB;
    /// a sine wave ≈ 3 dB; highly dynamic content is much higher.
    ///
    /// Returns 0.0 for empty or silent input.
    #[must_use]
    pub fn crest_factor_db(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let p = self.peak_db(samples);
        let r = self.rms_db(samples);
        // Both are SILENCE_DB for silence → difference is 0.
        p - r
    }
}

impl Default for LoudnessAnalyser {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts a linear amplitude to dBFS, clamped to [`SILENCE_DB`].
fn lin_to_db(lin: f32) -> f32 {
    if lin <= 0.0 {
        return SILENCE_DB;
    }
    let db = 20.0 * lin.log10();
    if db < SILENCE_DB {
        SILENCE_DB
    } else {
        db
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::f32::consts::PI;

    const SR: usize = 44100;

    fn sine(freq_hz: f32, amplitude: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| amplitude * (2.0 * PI * freq_hz * i as f32 / SR as f32).sin())
            .collect()
    }

    fn dc(level: f32, n: usize) -> Vec<f32> {
        vec![level; n]
    }

    fn square(amplitude: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| if i % 2 == 0 { amplitude } else { -amplitude })
            .collect()
    }

    #[test]
    fn empty_rms_is_zero() {
        assert_eq!(LoudnessAnalyser::new().rms(&[]), 0.0);
    }

    #[test]
    fn empty_peak_is_zero() {
        assert_eq!(LoudnessAnalyser::new().peak(&[]), 0.0);
    }

    #[test]
    fn empty_peak_db_is_silence() {
        assert_eq!(LoudnessAnalyser::new().peak_db(&[]), SILENCE_DB);
    }

    #[test]
    fn silence_peak_db_is_silence_floor() {
        let la = LoudnessAnalyser::new();
        assert_eq!(la.peak_db(&[0.0; 1024]), SILENCE_DB);
    }

    #[test]
    fn silence_rms_db_is_silence_floor() {
        let la = LoudnessAnalyser::new();
        assert_eq!(la.rms_db(&[0.0; 1024]), SILENCE_DB);
    }

    #[test]
    fn full_scale_sine_rms_approx_minus_3_db() {
        // A sine at peak amplitude 1.0 has RMS = 1/√2 ≈ −3.01 dBFS.
        let la = LoudnessAnalyser::new();
        let samples = sine(440.0, 1.0, SR);
        let rms_db = la.rms_db(&samples);
        assert!(
            (rms_db - (-3.01)).abs() < 0.1,
            "sine RMS = {rms_db:.3} dBFS, expected ≈ −3.01"
        );
    }

    #[test]
    fn full_scale_sine_peak_approx_zero_db() {
        let la = LoudnessAnalyser::new();
        let samples = sine(440.0, 1.0, SR);
        let p = la.peak_db(&samples);
        // peak will be very close to 1.0 for a long enough sine
        assert!(p > -0.1, "sine peak = {p:.3} dBFS, expected ≈ 0");
    }

    #[test]
    fn dc_rms_equals_dc_level() {
        let la = LoudnessAnalyser::new();
        let level = 0.5_f32;
        let samples = dc(level, 1024);
        // RMS of a constant signal equals its absolute value
        let rms = la.rms(&samples);
        assert!(
            (rms - level).abs() < 1e-5,
            "DC RMS = {rms}, expected {level}"
        );
    }

    #[test]
    fn dc_rms_db_matches_linear() {
        let la = LoudnessAnalyser::new();
        let level = 0.5_f32;
        let expected_db = 20.0 * level.log10(); // ≈ −6.02 dBFS
        let measured = la.rms_db(&dc(level, 1024));
        assert!(
            (measured - expected_db).abs() < 0.01,
            "DC RMS dB = {measured:.3}, expected {expected_db:.3}"
        );
    }

    #[test]
    fn square_wave_crest_factor_near_zero() {
        // A full-scale square wave: peak = RMS = amplitude → crest factor = 0 dB.
        let la = LoudnessAnalyser::new();
        let samples = square(1.0, 4096);
        let cf = la.crest_factor_db(&samples);
        assert!(
            cf.abs() < 0.01,
            "square wave crest factor = {cf:.4} dB, expected ≈ 0"
        );
    }

    #[test]
    fn sine_crest_factor_near_3_db() {
        // Sine crest factor = peak/RMS = 1.0 / (1/√2) = √2 ≈ 3.01 dB.
        let la = LoudnessAnalyser::new();
        let samples = sine(440.0, 1.0, SR);
        let cf = la.crest_factor_db(&samples);
        assert!(
            (cf - 3.01).abs() < 0.1,
            "sine crest factor = {cf:.3} dB, expected ≈ 3.01"
        );
    }

    #[test]
    fn peak_of_mixed_signs() {
        let la = LoudnessAnalyser::new();
        let samples = [-0.8_f32, 0.3, -0.5, 0.9, -0.1];
        assert!((la.peak(&samples) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn rms_known_values() {
        // [3, 4] → RMS = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        let la = LoudnessAnalyser::new();
        let samples = [3.0_f32, 4.0];
        let expected = (12.5_f32).sqrt();
        assert!((la.rms(&samples) - expected).abs() < 1e-5);
    }

    #[test]
    fn empty_crest_factor_is_zero() {
        assert_eq!(LoudnessAnalyser::new().crest_factor_db(&[]), 0.0);
    }
}
