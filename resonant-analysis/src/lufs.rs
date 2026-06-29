//! ITU-R BS.1770-4 integrated loudness (LUFS / LKFS) measurement.
//!
//! Applies K-weighting (a two-stage biquad cascade), computes mean-square
//! levels over 400 ms gating blocks, then applies the absolute and relative
//! gates defined in the standard to produce integrated loudness.

use alloc::vec;
use alloc::vec::Vec;

use resonant_filters::biquad::{Biquad, BiquadCoeffs};
use resonant_filters::PolyphaseResampler;

use crate::AnalysisError;

/// Channel configuration for multi-channel loudness measurement.
///
/// Per ITU-R BS.1770-4, each channel is K-weighted independently; the gating
/// operates on the weighted sum of per-channel mean-square levels.
///
/// Standard weights: L=1.0, R=1.0, C=1.0, LFE=0.0, Ls=√2, Rs=√2.
///
/// Use [`mono`], [`stereo`], [`surround_5_1`], or [`custom`] to construct.
/// Per-channel weights (linear, not dB) are readable via [`weights`].
///
/// [`mono`]: ChannelConfig::mono
/// [`stereo`]: ChannelConfig::stereo
/// [`surround_5_1`]: ChannelConfig::surround_5_1
/// [`custom`]: ChannelConfig::custom
/// [`weights`]: ChannelConfig::weights
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    weights: Vec<f32>,
}

impl ChannelConfig {
    /// Mono passthrough (weight = 1.0).
    #[must_use]
    pub fn mono() -> Self {
        Self { weights: vec![1.0] }
    }

    /// Stereo per BS.1770-4 (L=1.0, R=1.0).
    #[must_use]
    pub fn stereo() -> Self {
        Self {
            weights: vec![1.0, 1.0],
        }
    }

    /// 5.1 surround per BS.1770-4 channel order: L, R, C, LFE, Ls, Rs.
    ///
    /// LFE is excluded (weight 0.0). Surround channels Ls and Rs are weighted
    /// at √2 ≈ 1.4142 (3 dB higher than front channels per the standard).
    #[must_use]
    pub fn surround_5_1() -> Self {
        use core::f32::consts::SQRT_2;
        Self {
            weights: vec![1.0, 1.0, 1.0, 0.0, SQRT_2, SQRT_2],
        }
    }

    /// Creates a `ChannelConfig` with custom per-channel weights.
    ///
    /// Returns `None` if `weights` is empty or any weight is non-finite
    /// (NaN or ±∞). Zero weights are permitted (e.g. LFE exclusion).
    ///
    /// # Examples
    ///
    /// ```
    /// use resonant_analysis::lufs::ChannelConfig;
    ///
    /// // Equal-weight quad: L, R, Ls, Rs
    /// let cfg = ChannelConfig::custom(vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    /// assert_eq!(cfg.weights(), &[1.0, 1.0, 1.0, 1.0]);
    ///
    /// assert!(ChannelConfig::custom(vec![]).is_none());
    /// assert!(ChannelConfig::custom(vec![f32::NAN]).is_none());
    /// ```
    #[must_use]
    pub fn custom(weights: Vec<f32>) -> Option<Self> {
        if weights.is_empty() || !weights.iter().all(|w| w.is_finite()) {
            return None;
        }
        Some(Self { weights })
    }

    /// Returns the per-channel gain weights (linear, not dB).
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
}

/// Minimum loudness value returned for silence or near-silence.
const SILENCE_LUFS: f32 = -120.0;

/// Absolute gating threshold per ITU-R BS.1770-4 §2.7.
const ABS_GATE_LUFS: f32 = -70.0;

/// Relative gate offset: 10 LU below the absolute-gated mean.
const REL_GATE_OFFSET_LU: f32 = 10.0;

// K-weighting filter coefficients (ITU-R BS.1770-4 Annex 1).
//
// Stage 1 — high-shelf pre-filter: compensates for the acoustic effect of
// sound diffracted around the human head (~+4 dB above 1.5 kHz).
// Stage 2 — RLB high-pass: suppresses sub-bass content below ~40 Hz.

const PRE_44100: BiquadCoeffs = BiquadCoeffs {
    b0: 1.530_926_5_f32,
    b1: -2.651_179_f32,
    b2: 1.169_068_2_f32,
    a1: -1.663_758_3_f32,
    a2: 0.712_653_96_f32,
};
const RLB_44100: BiquadCoeffs = BiquadCoeffs {
    b0: 1.0_f32,
    b1: -2.0_f32,
    b2: 1.0_f32,
    a1: -1.988_378_9_f32,
    a2: 0.988_520_47_f32,
};
const PRE_48000: BiquadCoeffs = BiquadCoeffs {
    b0: 1.535_124_9_f32,
    b1: -2.691_696_2_f32,
    b2: 1.198_392_9_f32,
    a1: -1.690_659_3_f32,
    a2: 0.732_480_77_f32,
};
const RLB_48000: BiquadCoeffs = BiquadCoeffs {
    b0: 1.0_f32,
    b1: -2.0_f32,
    b2: 1.0_f32,
    a1: -1.990_047_5_f32,
    a2: 0.990_072_25_f32,
};
// Coefficients for 88200 and 96000 Hz derived from the same analog prototype
// as the 44100/48000 constants (shelving_high at 1681.97 Hz, Butterworth HP
// at 38.135 Hz) and frozen here to avoid runtime computation on every new().
const PRE_88200: BiquadCoeffs = BiquadCoeffs {
    b0: 1.554_272_3_f32,
    b1: -2.874_154_6_f32,
    b2: 1.336_357_1_f32,
    a1: -1.810_428_9_f32,
    a2: 0.826_903_76_f32,
};
const RLB_88200: BiquadCoeffs = BiquadCoeffs {
    b0: 0.998_080_85_f32,
    b1: -1.996_161_7_f32,
    b2: 0.998_080_85_f32,
    a1: -1.996_158_1_f32,
    a2: 0.996_165_45_f32,
};
const PRE_96000: BiquadCoeffs = BiquadCoeffs {
    b0: 1.556_729_1_f32,
    b1: -2.897_725_6_f32,
    b2: 1.355_005_5_f32,
    a1: -1.825_756_3_f32,
    a2: 0.839_765_3_f32,
};
const RLB_96000: BiquadCoeffs = BiquadCoeffs {
    b0: 0.998_236_66_f32,
    b1: -1.996_473_3_f32,
    b2: 0.998_236_66_f32,
    a1: -1.996_470_2_f32,
    a2: 0.996_476_5_f32,
};

/// ITU-R BS.1770-4 loudness analyser.
///
/// Applies K-weighting and computes gated integrated loudness (LUFS-I),
/// momentary loudness (LUFS-M), short-term loudness (LUFS-S), and true-peak
/// level (dBTP).
///
/// The K-weighting filter state persists across calls. Call [`reset`] between
/// independent signals to avoid state bleed. Alternatively, `clone` the
/// analyser before a block boundary to checkpoint the filter state; restore it
/// by replacing the analyser with the clone.
///
/// # Examples
///
/// ```
/// use resonant_analysis::LufsAnalyser;
///
/// let mut analyser = LufsAnalyser::new(44100.0).unwrap();
/// // Silence falls below the absolute gate — no blocks survive.
/// let result = analyser.integrated_loudness(&vec![0.0_f32; 44100 * 5]);
/// assert!(result.is_err());
/// ```
///
/// [`reset`]: LufsAnalyser::reset
#[derive(Clone)]
pub struct LufsAnalyser {
    sample_rate: f32,
    pre: Biquad,
    rlb: Biquad,
}

impl LufsAnalyser {
    /// Creates a K-weighted analyser for the given sample rate.
    ///
    /// ITU-R BS.1770-4 Annex 1 coefficients are used for 44100 and 48000 Hz.
    /// For 88200 and 96000 Hz, coefficients are derived from the same analog
    /// prototype via bilinear transform using the filter design utilities.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError::InvalidParameter`] for unsupported sample rates.
    /// Supported rates: 44100, 48000, 88200, 96000 Hz.
    ///
    /// # Examples
    ///
    /// ```
    /// use resonant_analysis::LufsAnalyser;
    ///
    /// assert!(LufsAnalyser::new(44100.0).is_ok());
    /// assert!(LufsAnalyser::new(48000.0).is_ok());
    /// assert!(LufsAnalyser::new(22050.0).is_err());
    /// ```
    pub fn new(sample_rate: f32) -> Result<Self, AnalysisError> {
        let (pre, rlb) = kweight_coeffs(sample_rate)?;
        Ok(Self {
            sample_rate,
            pre: Biquad::new(pre),
            rlb: Biquad::new(rlb),
        })
    }

    /// Integrated loudness (gated, LUFS-I) per ITU-R BS.1770-4.
    ///
    /// Applies the absolute gate (−70 LUFS) then the relative gate (−10 LU
    /// below the absolute-gated mean). Input must be mono.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError::EmptyInput`] if the input is empty, shorter than
    /// one 400 ms block, or if all blocks are silenced by the absolute gate.
    pub fn integrated_loudness(&mut self, samples: &[f32]) -> Result<f32, AnalysisError> {
        if samples.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }

        let block_samples = block_len(self.sample_rate);
        let hop_samples = hop_len(self.sample_rate);

        let kw_samples: Vec<f32> = samples.iter().map(|&x| self.k_weight(x)).collect();

        let block_levels = block_mean_sq(&kw_samples, block_samples, hop_samples);
        if block_levels.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }

        // Absolute gate — discard blocks below −70 LUFS.
        let abs_gate_ms = lufs_to_ms(ABS_GATE_LUFS);
        let abs_gated: Vec<f32> = block_levels
            .iter()
            .copied()
            .filter(|&z| z >= abs_gate_ms)
            .collect();
        if abs_gated.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }

        // Relative gate — discard blocks more than 10 LU below the gated mean.
        let rel_gate_lufs = ms_to_lufs(mean_f32(&abs_gated)) - REL_GATE_OFFSET_LU;
        let rel_gate_ms = lufs_to_ms(rel_gate_lufs);
        let rel_gated: Vec<f32> = block_levels
            .iter()
            .copied()
            .filter(|&z| z >= rel_gate_ms)
            .collect();
        if rel_gated.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }

        Ok(ms_to_lufs(mean_f32(&rel_gated)))
    }

    /// Momentary loudness (un-gated, 400 ms window, LUFS-M).
    ///
    /// Returns −120 dB for silence or empty input.
    #[must_use]
    pub fn momentary(&mut self, frame_400ms: &[f32]) -> f32 {
        ms_to_lufs(self.kw_mean_sq(frame_400ms))
    }

    /// Short-term loudness (un-gated, 3 s window, LUFS-S).
    ///
    /// Returns −120 dB for silence or empty input.
    #[must_use]
    pub fn short_term(&mut self, frame_3s: &[f32]) -> f32 {
        ms_to_lufs(self.kw_mean_sq(frame_3s))
    }

    /// True-peak level in dBTP (4× oversampled).
    ///
    /// Upsamples the signal 4× with a polyphase resampler before finding the
    /// peak. This detects inter-sample peaks that may exceed 0 dBFS in the
    /// continuous-time signal.
    ///
    /// Returns −120 dB for silence.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError::EmptyInput`] for empty input.
    pub fn true_peak_db(&self, samples: &[f32]) -> Result<f32, AnalysisError> {
        if samples.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }
        let mut resampler =
            PolyphaseResampler::new(4, 1).ok_or(AnalysisError::InvalidParameter {
                name: "true_peak",
                reason: "failed to construct 4× polyphase resampler",
            })?;
        let upsampled = resampler.process(samples);
        let peak = upsampled.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        if peak <= 0.0 {
            return Ok(SILENCE_LUFS);
        }
        Ok(20.0 * peak.log10())
    }

    /// Resets the K-weighting filter delay state.
    ///
    /// Call between independent analyses to prevent filter memory from one
    /// signal influencing the next.
    pub fn reset(&mut self) {
        let pre_coeffs = *self.pre.coeffs();
        let rlb_coeffs = *self.rlb.coeffs();
        self.pre = Biquad::new(pre_coeffs);
        self.rlb = Biquad::new(rlb_coeffs);
    }

    /// Integrated loudness (LUFS-I) from interleaved multi-channel audio.
    ///
    /// Each channel is K-weighted independently. The gating operates on the
    /// weighted sum of per-channel mean-square levels per ITU-R BS.1770-4 §2:
    ///
    /// `z_j = Σ G_c · mean_sq(channel_c, block_j)`
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError::EmptyInput`] if the input is empty, shorter than
    /// one 400 ms block, or all blocks fall below the absolute gate (silence).
    /// Returns [`AnalysisError::InvalidParameter`] if the channel config has no
    /// weights or the sample count is not a multiple of the channel count.
    pub fn integrated_loudness_multichannel(
        &mut self,
        interleaved: &[f32],
        config: &ChannelConfig,
    ) -> Result<f32, AnalysisError> {
        let n_channels = config.weights.len();
        if n_channels == 0 {
            return Err(AnalysisError::InvalidParameter {
                name: "config",
                reason: "channel config has no weights",
            });
        }
        if interleaved.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }
        if interleaved.len() % n_channels != 0 {
            return Err(AnalysisError::InvalidParameter {
                name: "interleaved",
                reason: "sample count is not a multiple of channel count",
            });
        }

        let n_frames = interleaved.len() / n_channels;
        let block_samples = block_len(self.sample_rate);
        let hop_samples = hop_len(self.sample_rate);

        if n_frames < block_samples {
            return Err(AnalysisError::EmptyInput);
        }

        let (pre_coeffs, rlb_coeffs) = kweight_coeffs(self.sample_rate)?;
        let mut channel_filters: Vec<(Biquad, Biquad)> = (0..n_channels)
            .map(|_| (Biquad::new(pre_coeffs), Biquad::new(rlb_coeffs)))
            .collect();

        // K-weight each channel independently.
        let mut channel_kw_samples: Vec<Vec<f32>> = (0..n_channels)
            .map(|_| Vec::with_capacity(n_frames))
            .collect();
        for frame in interleaved.chunks_exact(n_channels) {
            for ((ch_kw_samples, (pre, rlb)), &sample) in channel_kw_samples
                .iter_mut()
                .zip(channel_filters.iter_mut())
                .zip(frame.iter())
            {
                ch_kw_samples.push(rlb.process_sample(pre.process_sample(sample)));
            }
        }

        // z_j = Σ_c G_c · ms_c_j  (BS.1770-4 eq. 2)
        let n_blocks = (n_frames - block_samples) / hop_samples + 1;
        let block_levels: Vec<f32> = (0..n_blocks)
            .map(|block_idx| {
                let block_start = block_idx * hop_samples;
                config
                    .weights
                    .iter()
                    .zip(channel_kw_samples.iter())
                    .map(|(&weight, ch_kw)| {
                        let block_slice = &ch_kw[block_start..block_start + block_samples];
                        let block_ms =
                            block_slice.iter().map(|&x| x * x).sum::<f32>() / block_samples as f32;
                        weight * block_ms
                    })
                    .sum()
            })
            .collect();

        if block_levels.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }

        // Absolute gate — discard blocks below −70 LUFS.
        let abs_gate_ms = lufs_to_ms(ABS_GATE_LUFS);
        let abs_gated: Vec<f32> = block_levels
            .iter()
            .copied()
            .filter(|&z| z >= abs_gate_ms)
            .collect();
        if abs_gated.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }

        // Relative gate — discard blocks more than 10 LU below the absolute-gated mean.
        let rel_gate_lufs = ms_to_lufs(mean_f32(&abs_gated)) - REL_GATE_OFFSET_LU;
        let rel_gate_ms = lufs_to_ms(rel_gate_lufs);
        let rel_gated: Vec<f32> = block_levels
            .iter()
            .copied()
            .filter(|&z| z >= rel_gate_ms)
            .collect();
        if rel_gated.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }

        Ok(ms_to_lufs(mean_f32(&rel_gated)))
    }

    #[inline]
    fn k_weight(&mut self, x: f32) -> f32 {
        self.rlb.process_sample(self.pre.process_sample(x))
    }

    fn kw_mean_sq(&mut self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = samples
            .iter()
            .map(|&x| {
                let y = self.k_weight(x);
                y * y
            })
            .sum();
        sum_sq / samples.len() as f32
    }
}

/// Selects K-weighting filter coefficients for the given sample rate.
fn kweight_coeffs(sr: f32) -> Result<(BiquadCoeffs, BiquadCoeffs), AnalysisError> {
    match sr.round() as u32 {
        44100 => Ok((PRE_44100, RLB_44100)),
        48000 => Ok((PRE_48000, RLB_48000)),
        88200 => Ok((PRE_88200, RLB_88200)),
        96000 => Ok((PRE_96000, RLB_96000)),
        _ => Err(AnalysisError::InvalidParameter {
            name: "sample_rate",
            reason: "supported rates: 44100, 48000, 88200, 96000",
        }),
    }
}

/// Length of one 400 ms gating block in samples.
fn block_len(sr: f32) -> usize {
    (sr * 0.4).round() as usize
}

/// Hop size (100 ms) giving 75% overlap between adjacent blocks.
fn hop_len(sr: f32) -> usize {
    (sr * 0.1).round() as usize
}

/// Computes mean-square over each overlapping `block`-length window.
fn block_mean_sq(signal: &[f32], block: usize, hop: usize) -> Vec<f32> {
    if block == 0 || hop == 0 || signal.len() < block {
        return vec![];
    }
    let n_blocks = (signal.len() - block) / hop + 1;
    (0..n_blocks)
        .map(|i| {
            let block_slice = &signal[i * hop..i * hop + block];
            block_slice.iter().map(|&x| x * x).sum::<f32>() / block as f32
        })
        .collect()
}

/// Converts linear mean-square to LUFS: `L = −0.691 + 10 log₁₀(mean_sq)`.
fn ms_to_lufs(mean_sq: f32) -> f32 {
    if mean_sq <= 0.0 {
        SILENCE_LUFS
    } else {
        (-0.691 + 10.0 * mean_sq.log10()).max(SILENCE_LUFS)
    }
}

/// Converts a LUFS level to the corresponding linear mean-square.
fn lufs_to_ms(lufs: f32) -> f32 {
    10f32.powf((lufs + 0.691) / 10.0)
}

fn mean_f32(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len() as f32
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use core::f32::consts::PI;

    const SR: f32 = 44100.0;

    fn sine(freq: f32, amplitude: f32, num_samples: usize, sr: f32) -> Vec<f32> {
        (0..num_samples)
            .map(|i| amplitude * (2.0 * PI * freq * i as f32 / sr).sin())
            .collect()
    }

    /// Constructs a `LufsAnalyser`; panics if the sample rate is unsupported.
    fn make(sr: f32) -> LufsAnalyser {
        match LufsAnalyser::new(sr) {
            Ok(analyser) => analyser,
            Err(e) => panic!("LufsAnalyser::new({sr}) failed: {e}"),
        }
    }

    #[test]
    fn new_44100_succeeds() {
        assert!(LufsAnalyser::new(44100.0).is_ok());
    }

    #[test]
    fn new_48000_succeeds() {
        assert!(LufsAnalyser::new(48000.0).is_ok());
    }

    #[test]
    fn new_88200_succeeds() {
        assert!(LufsAnalyser::new(88200.0).is_ok());
    }

    #[test]
    fn new_96000_succeeds() {
        assert!(LufsAnalyser::new(96000.0).is_ok());
    }

    #[test]
    fn high_rate_constants_match_design_functions() {
        // Verify frozen constants against the design functions that produced them.
        // A mismatch here means a constant was transcribed incorrectly or the
        // analog prototype parameters drifted.
        use resonant_filters::design;

        fn coeffs_close(a: BiquadCoeffs, b: BiquadCoeffs) -> bool {
            [
                (a.b0, b.b0),
                (a.b1, b.b1),
                (a.b2, b.b2),
                (a.a1, b.a1),
                (a.a2, b.a2),
            ]
            .iter()
            .all(|(x, y)| (x - y).abs() < 1e-6)
        }

        for (sr, pre_const, rlb_const) in [
            (88200.0_f32, PRE_88200, RLB_88200),
            (96000.0_f32, PRE_96000, RLB_96000),
        ] {
            let pre_live = design::shelving_high(4.0, 1681.97, sr).unwrap();
            let rlb_live = design::butterworth_highpass(38.135_f64, f64::from(sr)).unwrap();
            assert!(
                coeffs_close(pre_const, pre_live),
                "PRE constant mismatch at {sr} Hz"
            );
            assert!(
                coeffs_close(rlb_const, rlb_live),
                "RLB constant mismatch at {sr} Hz"
            );
        }
    }

    #[test]
    fn unsupported_rate_returns_error() {
        assert!(LufsAnalyser::new(22050.0).is_err());
        assert!(LufsAnalyser::new(16000.0).is_err());
        assert!(LufsAnalyser::new(0.0).is_err());
    }

    #[test]
    fn empty_input_returns_error() {
        let mut analyser = make(SR);
        assert!(matches!(
            analyser.integrated_loudness(&[]),
            Err(AnalysisError::EmptyInput)
        ));
    }

    #[test]
    fn silence_gated_out() {
        let mut analyser = make(SR);
        let result = analyser.integrated_loudness(&vec![0.0_f32; 44100 * 5]);
        assert!(matches!(result, Err(AnalysisError::EmptyInput)));
    }

    #[test]
    fn too_short_for_one_block() {
        let mut analyser = make(SR);
        let result = analyser.integrated_loudness(&[0.1_f32; 100]);
        assert!(matches!(result, Err(AnalysisError::EmptyInput)));
    }

    // A 1 kHz sine at amplitude 0.10 gives approximately −23 LUFS.
    // The K-weighting gain at 1 kHz (~+0.69 dB) and the −0.691 dB formula
    // offset together place amplitude 0.10 (−20 dBFS peak) close to −23 LUFS.
    #[test]
    fn integrated_lufs_calibration_tone() {
        let num_samples = (SR * 5.0) as usize;
        let sig = sine(1000.0, 0.10, num_samples, SR);
        let mut analyser = make(SR);
        match analyser.integrated_loudness(&sig) {
            Ok(lufs) => assert!(
                (lufs - (-23.0)).abs() < 0.2,
                "expected ≈ −23 LUFS, got {lufs:.3}"
            ),
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn integrated_lufs_6lu_per_6db() {
        let num_samples = (SR * 5.0) as usize;
        let signal_quiet = sine(1000.0, 0.10, num_samples, SR);
        let signal_loud = sine(1000.0, 0.20, num_samples, SR);
        let mut analyser = make(SR);
        let lufs_quiet = match analyser.integrated_loudness(&signal_quiet) {
            Ok(lufs) => lufs,
            Err(e) => panic!("quiet signal error: {e}"),
        };
        analyser.reset();
        let lufs_loud = match analyser.integrated_loudness(&signal_loud) {
            Ok(lufs) => lufs,
            Err(e) => panic!("loud signal error: {e}"),
        };
        assert!(
            (lufs_loud - lufs_quiet - 6.02).abs() < 0.05,
            "doubling amplitude should give +6.02 LU, got {:.3}",
            lufs_loud - lufs_quiet
        );
    }

    #[test]
    fn true_peak_full_scale_sine_near_zero_dbtp() {
        let num_samples = SR as usize;
        let sig: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 997.0 * i as f32 / SR).sin())
            .collect();
        let analyser = make(SR);
        match analyser.true_peak_db(&sig) {
            Ok(true_peak) => assert!(
                true_peak.abs() < 0.5,
                "full-scale 997 Hz sine ≈ 0 dBTP, got {true_peak:.3}"
            ),
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn true_peak_empty_returns_error() {
        let analyser = make(SR);
        assert!(matches!(
            analyser.true_peak_db(&[]),
            Err(AnalysisError::EmptyInput)
        ));
    }

    #[test]
    fn momentary_reasonable_for_sine() {
        let block_samples = block_len(SR);
        let sig = sine(1000.0, 0.10, block_samples, SR);
        let momentary_lufs = {
            let mut analyser = make(SR);
            analyser.momentary(&sig)
        };
        assert!(
            momentary_lufs > -35.0 && momentary_lufs < -15.0,
            "momentary LUFS out of range: {momentary_lufs:.2}"
        );
    }

    #[test]
    fn short_term_reasonable_for_sine() {
        let num_samples = (SR * 3.0) as usize;
        let sig = sine(1000.0, 0.10, num_samples, SR);
        let short_term_lufs = {
            let mut analyser = make(SR);
            analyser.short_term(&sig)
        };
        assert!(
            short_term_lufs > -35.0 && short_term_lufs < -15.0,
            "short-term LUFS out of range: {short_term_lufs:.2}"
        );
    }

    #[test]
    fn clone_checkpoints_filter_state() {
        // Feed some signal to give the filters non-trivial state, then clone.
        // Both the original and the clone must produce identical results on the
        // same subsequent input, confirming the state was captured correctly and
        // the two instances are independent.
        let block_samples = block_len(SR);
        let warmup = sine(200.0, 0.3, block_samples, SR);
        let test_sig = sine(1000.0, 0.10, block_samples, SR);

        let mut analyser = make(SR);
        let _ = analyser.momentary(&warmup);

        let mut checkpoint = analyser.clone();

        let lufs_original = analyser.momentary(&test_sig);
        let lufs_clone = checkpoint.momentary(&test_sig);

        assert!(
            (lufs_original - lufs_clone).abs() < 1e-6,
            "clone must reproduce original result: {lufs_original:.6} vs {lufs_clone:.6}"
        );
    }

    #[test]
    fn reset_clears_filter_state() {
        let block_samples = block_len(SR);
        let pre_warm = sine(100.0, 1.0, block_samples, SR);
        let test_sig = sine(1000.0, 0.10, block_samples, SR);

        let lufs_after_reset = {
            let mut analyser = make(SR);
            let _ = analyser.momentary(&pre_warm);
            analyser.reset();
            analyser.momentary(&test_sig)
        };
        let lufs_fresh = {
            let mut analyser = make(SR);
            analyser.momentary(&test_sig)
        };
        assert!(
            (lufs_after_reset - lufs_fresh).abs() < 1e-5,
            "reset should give same result as fresh: {lufs_after_reset:.6} vs {lufs_fresh:.6}"
        );
    }

    #[test]
    fn block_mean_sq_constant_signal() {
        let sig = vec![0.5_f32; 100];
        let block_levels = block_mean_sq(&sig, 50, 25);
        assert_eq!(block_levels.len(), 3);
        for level in &block_levels {
            assert!((level - 0.25).abs() < 1e-6, "expected 0.25, got {level}");
        }
    }

    #[test]
    fn block_mean_sq_too_short() {
        assert!(block_mean_sq(&[1.0_f32; 10], 50, 25).is_empty());
    }

    #[test]
    fn ms_to_lufs_known_value() {
        // L = -0.691 + 10*log10(0.5) = -0.691 - 3.0103 ≈ -3.701
        let lufs = ms_to_lufs(0.5);
        assert!((lufs - (-3.701)).abs() < 0.001, "got {lufs:.4}");
    }

    #[test]
    fn ms_to_lufs_silence() {
        assert_eq!(ms_to_lufs(0.0), SILENCE_LUFS);
        assert_eq!(ms_to_lufs(-1.0), SILENCE_LUFS);
    }

    #[test]
    fn lufs_to_ms_roundtrip() {
        let lufs = -23.0_f32;
        let mean_sq = lufs_to_ms(lufs);
        let roundtripped_lufs = ms_to_lufs(mean_sq);
        assert!(
            (roundtripped_lufs - lufs).abs() < 0.001,
            "roundtrip: {lufs} → {roundtripped_lufs:.4}"
        );
    }

    // --- integrated_loudness_multichannel ---

    #[test]
    fn multichannel_empty_input_returns_error() {
        let mut analyser = make(SR);
        assert!(matches!(
            analyser.integrated_loudness_multichannel(&[], &ChannelConfig::stereo()),
            Err(AnalysisError::EmptyInput)
        ));
    }

    // --- ChannelConfig constructor / accessor tests ---

    #[test]
    fn channel_config_custom_empty_returns_none() {
        assert!(ChannelConfig::custom(vec![]).is_none());
    }

    #[test]
    fn channel_config_custom_nan_returns_none() {
        assert!(ChannelConfig::custom(vec![1.0, f32::NAN]).is_none());
    }

    #[test]
    fn channel_config_custom_infinite_returns_none() {
        assert!(ChannelConfig::custom(vec![1.0, f32::INFINITY]).is_none());
        assert!(ChannelConfig::custom(vec![f32::NEG_INFINITY]).is_none());
    }

    #[test]
    fn channel_config_custom_valid_zero_weight_allowed() {
        // Zero is a valid weight (LFE exclusion); must not be rejected.
        let cfg = ChannelConfig::custom(vec![1.0, 1.0, 0.0]).unwrap();
        assert_eq!(cfg.weights(), &[1.0, 1.0, 0.0]);
    }

    #[test]
    fn channel_config_weights_accessor_matches_factory_methods() {
        assert_eq!(ChannelConfig::mono().weights(), &[1.0_f32]);
        assert_eq!(ChannelConfig::stereo().weights(), &[1.0_f32, 1.0]);
        assert_eq!(ChannelConfig::surround_5_1().weights().len(), 6);
    }

    #[test]
    fn multichannel_mismatched_samples_returns_error() {
        let mut analyser = make(SR);
        // 3 samples is not a multiple of 2 channels.
        assert!(matches!(
            analyser.integrated_loudness_multichannel(&[0.1_f32; 3], &ChannelConfig::stereo()),
            Err(AnalysisError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn multichannel_mono_matches_integrated_loudness() {
        // ChannelConfig::mono() wrapping a mono signal must give the same LUFS
        // as integrated_loudness() on the same samples.
        let num_samples = (SR * 5.0) as usize;
        let sig = sine(1000.0, 0.10, num_samples, SR);
        let mut analyser_mono = make(SR);
        let mut analyser_mc = make(SR);
        let lufs_mono = analyser_mono
            .integrated_loudness(&sig)
            .unwrap_or_else(|e| panic!("integrated_loudness error: {e}"));
        let lufs_mc = analyser_mc
            .integrated_loudness_multichannel(&sig, &ChannelConfig::mono())
            .unwrap_or_else(|e| panic!("multichannel mono error: {e}"));
        assert!(
            (lufs_mono - lufs_mc).abs() < 0.01,
            "mono config should match integrated_loudness: {lufs_mono:.3} vs {lufs_mc:.3}"
        );
    }

    #[test]
    fn stereo_identical_channels_is_three_lu_above_mono() {
        // BS.1770-4 with G_L=G_R=1.0: z_stereo = 2 · z_mono →
        // LUFS_stereo = LUFS_mono + 10·log10(2) ≈ LUFS_mono + 3.01 LU.
        let num_samples = (SR * 5.0) as usize;
        let mono = sine(1000.0, 0.10, num_samples, SR);
        let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s, s]).collect();

        let mut analyser_mono = make(SR);
        let mut analyser_stereo = make(SR);

        let lufs_mono = analyser_mono
            .integrated_loudness(&mono)
            .unwrap_or_else(|e| panic!("mono error: {e}"));
        let lufs_stereo = analyser_stereo
            .integrated_loudness_multichannel(&stereo, &ChannelConfig::stereo())
            .unwrap_or_else(|e| panic!("stereo error: {e}"));

        let diff = lufs_stereo - lufs_mono;
        assert!(
            (diff - 3.01).abs() < 0.1,
            "identical stereo should be +3.01 LU above mono, got {diff:.3} LU"
        );
    }

    #[test]
    fn surround_5_1_lfe_only_is_gated_out() {
        // LFE channel (index 3) has weight 0.0 — even a loud LFE signal must
        // result in all blocks being silenced by the absolute gate.
        let num_samples = (SR * 5.0) as usize;
        let lfe_signal = sine(60.0, 0.5, num_samples, SR);
        // Interleave: L=0, R=0, C=0, LFE=signal, Ls=0, Rs=0
        let interleaved: Vec<f32> = (0..num_samples)
            .flat_map(|i| [0.0, 0.0, 0.0, lfe_signal[i], 0.0, 0.0])
            .collect();

        let mut analyser = make(SR);
        assert!(
            matches!(
                analyser
                    .integrated_loudness_multichannel(&interleaved, &ChannelConfig::surround_5_1()),
                Err(AnalysisError::EmptyInput)
            ),
            "LFE-only 5.1 should be gated out"
        );
    }

    #[test]
    fn stereo_silence_gated_out() {
        let num_samples = (SR * 5.0) as usize;
        let stereo = vec![0.0_f32; num_samples * 2];
        let mut analyser = make(SR);
        assert!(matches!(
            analyser.integrated_loudness_multichannel(&stereo, &ChannelConfig::stereo()),
            Err(AnalysisError::EmptyInput)
        ));
    }

    #[test]
    fn stereo_too_short_for_one_block_returns_error() {
        let mut analyser = make(SR);
        let stereo = vec![0.1_f32; 200]; // 100 stereo frames, well under 400 ms
        assert!(matches!(
            analyser.integrated_loudness_multichannel(&stereo, &ChannelConfig::stereo()),
            Err(AnalysisError::EmptyInput)
        ));
    }
}
