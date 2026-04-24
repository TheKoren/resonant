//! ITU-R BS.1770-4 integrated loudness (LUFS / LKFS) measurement.
//!
//! Applies K-weighting (a two-stage biquad cascade), computes mean-square
//! levels over 400 ms gating blocks, then applies the absolute and relative
//! gates defined in the standard to produce integrated loudness.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use resonant_filters::biquad::{Biquad, BiquadCoeffs};
use resonant_filters::{design, PolyphaseResampler};

use crate::AnalysisError;

/// Channel configuration for multi-channel loudness measurement.
///
/// Per ITU-R BS.1770-4, each channel is K-weighted independently; the gating
/// operates on the weighted sum of per-channel mean-square levels.
///
/// Standard weights: L=1.0, R=1.0, C=1.0, LFE=0.0, Ls=√2, Rs=√2.
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    /// Per-channel gain weights (linear, not dB).
    pub weights: Vec<f32>,
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

/// ITU-R BS.1770-4 loudness analyser.
///
/// Applies K-weighting and computes gated integrated loudness (LUFS-I),
/// momentary loudness (LUFS-M), short-term loudness (LUFS-S), and true-peak
/// level (dBTP).
///
/// The K-weighting filter state persists across calls. Call [`reset`] between
/// independent signals to avoid state bleed.
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

        let block = block_len(self.sample_rate);
        let hop = hop_len(self.sample_rate);

        let kw: Vec<f32> = samples.iter().map(|&x| self.k_weight(x)).collect();

        let zs = block_mean_sq(&kw, block, hop);
        if zs.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }

        // Absolute gate — discard blocks below −70 LUFS.
        let abs_z = lufs_to_ms(ABS_GATE_LUFS);
        let pass_abs: Vec<f32> = zs.iter().copied().filter(|&z| z >= abs_z).collect();
        if pass_abs.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }

        // Relative gate — discard blocks more than 10 LU below the gated mean.
        let j_gate = ms_to_lufs(mean_f32(&pass_abs)) - REL_GATE_OFFSET_LU;
        let rel_z = lufs_to_ms(j_gate);
        let pass_rel: Vec<f32> = zs.iter().copied().filter(|&z| z >= rel_z).collect();
        if pass_rel.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }

        Ok(ms_to_lufs(mean_f32(&pass_rel)))
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
        let pre_c = *self.pre.coeffs();
        let rlb_c = *self.rlb.coeffs();
        self.pre = Biquad::new(pre_c);
        self.rlb = Biquad::new(rlb_c);
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
        let block = block_len(self.sample_rate);
        let hop = hop_len(self.sample_rate);

        if n_frames < block {
            return Err(AnalysisError::EmptyInput);
        }

        let (pre_coeffs, rlb_coeffs) = kweight_coeffs(self.sample_rate)?;
        let mut channel_filters: Vec<(Biquad, Biquad)> = (0..n_channels)
            .map(|_| (Biquad::new(pre_coeffs), Biquad::new(rlb_coeffs)))
            .collect();

        // K-weight each channel independently.
        let mut channel_kw: Vec<Vec<f32>> = (0..n_channels)
            .map(|_| Vec::with_capacity(n_frames))
            .collect();
        for frame in interleaved.chunks_exact(n_channels) {
            for ((ch_kw, (pre, rlb)), &sample) in channel_kw
                .iter_mut()
                .zip(channel_filters.iter_mut())
                .zip(frame.iter())
            {
                ch_kw.push(rlb.process_sample(pre.process_sample(sample)));
            }
        }

        // z_j = Σ_c G_c · ms_c_j  (BS.1770-4 eq. 2)
        let n_blocks = (n_frames - block) / hop + 1;
        let zs: Vec<f32> = (0..n_blocks)
            .map(|j| {
                let start = j * hop;
                config
                    .weights
                    .iter()
                    .zip(channel_kw.iter())
                    .map(|(&weight, ch_kw)| {
                        let sl = &ch_kw[start..start + block];
                        let ms = sl.iter().map(|&x| x * x).sum::<f32>() / block as f32;
                        weight * ms
                    })
                    .sum()
            })
            .collect();

        if zs.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }

        // Absolute gate — discard blocks below −70 LUFS.
        let abs_z = lufs_to_ms(ABS_GATE_LUFS);
        let pass_abs: Vec<f32> = zs.iter().copied().filter(|&z| z >= abs_z).collect();
        if pass_abs.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }

        // Relative gate — discard blocks more than 10 LU below the absolute-gated mean.
        let j_gate = ms_to_lufs(mean_f32(&pass_abs)) - REL_GATE_OFFSET_LU;
        let rel_z = lufs_to_ms(j_gate);
        let pass_rel: Vec<f32> = zs.iter().copied().filter(|&z| z >= rel_z).collect();
        if pass_rel.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }

        Ok(ms_to_lufs(mean_f32(&pass_rel)))
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
        88200 | 96000 => {
            // Analog prototype: +4 dB high-shelf at 1681.97 Hz (pre-filter);
            // 2nd-order Butterworth HP at 38.135 Hz (RLB high-pass).
            let pre =
                design::shelving_high(4.0, 1681.97, sr).ok_or(AnalysisError::InvalidParameter {
                    name: "sample_rate",
                    reason: "K-weighting pre-filter design failed",
                })?;
            let rlb = design::butterworth_highpass(38.135_f64, f64::from(sr)).ok_or(
                AnalysisError::InvalidParameter {
                    name: "sample_rate",
                    reason: "K-weighting RLB high-pass design failed",
                },
            )?;
            Ok((pre, rlb))
        }
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
    let n = (signal.len() - block) / hop + 1;
    (0..n)
        .map(|i| {
            let sl = &signal[i * hop..i * hop + block];
            sl.iter().map(|&x| x * x).sum::<f32>() / block as f32
        })
        .collect()
}

/// Converts linear mean-square to LUFS: `L = −0.691 + 10 log₁₀(z)`.
fn ms_to_lufs(z: f32) -> f32 {
    if z <= 0.0 {
        SILENCE_LUFS
    } else {
        (-0.691 + 10.0 * z.log10()).max(SILENCE_LUFS)
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
    use super::*;
    use core::f32::consts::PI;

    const SR: f32 = 44100.0;

    fn sine(freq: f32, amplitude: f32, n: usize, sr: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amplitude * (2.0 * PI * freq * i as f32 / sr).sin())
            .collect()
    }

    /// Constructs a `LufsAnalyser`; panics if the sample rate is unsupported.
    fn make(sr: f32) -> LufsAnalyser {
        match LufsAnalyser::new(sr) {
            Ok(a) => a,
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
    fn unsupported_rate_returns_error() {
        assert!(LufsAnalyser::new(22050.0).is_err());
        assert!(LufsAnalyser::new(16000.0).is_err());
        assert!(LufsAnalyser::new(0.0).is_err());
    }

    #[test]
    fn empty_input_returns_error() {
        let mut a = make(SR);
        assert!(matches!(
            a.integrated_loudness(&[]),
            Err(AnalysisError::EmptyInput)
        ));
    }

    #[test]
    fn silence_gated_out() {
        let mut a = make(SR);
        let r = a.integrated_loudness(&vec![0.0_f32; 44100 * 5]);
        assert!(matches!(r, Err(AnalysisError::EmptyInput)));
    }

    #[test]
    fn too_short_for_one_block() {
        let mut a = make(SR);
        let r = a.integrated_loudness(&[0.1_f32; 100]);
        assert!(matches!(r, Err(AnalysisError::EmptyInput)));
    }

    // A 1 kHz sine at amplitude 0.10 gives approximately −23 LUFS.
    // The K-weighting gain at 1 kHz (~+0.69 dB) and the −0.691 dB formula
    // offset together place amplitude 0.10 (−20 dBFS peak) close to −23 LUFS.
    #[test]
    fn integrated_lufs_calibration_tone() {
        let n = (SR * 5.0) as usize;
        let sig = sine(1000.0, 0.10, n, SR);
        let mut a = make(SR);
        match a.integrated_loudness(&sig) {
            Ok(l) => assert!((l - (-23.0)).abs() < 0.2, "expected ≈ −23 LUFS, got {l:.3}"),
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn integrated_lufs_6lu_per_6db() {
        let n = (SR * 5.0) as usize;
        let s1 = sine(1000.0, 0.10, n, SR);
        let s2 = sine(1000.0, 0.20, n, SR);
        let mut a = make(SR);
        let l1 = match a.integrated_loudness(&s1) {
            Ok(l) => l,
            Err(e) => panic!("l1 error: {e}"),
        };
        a.reset();
        let l2 = match a.integrated_loudness(&s2) {
            Ok(l) => l,
            Err(e) => panic!("l2 error: {e}"),
        };
        assert!(
            (l2 - l1 - 6.02).abs() < 0.05,
            "doubling amplitude should give +6.02 LU, got {:.3}",
            l2 - l1
        );
    }

    #[test]
    fn true_peak_full_scale_sine_near_zero_dbtp() {
        let n = SR as usize;
        let sig: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 997.0 * i as f32 / SR).sin())
            .collect();
        let a = make(SR);
        match a.true_peak_db(&sig) {
            Ok(tp) => assert!(
                tp.abs() < 0.5,
                "full-scale 997 Hz sine ≈ 0 dBTP, got {tp:.3}"
            ),
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn true_peak_empty_returns_error() {
        let a = make(SR);
        assert!(matches!(
            a.true_peak_db(&[]),
            Err(AnalysisError::EmptyInput)
        ));
    }

    #[test]
    fn momentary_reasonable_for_sine() {
        let block = block_len(SR);
        let sig = sine(1000.0, 0.10, block, SR);
        let m = {
            let mut a = make(SR);
            a.momentary(&sig)
        };
        assert!(
            m > -35.0 && m < -15.0,
            "momentary LUFS out of range: {m:.2}"
        );
    }

    #[test]
    fn short_term_reasonable_for_sine() {
        let n = (SR * 3.0) as usize;
        let sig = sine(1000.0, 0.10, n, SR);
        let st = {
            let mut a = make(SR);
            a.short_term(&sig)
        };
        assert!(
            st > -35.0 && st < -15.0,
            "short-term LUFS out of range: {st:.2}"
        );
    }

    #[test]
    fn reset_clears_filter_state() {
        let block = block_len(SR);
        let pre_warm = sine(100.0, 1.0, block, SR);
        let test_sig = sine(1000.0, 0.10, block, SR);

        let warmed = {
            let mut a = make(SR);
            let _ = a.momentary(&pre_warm);
            a.reset();
            a.momentary(&test_sig)
        };
        let fresh = {
            let mut a = make(SR);
            a.momentary(&test_sig)
        };
        assert!(
            (warmed - fresh).abs() < 1e-5,
            "reset should give same result as fresh: {warmed:.6} vs {fresh:.6}"
        );
    }

    #[test]
    fn block_mean_sq_constant_signal() {
        let sig = vec![0.5_f32; 100];
        let zs = block_mean_sq(&sig, 50, 25);
        assert_eq!(zs.len(), 3);
        for z in &zs {
            assert!((z - 0.25).abs() < 1e-6, "expected 0.25, got {z}");
        }
    }

    #[test]
    fn block_mean_sq_too_short() {
        assert!(block_mean_sq(&[1.0_f32; 10], 50, 25).is_empty());
    }

    #[test]
    fn ms_to_lufs_known_value() {
        // L = -0.691 + 10*log10(0.5) = -0.691 - 3.0103 ≈ -3.701
        let l = ms_to_lufs(0.5);
        assert!((l - (-3.701)).abs() < 0.001, "got {l:.4}");
    }

    #[test]
    fn ms_to_lufs_silence() {
        assert_eq!(ms_to_lufs(0.0), SILENCE_LUFS);
        assert_eq!(ms_to_lufs(-1.0), SILENCE_LUFS);
    }

    #[test]
    fn lufs_to_ms_roundtrip() {
        let lufs = -23.0_f32;
        let ms = lufs_to_ms(lufs);
        let back = ms_to_lufs(ms);
        assert!((back - lufs).abs() < 0.001, "roundtrip: {lufs} → {back:.4}");
    }

    // --- integrated_loudness_multichannel ---

    #[test]
    fn multichannel_empty_input_returns_error() {
        let mut a = make(SR);
        assert!(matches!(
            a.integrated_loudness_multichannel(&[], &ChannelConfig::stereo()),
            Err(AnalysisError::EmptyInput)
        ));
    }

    #[test]
    fn multichannel_empty_config_returns_error() {
        let mut a = make(SR);
        let config = ChannelConfig { weights: vec![] };
        assert!(matches!(
            a.integrated_loudness_multichannel(&[0.1_f32; 100], &config),
            Err(AnalysisError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn multichannel_mismatched_samples_returns_error() {
        let mut a = make(SR);
        // 3 samples is not a multiple of 2 channels.
        assert!(matches!(
            a.integrated_loudness_multichannel(&[0.1_f32; 3], &ChannelConfig::stereo()),
            Err(AnalysisError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn multichannel_mono_matches_integrated_loudness() {
        // ChannelConfig::mono() wrapping a mono signal must give the same LUFS
        // as integrated_loudness() on the same samples.
        let n = (SR * 5.0) as usize;
        let sig = sine(1000.0, 0.10, n, SR);
        let mut a1 = make(SR);
        let mut a2 = make(SR);
        let lufs_mono = a1
            .integrated_loudness(&sig)
            .unwrap_or_else(|e| panic!("integrated_loudness error: {e}"));
        let lufs_mc = a2
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
        let n = (SR * 5.0) as usize;
        let mono = sine(1000.0, 0.10, n, SR);
        let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s, s]).collect();

        let mut a_mono = make(SR);
        let mut a_stereo = make(SR);

        let lufs_mono = a_mono
            .integrated_loudness(&mono)
            .unwrap_or_else(|e| panic!("mono error: {e}"));
        let lufs_stereo = a_stereo
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
        let n = (SR * 5.0) as usize;
        let lfe_signal = sine(60.0, 0.5, n, SR);
        // Interleave: L=0, R=0, C=0, LFE=signal, Ls=0, Rs=0
        let interleaved: Vec<f32> = (0..n)
            .flat_map(|i| [0.0, 0.0, 0.0, lfe_signal[i], 0.0, 0.0])
            .collect();

        let mut a = make(SR);
        assert!(
            matches!(
                a.integrated_loudness_multichannel(&interleaved, &ChannelConfig::surround_5_1()),
                Err(AnalysisError::EmptyInput)
            ),
            "LFE-only 5.1 should be gated out"
        );
    }

    #[test]
    fn stereo_silence_gated_out() {
        let n = (SR * 5.0) as usize;
        let stereo = vec![0.0_f32; n * 2];
        let mut a = make(SR);
        assert!(matches!(
            a.integrated_loudness_multichannel(&stereo, &ChannelConfig::stereo()),
            Err(AnalysisError::EmptyInput)
        ));
    }

    #[test]
    fn stereo_too_short_for_one_block_returns_error() {
        let mut a = make(SR);
        let stereo = vec![0.1_f32; 200]; // 100 stereo frames, well under 400 ms
        assert!(matches!(
            a.integrated_loudness_multichannel(&stereo, &ChannelConfig::stereo()),
            Err(AnalysisError::EmptyInput)
        ));
    }
}
