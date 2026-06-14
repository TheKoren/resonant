//! Phase vocoder for time-stretching and pitch-shifting.
//!
//! Implements a standard STFT-based phase vocoder with identity phase locking
//! (Laroche & Dolson, 1999) to suppress the "phasiness" artefact common in
//! naive phase-vocoder implementations.
//!
//! # How it works
//!
//! The phase vocoder modifies playback speed without changing pitch by
//! manipulating STFT bin phases frame-by-frame:
//!
//! 1. For each bin, the instantaneous frequency is estimated from the phase
//!    difference between consecutive analysis frames (principal-argument
//!    wrapping removes 2π ambiguity).
//! 2. Output phases are accumulated at the synthesis hop rate (scaled by
//!    `stretch_factor`).
//! 3. Identity phase locking propagates accumulated peak phases to all
//!    neighbouring bins, preserving harmonic phase relationships.
//!
//! # Usage
//!
//! ```
//! use resonant_fft::phase_vocoder::PhaseVocoder;
//!
//! let fft_size = 1024;
//! let hop = 256;
//! let mut pv = PhaseVocoder::new(fft_size, hop, 1.5).unwrap(); // 1.5× slower
//!
//! let num_bins = fft_size / 2 + 1;
//! let mags = vec![0.0_f32; num_bins];
//! let phases = vec![0.0_f32; num_bins];
//! let mut out_mags = vec![0.0_f32; num_bins];
//! let mut out_phases = vec![0.0_f32; num_bins];
//!
//! pv.process_frame(&mags, &phases, &mut out_mags, &mut out_phases);
//! ```

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use core::f32::consts::PI;

// Brings .floor() into scope for f32 in no_std + alloc builds via libm.
#[allow(unused_imports)]
use num_traits::float::Float as _;

use crate::FftError;

const TWO_PI: f32 = 2.0 * PI;

/// Phase vocoder for time-stretching via STFT-based phase manipulation.
///
/// Construct with [`PhaseVocoder::new`], then call [`process_frame`] once per
/// STFT analysis frame. Combine the returned magnitudes and phases back into
/// complex bins and perform an IFFT to obtain the synthesis frame.
///
/// [`process_frame`]: PhaseVocoder::process_frame
#[derive(Debug, Clone)]
pub struct PhaseVocoder {
    fft_size: usize,
    hop_analysis: usize,
    hop_synthesis: usize,
    num_bins: usize,
    prev_phases: Vec<f32>,
    phase_accum: Vec<f32>,
}

impl PhaseVocoder {
    /// Creates a new `PhaseVocoder`.
    ///
    /// # Parameters
    ///
    /// - `fft_size` — FFT frame size. Must be a non-zero power of two.
    /// - `hop_size` — Analysis hop in samples. Must be > 0.
    /// - `stretch_factor` — Time-stretch ratio. Values > 1 slow playback; < 1
    ///   speed it up; 1.0 is pass-through. Must be positive.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::NotPowerOfTwo`] if `fft_size` is zero or not a
    /// power of two.
    ///
    /// # Panics
    ///
    /// Panics if `hop_size` is zero or `stretch_factor` is not positive.
    pub fn new(fft_size: usize, hop_size: usize, stretch_factor: f32) -> Result<Self, FftError> {
        if !fft_size.is_power_of_two() {
            return Err(FftError::NotPowerOfTwo(fft_size));
        }
        assert!(hop_size > 0, "hop_size must be > 0");
        assert!(stretch_factor > 0.0, "stretch_factor must be positive");

        let num_bins = fft_size / 2 + 1;
        let hop_synthesis = ((hop_size as f32) * stretch_factor).round() as usize;
        let hop_synthesis = hop_synthesis.max(1);

        Ok(Self {
            fft_size,
            hop_analysis: hop_size,
            hop_synthesis,
            num_bins,
            prev_phases: vec![0.0; num_bins],
            phase_accum: vec![0.0; num_bins],
        })
    }

    /// Returns the FFT frame size.
    #[must_use]
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Returns the analysis hop size in samples.
    #[must_use]
    pub fn hop_analysis(&self) -> usize {
        self.hop_analysis
    }

    /// Returns the synthesis hop size in samples.
    ///
    /// Equals `round(hop_analysis * stretch_factor)`, minimum 1.
    #[must_use]
    pub fn hop_synthesis(&self) -> usize {
        self.hop_synthesis
    }

    /// Processes one STFT analysis frame through the phase vocoder.
    ///
    /// `magnitudes` and `phases` are the positive-frequency half of the
    /// current frame: `fft_size / 2 + 1` bins. `out_magnitudes` and
    /// `out_phases` receive the corresponding synthesis values.
    ///
    /// After this call, reconstruct the complex spectrum from
    /// `out_magnitudes[k] * exp(i * out_phases[k])` and run an IFFT to
    /// obtain the synthesis frame.
    ///
    /// # Panics
    ///
    /// Panics if any slice length is not `fft_size / 2 + 1`.
    #[inline]
    pub fn process_frame(
        &mut self,
        magnitudes: &[f32],
        phases: &[f32],
        out_magnitudes: &mut [f32],
        out_phases: &mut [f32],
    ) {
        let num_bins = self.num_bins;
        assert_eq!(
            magnitudes.len(),
            num_bins,
            "magnitudes length must equal fft_size/2+1"
        );
        assert_eq!(
            phases.len(),
            num_bins,
            "phases length must equal fft_size/2+1"
        );
        assert_eq!(
            out_magnitudes.len(),
            num_bins,
            "out_magnitudes length must equal fft_size/2+1"
        );
        assert_eq!(
            out_phases.len(),
            num_bins,
            "out_phases length must equal fft_size/2+1"
        );

        let analysis_hop_f = self.hop_analysis as f32;
        let stretch_ratio = self.hop_synthesis as f32 / analysis_hop_f;

        for (k, ((&phase, &prev_phase), accum)) in phases
            .iter()
            .zip(self.prev_phases.iter())
            .zip(self.phase_accum.iter_mut())
            .enumerate()
        {
            // Phase advance expected for a pure tone at bin centre frequency.
            let expected_advance = TWO_PI * (k as f32) * analysis_hop_f / (self.fft_size as f32);
            // Deviation from expected, wrapped to [-π, π].
            let delta = phase - prev_phase - expected_advance;
            let true_advance = expected_advance + principal_argument(delta);
            *accum += true_advance * stretch_ratio;
        }
        self.prev_phases.copy_from_slice(phases);
        out_magnitudes.copy_from_slice(magnitudes);

        // Identity phase locking: propagate peak accumulated phases to their
        // neighbouring bins to suppress phasiness.
        apply_identity_phase_lock(magnitudes, phases, &self.phase_accum, out_phases, num_bins);
    }

    /// Resets phase accumulator and frame-to-frame phase memory to zero.
    ///
    /// Call between independent audio segments to avoid phase discontinuities.
    pub fn reset(&mut self) {
        for v in &mut self.prev_phases {
            *v = 0.0;
        }
        for v in &mut self.phase_accum {
            *v = 0.0;
        }
    }
}

/// Wraps `phase` to `[-π, π]` (the principal argument).
fn principal_argument(phase: f32) -> f32 {
    phase - TWO_PI * ((phase + PI) / TWO_PI).floor()
}

/// Applies identity phase locking (Laroche & Dolson, 1999).
///
/// Local magnitude peaks keep their accumulated phases. Each non-peak bin is
/// set relative to the nearest peak so that harmonic phase relationships
/// are preserved across the synthesis hop.
fn apply_identity_phase_lock(
    magnitudes: &[f32],
    in_phases: &[f32],
    phase_accum: &[f32],
    out_phases: &mut [f32],
    num_bins: usize,
) {
    if num_bins == 0 {
        return;
    }

    // Identify local magnitude maxima.
    let mut is_peak = vec![false; num_bins];
    if num_bins == 1 {
        is_peak[0] = true;
    } else {
        is_peak[0] = magnitudes[0] >= magnitudes[1];
        is_peak[num_bins - 1] = magnitudes[num_bins - 1] >= magnitudes[num_bins - 2];
        for k in 1..num_bins - 1 {
            is_peak[k] = magnitudes[k] > magnitudes[k - 1] && magnitudes[k] > magnitudes[k + 1];
        }
    }

    // If the spectrum is flat (no peaks), fall back to raw accumulated phases.
    if !is_peak.iter().any(|&p| p) {
        out_phases.copy_from_slice(phase_accum);
        return;
    }

    // Two-pass scan: record the nearest peak to the left and to the right
    // of each bin, then assign each bin to whichever is closer.
    let mut peak_left: Vec<Option<usize>> = vec![None; num_bins];
    let mut peak_right: Vec<Option<usize>> = vec![None; num_bins];

    let mut last_peak = None;
    for k in 0..num_bins {
        if is_peak[k] {
            last_peak = Some(k);
        }
        peak_left[k] = last_peak;
    }

    let mut last_peak = None;
    for k in (0..num_bins).rev() {
        if is_peak[k] {
            last_peak = Some(k);
        }
        peak_right[k] = last_peak;
    }

    for k in 0..num_bins {
        let nearest_peak = match (peak_left[k], peak_right[k]) {
            (Some(left_peak), Some(right_peak)) => {
                // k >= left_peak and right_peak >= k by construction, so no underflow.
                if k - left_peak <= right_peak - k {
                    left_peak
                } else {
                    right_peak
                }
            }
            (Some(left_peak), None) => left_peak,
            (None, Some(right_peak)) => right_peak,
            (None, None) => k,
        };
        // Phase relative to the peak in the input frame, shifted by the
        // peak's accumulated output phase.
        out_phases[k] = phase_accum[nearest_peak] + (in_phases[k] - in_phases[nearest_peak]);
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;

    #[test]
    fn new_power_of_two_succeeds() {
        assert!(PhaseVocoder::new(1024, 256, 1.0).is_ok());
        assert!(PhaseVocoder::new(64, 16, 2.0).is_ok());
        assert!(PhaseVocoder::new(4, 1, 0.5).is_ok());
    }

    #[test]
    fn new_non_power_of_two_returns_error() {
        assert!(matches!(
            PhaseVocoder::new(1000, 256, 1.0),
            Err(FftError::NotPowerOfTwo(1000))
        ));
        assert!(matches!(
            PhaseVocoder::new(3, 1, 1.0),
            Err(FftError::NotPowerOfTwo(3))
        ));
    }

    #[test]
    fn new_zero_fft_size_returns_error() {
        assert!(matches!(
            PhaseVocoder::new(0, 256, 1.0),
            Err(FftError::NotPowerOfTwo(0))
        ));
    }

    #[test]
    fn accessors_return_correct_values() {
        let pv = PhaseVocoder::new(1024, 256, 2.0).unwrap();
        assert_eq!(pv.fft_size(), 1024);
        assert_eq!(pv.hop_analysis(), 256);
        assert_eq!(pv.hop_synthesis(), 512);
    }

    #[test]
    fn stretch_half_gives_half_synthesis_hop() {
        let pv = PhaseVocoder::new(512, 128, 0.5).unwrap();
        assert_eq!(pv.hop_synthesis(), 64);
    }

    #[test]
    fn process_frame_passes_magnitudes_through() {
        let fft_size = 16;
        let mut pv = PhaseVocoder::new(fft_size, 4, 1.0).unwrap();
        let num_bins = fft_size / 2 + 1;
        let mags: Vec<f32> = (0..num_bins).map(|k| k as f32).collect();
        let phases = std::vec![0.0_f32; num_bins];
        let mut out_mags = std::vec![0.0_f32; num_bins];
        let mut out_phases = std::vec![0.0_f32; num_bins];

        pv.process_frame(&mags, &phases, &mut out_mags, &mut out_phases);

        assert_eq!(out_mags, mags);
    }

    #[test]
    fn dc_bin_phase_stays_zero_with_constant_zero_input() {
        // DC bin (k=0) has expected_advance = 0, so phase_accum[0] stays 0
        // regardless of the number of frames processed.
        let fft_size = 32;
        let mut pv = PhaseVocoder::new(fft_size, 8, 1.0).unwrap();
        let num_bins = fft_size / 2 + 1;
        let mags = std::vec![1.0_f32; num_bins];
        let phases = std::vec![0.0_f32; num_bins];
        let mut out_mags = std::vec![0.0_f32; num_bins];
        let mut out_phases = std::vec![0.0_f32; num_bins];

        for _ in 0..8 {
            pv.process_frame(&mags, &phases, &mut out_mags, &mut out_phases);
        }

        assert!(
            out_phases[0].abs() < 1e-6,
            "DC bin phase should be ~0, got {}",
            out_phases[0]
        );
    }

    #[test]
    fn reset_restores_initial_state() {
        let fft_size = 16;
        let num_bins = fft_size / 2 + 1;
        let mags = std::vec![1.0_f32; num_bins];
        let phases: Vec<f32> = (0..num_bins).map(|k| k as f32 * 0.3).collect();
        let mut out_mags = std::vec![0.0_f32; num_bins];
        let mut out_phases_after_reset = std::vec![0.0_f32; num_bins];
        let mut out_phases_fresh = std::vec![0.0_f32; num_bins];

        let mut pv = PhaseVocoder::new(fft_size, 4, 1.0).unwrap();
        pv.process_frame(&mags, &phases, &mut out_mags, &mut out_phases_after_reset);
        pv.reset();

        // After reset, the same frame should give the same result as a brand-new vocoder.
        let mut pv_fresh = PhaseVocoder::new(fft_size, 4, 1.0).unwrap();
        pv.process_frame(&mags, &phases, &mut out_mags, &mut out_phases_after_reset);
        pv_fresh.process_frame(&mags, &phases, &mut out_mags, &mut out_phases_fresh);

        for (a, b) in out_phases_after_reset.iter().zip(out_phases_fresh.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "phase mismatch after reset: got {a}, expected {b}"
            );
        }
    }

    #[test]
    fn continuous_tone_phase_increments_by_expected_advance_per_frame() {
        // For a pure tone at bin k_target whose input phase advances by exactly
        // expected_advance each hop, the output phase should also increment by
        // expected_advance each frame (stretch=1.0).
        //
        // The first frame is a "priming" frame: prev_phases starts at 0 while
        // the input phase is also 0, so the delta wraps and the first increment
        // is ambiguous.  We therefore verify the per-frame increment from frame
        // 2 onward, where prev_phases correctly tracks the input.
        let fft_size = 32;
        let hop = 8;
        let mut pv = PhaseVocoder::new(fft_size, hop, 1.0).unwrap();
        let num_bins = fft_size / 2 + 1;

        let k_target = 3usize;
        let expected_advance = TWO_PI * (k_target as f32) * (hop as f32) / (fft_size as f32);

        let mut mags = std::vec![0.0_f32; num_bins];
        mags[k_target] = 1.0;

        let mut out_mags = std::vec![0.0_f32; num_bins];
        let mut out_phases = std::vec![0.0_f32; num_bins];

        // Priming frame: prev_phases gets set to the initial input phases.
        let zero_phases = std::vec![0.0_f32; num_bins];
        pv.process_frame(&mags, &zero_phases, &mut out_mags, &mut out_phases);
        let mut prev_out_phase = out_phases[k_target];
        let mut accumulated_input_phase = expected_advance;

        // Frames 2-8: verify the per-frame phase increment is constant.
        for _ in 0..7 {
            let mut phases = std::vec![0.0_f32; num_bins];
            phases[k_target] = accumulated_input_phase;

            pv.process_frame(&mags, &phases, &mut out_mags, &mut out_phases);

            let increment = out_phases[k_target] - prev_out_phase;
            assert!(
                (increment - expected_advance).abs() < 1e-3,
                "per-frame phase increment {increment:.4} differs from expected {expected_advance:.4}"
            );

            prev_out_phase = out_phases[k_target];
            accumulated_input_phase += expected_advance;
        }
    }

    #[test]
    fn principal_argument_wraps_correctly() {
        // principal_argument must return values in [-π, π].
        let cases = [0.0, PI, -PI, 3.0 * PI, -3.0 * PI, 0.1, -0.1, 7.0, -7.0];
        for &x in &cases {
            let wrapped = principal_argument(x);
            assert!(
                wrapped >= -PI - 1e-5 && wrapped <= PI + 1e-5,
                "principal_argument({x}) = {wrapped} is out of [-π, π]"
            );
        }
    }
}
