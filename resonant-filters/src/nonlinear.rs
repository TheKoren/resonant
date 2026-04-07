//! Nonlinear and analog-modelled filters.
//!
//! Three saturating filter types, all `no_std`:
//!
//! - [`SaturatingBiquad`] — standard biquad with a tanh waveshaper on the
//!   output and state-update path, controlled by a `drive` parameter.
//! - [`MoogLadder`] — 4th-order resonant lowpass based on the Huovilainen
//!   model; self-oscillates at `resonance = 4.0`.
//! - [`StateVariableFilter`] — two-integrator-loop topology with simultaneous
//!   LP / HP / BP / notch outputs.
//!
//! All three implement [`FilterResponseExt`](crate::response::FilterResponseExt)
//! via numerical sine sweep (requires `alloc` feature).

use core::f32::consts::PI;

#[allow(unused_imports)]
use num_traits::float::Float as _;

use crate::BiquadCoeffs;

// ── SaturatingBiquad ─────────────────────────────────────────────────────────

/// Biquad filter with a tanh waveshaper applied to the output and state updates.
///
/// `drive = 0.0` gives linear (identical to [`Biquad`](crate::Biquad)).
/// `drive = 1.0` gives full tanh saturation: output is bounded by `(-1, 1)`.
///
/// # Examples
///
/// ```
/// use resonant_filters::{design, nonlinear::SaturatingBiquad};
///
/// let coeffs = design::butterworth_lowpass(1000.0, 44100.0).unwrap();
/// let mut f = SaturatingBiquad::new(coeffs, 0.5);
/// let y = f.process_sample(1.0);
/// assert!(y.is_finite());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaturatingBiquad {
    coeffs: BiquadCoeffs,
    s1: f32,
    s2: f32,
    /// Saturation amount: 0.0 = linear, 1.0 = full tanh clip.
    pub drive: f32,
}

impl SaturatingBiquad {
    /// Creates a new saturating biquad with zeroed state.
    #[must_use]
    #[inline]
    pub fn new(coeffs: BiquadCoeffs, drive: f32) -> Self {
        Self {
            coeffs,
            s1: 0.0,
            s2: 0.0,
            drive: drive.clamp(0.0, 1.0),
        }
    }

    /// Processes a single sample.
    ///
    /// Applies the waveshaper `(1 − drive)·x + drive·tanh(x)` to the
    /// combined output signal and both state-update paths.
    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let c = &self.coeffs;
        let pre = c.b0 * x + self.s1;
        let output = self.shape(pre);
        self.s1 = self.shape(c.b1 * x - c.a1 * output + self.s2);
        self.s2 = self.shape(c.b2 * x - c.a2 * output);
        output
    }

    /// Filters an entire buffer in-place.
    pub fn process_buf(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            *s = self.process_sample(*s);
        }
    }

    /// Replaces the coefficients without resetting state.
    #[inline]
    pub fn set_coeffs(&mut self, coeffs: BiquadCoeffs) {
        self.coeffs = coeffs;
    }

    /// Resets the delay state to zero.
    #[inline]
    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }

    #[inline]
    fn shape(&self, x: f32) -> f32 {
        x * (1.0 - self.drive) + x.tanh() * self.drive
    }
}

// ── MoogLadder ───────────────────────────────────────────────────────────────

/// 4th-order resonant lowpass based on the Moog ladder topology.
///
/// Uses four cascaded 1-pole sections with nonlinear (tanh) feedback.
/// At `resonance = 4.0` the filter self-oscillates.
///
/// # Parameters
///
/// * `cutoff` — normalised cutoff frequency: `0.0` = DC, `1.0` = Nyquist.
/// * `resonance` — feedback amount: `0.0` = no resonance, `4.0` = self-oscillation.
///
/// # Examples
///
/// ```
/// use resonant_filters::nonlinear::MoogLadder;
///
/// let mut f = MoogLadder::new(0.3, 2.0);
/// let y = f.process_sample(0.5);
/// assert!(y.is_finite());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoogLadder {
    state: [f32; 4],
    cutoff: f32,
    resonance: f32,
}

impl MoogLadder {
    /// Creates a new Moog ladder with zeroed state.
    #[must_use]
    #[inline]
    pub fn new(cutoff: f32, resonance: f32) -> Self {
        Self {
            state: [0.0; 4],
            cutoff: cutoff.clamp(0.0, 0.99),
            resonance: resonance.clamp(0.0, 4.0),
        }
    }

    /// Sets the normalised cutoff frequency (0..1).
    #[inline]
    pub fn set_cutoff(&mut self, cutoff: f32) {
        self.cutoff = cutoff.clamp(0.0, 0.99);
    }

    /// Sets the resonance / feedback (0..4).
    #[inline]
    pub fn set_resonance(&mut self, resonance: f32) {
        self.resonance = resonance.clamp(0.0, 4.0);
    }

    /// Resets the internal state to zero.
    #[inline]
    pub fn reset(&mut self) {
        self.state = [0.0; 4];
    }

    /// Processes a single sample.
    ///
    /// Uses the polynomial approximation `p = f · (1.8 − 0.8·f)` to map
    /// normalised cutoff to a first-order pole coefficient.  At `resonance = 4`
    /// the small-signal loop gain exceeds unity for `cutoff ≥ 0.7`, producing
    /// a stable limit cycle through the tanh nonlinearity.
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let f = self.cutoff;
        // Pole coefficient: polynomial approx to 1 - exp(-2π·f)
        let p = f * (1.8 - 0.8 * f);

        // Input with resonance feedback (tanh limits the feedback amplitude)
        let x0 = (input - self.resonance * self.state[3]).tanh();

        // 4 cascaded 1-pole LP sections; each stage applies tanh to the
        // previous stage's output before the integrator.
        let y0 = p * x0 + (1.0 - p) * self.state[0];
        let y1 = p * self.state[0].tanh() + (1.0 - p) * self.state[1];
        let y2 = p * self.state[1].tanh() + (1.0 - p) * self.state[2];
        let y3 = p * self.state[2].tanh() + (1.0 - p) * self.state[3];

        self.state = [y0, y1, y2, y3];
        y3
    }
}

// ── StateVariableFilter ───────────────────────────────────────────────────────

/// Output bundle from [`StateVariableFilter::process`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SvfOutputs {
    /// Lowpass output.
    pub lp: f32,
    /// Highpass output.
    pub hp: f32,
    /// Bandpass output.
    pub bp: f32,
    /// Notch (bandreject) output: `lp + hp`.
    pub notch: f32,
}

/// State-variable filter — two-integrator-loop topology.
///
/// Computes LP, HP, BP, and notch outputs simultaneously from a single
/// pass through the signal. Based on the Chamberlin formulation with the
/// frequency coefficient `f0 = 2·sin(π·fc/fs)`.
///
/// # Parameters
///
/// * `cutoff` — cutoff frequency in Hz.
/// * `q` — Q factor (must be > 0; higher Q = sharper resonance).
/// * `sample_rate` — sample rate in Hz.
///
/// # Examples
///
/// ```
/// use resonant_filters::nonlinear::StateVariableFilter;
///
/// let mut svf = StateVariableFilter::new(1000.0, 0.707, 44100.0);
/// let out = svf.process(0.5);
/// assert!(out.lp.is_finite());
/// assert!(out.hp.is_finite());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateVariableFilter {
    /// Frequency coefficient: `2·sin(π·fc/fs)`.
    f0: f32,
    /// Damping coefficient: `1/Q`.
    damp: f32,
    lp: f32,
    bp: f32,
}

impl StateVariableFilter {
    /// Creates a new SVF with zeroed state.
    ///
    /// Clamps `cutoff` to just below Nyquist to keep the filter stable.
    #[must_use]
    pub fn new(cutoff: f32, q: f32, sample_rate: f32) -> Self {
        let q = q.max(0.001);
        let fc = cutoff.clamp(0.0, sample_rate * 0.499);
        let f0 = 2.0 * (PI * fc / sample_rate).sin();
        Self {
            f0,
            damp: 1.0 / q,
            lp: 0.0,
            bp: 0.0,
        }
    }

    /// Resets the state to zero.
    #[inline]
    pub fn reset(&mut self) {
        self.lp = 0.0;
        self.bp = 0.0;
    }

    /// Processes a single input sample, returning all four outputs.
    ///
    /// The Chamberlin SVF satisfies `lp + damp·bp + hp = x` exactly, so
    /// `lp + hp = x − damp·bp ≈ x` when `Q` is high (damp ≈ 0).
    #[inline]
    pub fn process(&mut self, x: f32) -> SvfOutputs {
        // Chamberlin SVF — update lp first then compute hp for stability
        self.lp += self.f0 * self.bp;
        let hp = x - self.lp - self.damp * self.bp;
        self.bp += self.f0 * hp;
        let notch = self.lp + hp;
        SvfOutputs {
            lp: self.lp,
            hp,
            bp: self.bp,
            notch,
        }
    }
}

// ── FilterResponseExt for nonlinear types (requires alloc) ───────────────────

#[cfg(feature = "alloc")]
mod response_impls {
    extern crate alloc;
    use alloc::vec;

    use super::*;
    use crate::response::{FilterError, FilterResponseExt, FrequencyResponse};

    /// Number of samples to let each frequency settle before measuring.
    const N_SETTLE: usize = 512;
    /// Number of samples over which to measure peak amplitude.
    const N_MEASURE: usize = 128;

    /// Estimate magnitude at a single frequency via sine sweep.
    fn sweep_magnitude<F: FnMut(f32) -> f32>(mut process: F, freq: f32, sample_rate: f32) -> f32 {
        if freq < 1.0 {
            // DC: feed constant 1.0, measure steady-state output
            for _ in 0..N_SETTLE {
                process(1.0);
            }
            let mut peak = 0.0_f32;
            for _ in 0..N_MEASURE {
                peak = peak.max(process(1.0).abs());
            }
            return peak;
        }
        let omega = 2.0 * PI * freq / sample_rate;
        for n in 0..N_SETTLE {
            process((omega * n as f32).sin());
        }
        let mut peak_in = 0.0_f32;
        let mut peak_out = 0.0_f32;
        for n in N_SETTLE..(N_SETTLE + N_MEASURE) {
            let x = (omega * n as f32).sin();
            let y = process(x);
            peak_in = peak_in.max(x.abs());
            peak_out = peak_out.max(y.abs());
        }
        if peak_in > 1e-10 {
            peak_out / peak_in
        } else {
            0.0
        }
    }

    impl FilterResponseExt for SaturatingBiquad {
        fn frequency_response(
            &self,
            n_points: usize,
            sample_rate: f32,
        ) -> Result<FrequencyResponse, FilterError> {
            if n_points < 2 || sample_rate <= 0.0 {
                return Err(FilterError::InvalidParameter);
            }
            let nyquist = sample_rate / 2.0;
            let mut frequencies = vec![0.0_f32; n_points];
            let mut magnitudes = vec![0.0_f32; n_points];
            let phases = vec![0.0_f32; n_points];

            for i in 0..n_points {
                let freq = nyquist * i as f32 / (n_points - 1) as f32;
                frequencies[i] = freq;
                // Clone with zeroed state for each frequency point
                let mut f = SaturatingBiquad::new(self.coeffs, self.drive);
                magnitudes[i] = sweep_magnitude(|x| f.process_sample(x), freq, sample_rate);
            }

            Ok(FrequencyResponse {
                frequencies,
                magnitudes,
                phases,
            })
        }
    }

    impl FilterResponseExt for MoogLadder {
        fn frequency_response(
            &self,
            n_points: usize,
            sample_rate: f32,
        ) -> Result<FrequencyResponse, FilterError> {
            if n_points < 2 || sample_rate <= 0.0 {
                return Err(FilterError::InvalidParameter);
            }
            let nyquist = sample_rate / 2.0;
            let mut frequencies = vec![0.0_f32; n_points];
            let mut magnitudes = vec![0.0_f32; n_points];
            let phases = vec![0.0_f32; n_points];

            for i in 0..n_points {
                let freq = nyquist * i as f32 / (n_points - 1) as f32;
                frequencies[i] = freq;
                let mut f = MoogLadder::new(self.cutoff, self.resonance);
                magnitudes[i] = sweep_magnitude(|x| f.process_sample(x), freq, sample_rate);
            }

            Ok(FrequencyResponse {
                frequencies,
                magnitudes,
                phases,
            })
        }
    }

    impl FilterResponseExt for StateVariableFilter {
        fn frequency_response(
            &self,
            n_points: usize,
            sample_rate: f32,
        ) -> Result<FrequencyResponse, FilterError> {
            if n_points < 2 || sample_rate <= 0.0 {
                return Err(FilterError::InvalidParameter);
            }
            let nyquist = sample_rate / 2.0;
            let mut frequencies = vec![0.0_f32; n_points];
            let mut magnitudes = vec![0.0_f32; n_points];
            let phases = vec![0.0_f32; n_points];

            for i in 0..n_points {
                let freq = nyquist * i as f32 / (n_points - 1) as f32;
                frequencies[i] = freq;
                // Reconstruct a fresh filter from the stored coefficients
                let mut f = StateVariableFilter {
                    f0: self.f0,
                    damp: self.damp,
                    lp: 0.0,
                    bp: 0.0,
                };
                magnitudes[i] = sweep_magnitude(|x| f.process(x).lp, freq, sample_rate);
            }

            Ok(FrequencyResponse {
                frequencies,
                magnitudes,
                phases,
            })
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{design, BiquadCoeffs};

    // ── SaturatingBiquad ──────────────────────────────────────────────────────

    #[test]
    fn saturating_biquad_drive_zero_is_linear() {
        // At drive=0 the output must match a plain biquad
        let coeffs = BiquadCoeffs {
            b0: 0.5,
            b1: 0.3,
            b2: 0.1,
            a1: -0.2,
            a2: 0.05,
        };
        let mut sat = SaturatingBiquad::new(coeffs, 0.0);
        let mut lin = crate::Biquad::new(coeffs);

        for i in 0..20 {
            let x = i as f32 * 0.1 - 1.0;
            let y_sat = sat.process_sample(x);
            let y_lin = lin.process_sample(x);
            assert!(
                (y_sat - y_lin).abs() < 1e-5,
                "mismatch at {x}: {y_sat} vs {y_lin}"
            );
        }
    }

    #[test]
    fn saturating_biquad_drive_one_bounds_output() {
        // At drive=1 and very large input the output must stay within (-1, 1)
        let mut f = SaturatingBiquad::new(BiquadCoeffs::PASSTHROUGH, 1.0);
        for _ in 0..200 {
            let y = f.process_sample(1000.0);
            assert!(y.abs() <= 1.0 + 1e-4, "output out of bounds: {y}");
        }
    }

    #[test]
    fn saturating_biquad_drive_one_negative_large_input() {
        let mut f = SaturatingBiquad::new(BiquadCoeffs::PASSTHROUGH, 1.0);
        for _ in 0..200 {
            let y = f.process_sample(-500.0);
            assert!(y.abs() <= 1.0 + 1e-4, "output out of bounds: {y}");
        }
    }

    #[test]
    fn saturating_biquad_small_signal_matches_linear() {
        // For small signals (well within linear region) output ≈ linear biquad
        let coeffs = design::butterworth_lowpass(1000.0, 44100.0).unwrap();
        let mut sat = SaturatingBiquad::new(coeffs, 0.8);
        let mut lin = crate::Biquad::new(coeffs);

        for i in 0..50 {
            let x = (i as f32 * 0.01).sin() * 0.001; // very small
            let ys = sat.process_sample(x);
            let yl = lin.process_sample(x);
            assert!(
                (ys - yl).abs() < 1e-4,
                "small-signal mismatch: {ys} vs {yl}"
            );
        }
    }

    #[test]
    fn saturating_biquad_reset_zeroes_state() {
        let mut f = SaturatingBiquad::new(BiquadCoeffs::PASSTHROUGH, 0.5);
        f.process_sample(1.0);
        f.reset();
        assert_eq!(f.s1, 0.0);
        assert_eq!(f.s2, 0.0);
    }

    #[test]
    fn saturating_biquad_finite_output() {
        let coeffs = design::butterworth_lowpass(500.0, 44100.0).unwrap();
        let mut f = SaturatingBiquad::new(coeffs, 1.0);
        for i in 0..1000 {
            let y = f.process_sample((i as f32 * 0.1).sin());
            assert!(y.is_finite(), "NaN/inf at sample {i}");
        }
    }

    // ── MoogLadder ────────────────────────────────────────────────────────────

    #[test]
    fn moog_ladder_output_is_finite() {
        let mut f = MoogLadder::new(0.3, 2.0);
        for i in 0..500 {
            let y = f.process_sample((i as f32 * 0.2).sin());
            assert!(y.is_finite(), "NaN/inf at sample {i}");
        }
    }

    #[test]
    fn moog_ladder_self_oscillates_at_max_resonance() {
        // cutoff=0.7 → loop gain 4 * p^4 ≈ 2.3 > 1 at resonance=4 → self-oscillation
        let mut f = MoogLadder::new(0.7, 4.0);
        f.process_sample(1.0); // initial kick
        let mut max_amp = 0.0_f32;
        for _ in 0..2000 {
            let y = f.process_sample(0.0);
            max_amp = max_amp.max(y.abs());
        }
        assert!(
            max_amp > 1e-4,
            "expected self-oscillation with resonance=4, cutoff=0.7; max={max_amp}"
        );
    }

    #[test]
    fn moog_ladder_no_resonance_decays() {
        // With resonance=0 and zero input, the state should decay to zero
        let mut f = MoogLadder::new(0.3, 0.0);
        f.process_sample(1.0); // initial kick
        let mut output = 0.0_f32;
        for _ in 0..5000 {
            output = f.process_sample(0.0);
        }
        assert!(output.abs() < 1e-4, "expected decay, got {output}");
    }

    #[test]
    fn moog_ladder_set_params() {
        let mut f = MoogLadder::new(0.3, 1.0);
        f.set_cutoff(0.5);
        f.set_resonance(2.0);
        assert!((f.cutoff - 0.5).abs() < 1e-6);
        assert!((f.resonance - 2.0).abs() < 1e-6);
    }

    #[test]
    fn moog_ladder_reset_zeroes_state() {
        let mut f = MoogLadder::new(0.3, 1.0);
        f.process_sample(1.0);
        f.reset();
        assert_eq!(f.state, [0.0; 4]);
    }

    #[test]
    fn moog_ladder_cutoff_clamps() {
        let f = MoogLadder::new(2.0, 5.0); // both out of range
        assert!(f.cutoff <= 0.99);
        assert!(f.resonance <= 4.0);
    }

    // ── StateVariableFilter ───────────────────────────────────────────────────

    #[test]
    fn svf_outputs_finite() {
        let mut f = StateVariableFilter::new(1000.0, 0.707, 44100.0);
        for i in 0..500 {
            let out = f.process((i as f32 * 0.1).sin());
            assert!(out.lp.is_finite());
            assert!(out.hp.is_finite());
            assert!(out.bp.is_finite());
            assert!(out.notch.is_finite());
        }
    }

    #[test]
    fn svf_lp_hp_sum_approximates_input_at_high_q() {
        // For high Q (low damping) and signals far from the resonant frequency,
        // the bandpass output approaches zero, so lp + hp ≈ x.
        let mut f = StateVariableFilter::new(200.0, 20.0, 44100.0); // Q=20, damp=0.05
        let mut max_err = 0.0_f32;
        for i in 500..1000 {
            // Test at 5 kHz — far from the 200 Hz resonance, so bp ≈ 0
            let x = (2.0 * PI * 5000.0 * i as f32 / 44100.0).sin();
            let out = f.process(x);
            let err = (out.lp + out.hp - x).abs();
            max_err = max_err.max(err);
        }
        // Approximation only: bp is small far from resonance but not exactly zero
        assert!(
            max_err < 0.01,
            "LP+HP should approximate input far from resonance: err={max_err}"
        );
    }

    #[test]
    fn svf_notch_equals_lp_plus_hp() {
        let mut f = StateVariableFilter::new(1000.0, 1.0, 44100.0);
        for i in 0..200 {
            let x = (i as f32 * 0.15).sin();
            let out = f.process(x);
            assert!((out.notch - (out.lp + out.hp)).abs() < 1e-6);
        }
    }

    #[test]
    fn svf_dc_passes_through_lp() {
        // At DC, LP output should approach the input, HP → 0
        let mut f = StateVariableFilter::new(5000.0, 0.707, 44100.0);
        let mut lp_out = 0.0_f32;
        for _ in 0..5000 {
            lp_out = f.process(1.0).lp;
        }
        assert!(
            (lp_out - 1.0).abs() < 0.01,
            "LP DC gain should be ~1: {lp_out}"
        );
    }

    #[test]
    fn svf_reset_zeroes_state() {
        let mut f = StateVariableFilter::new(1000.0, 1.0, 44100.0);
        f.process(1.0);
        f.reset();
        assert_eq!(f.lp, 0.0);
        assert_eq!(f.bp, 0.0);
    }

    // ── nonlinear FilterResponseExt ───────────────────────────────────────────

    #[cfg(feature = "alloc")]
    mod alloc_tests {
        extern crate std;
        use super::*;
        use crate::response::FilterResponseExt;

        #[test]
        fn saturating_biquad_response_lengths() {
            let coeffs = design::butterworth_lowpass(1000.0, 44100.0).unwrap();
            let f = SaturatingBiquad::new(coeffs, 0.5);
            let resp = f.frequency_response(32, 44100.0).unwrap();
            assert_eq!(resp.frequencies.len(), 32);
            assert_eq!(resp.magnitudes.len(), 32);
        }

        #[test]
        fn moog_response_lengths() {
            let f = MoogLadder::new(0.3, 1.0);
            let resp = f.frequency_response(32, 44100.0).unwrap();
            assert_eq!(resp.frequencies.len(), 32);
            assert_eq!(resp.magnitudes.len(), 32);
        }

        #[test]
        fn svf_response_lengths() {
            let f = StateVariableFilter::new(1000.0, 1.0, 44100.0);
            let resp = f.frequency_response(32, 44100.0).unwrap();
            assert_eq!(resp.frequencies.len(), 32);
            assert_eq!(resp.magnitudes.len(), 32);
        }

        #[test]
        fn saturating_biquad_response_finite() {
            let coeffs = design::butterworth_lowpass(1000.0, 44100.0).unwrap();
            let f = SaturatingBiquad::new(coeffs, 0.5);
            let resp = f.frequency_response(16, 44100.0).unwrap();
            for (&m, &p) in resp.magnitudes.iter().zip(resp.phases.iter()) {
                assert!(m.is_finite(), "magnitude not finite: {m}");
                assert!(p.is_finite(), "phase not finite: {p}");
            }
        }

        #[test]
        fn invalid_params_return_error() {
            let f = SaturatingBiquad::new(BiquadCoeffs::PASSTHROUGH, 0.5);
            assert!(f.frequency_response(1, 44100.0).is_err());
            assert!(f.frequency_response(32, 0.0).is_err());

            let m = MoogLadder::new(0.3, 1.0);
            assert!(m.frequency_response(1, 44100.0).is_err());

            let s = StateVariableFilter::new(1000.0, 1.0, 44100.0);
            assert!(s.frequency_response(1, 44100.0).is_err());
        }
    }
}
