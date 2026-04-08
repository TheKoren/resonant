use core::f32::consts::PI;

#[allow(unused_imports)]
use num_traits::float::Float as _;

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
    pub(super) f0: f32,
    /// Damping coefficient: `1/Q`.
    pub(super) damp: f32,
    pub(super) lp: f32,
    pub(super) bp: f32,
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

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    #[test]
    fn outputs_finite() {
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
    fn lp_hp_sum_approximates_input_at_high_q() {
        let mut f = StateVariableFilter::new(200.0, 20.0, 44100.0);
        let mut max_err = 0.0_f32;
        for i in 500..1000 {
            let x = (2.0 * PI * 5000.0 * i as f32 / 44100.0).sin();
            let out = f.process(x);
            let err = (out.lp + out.hp - x).abs();
            max_err = max_err.max(err);
        }
        assert!(
            max_err < 0.01,
            "LP+HP should approximate input far from resonance: err={max_err}"
        );
    }

    #[test]
    fn notch_equals_lp_plus_hp() {
        let mut f = StateVariableFilter::new(1000.0, 1.0, 44100.0);
        for i in 0..200 {
            let x = (i as f32 * 0.15).sin();
            let out = f.process(x);
            assert!((out.notch - (out.lp + out.hp)).abs() < 1e-6);
        }
    }

    #[test]
    fn dc_passes_through_lp() {
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
    fn reset_zeroes_state() {
        let mut f = StateVariableFilter::new(1000.0, 1.0, 44100.0);
        f.process(1.0);
        f.reset();
        assert_eq!(f.lp, 0.0);
        assert_eq!(f.bp, 0.0);
    }
}
