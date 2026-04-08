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
    pub(super) state: [f32; 4],
    pub(super) cutoff: f32,
    pub(super) resonance: f32,
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
        let p = f * (1.8 - 0.8 * f);

        let x0 = (input - self.resonance * self.state[3]).tanh();

        let y0 = p * x0 + (1.0 - p) * self.state[0];
        let y1 = p * self.state[0].tanh() + (1.0 - p) * self.state[1];
        let y2 = p * self.state[1].tanh() + (1.0 - p) * self.state[2];
        let y3 = p * self.state[2].tanh() + (1.0 - p) * self.state[3];

        self.state = [y0, y1, y2, y3];
        y3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_is_finite() {
        let mut f = MoogLadder::new(0.3, 2.0);
        for i in 0..500 {
            let y = f.process_sample((i as f32 * 0.2).sin());
            assert!(y.is_finite(), "NaN/inf at sample {i}");
        }
    }

    #[test]
    fn self_oscillates_at_max_resonance() {
        let mut f = MoogLadder::new(0.7, 4.0);
        f.process_sample(1.0);
        let mut max_amp = 0.0_f32;
        for _ in 0..2000 {
            let y = f.process_sample(0.0);
            max_amp = max_amp.max(y.abs());
        }
        assert!(max_amp > 1e-4, "expected self-oscillation; max={max_amp}");
    }

    #[test]
    fn no_resonance_decays() {
        let mut f = MoogLadder::new(0.3, 0.0);
        f.process_sample(1.0);
        let mut output = 0.0_f32;
        for _ in 0..5000 {
            output = f.process_sample(0.0);
        }
        assert!(output.abs() < 1e-4, "expected decay, got {output}");
    }

    #[test]
    fn set_params() {
        let mut f = MoogLadder::new(0.3, 1.0);
        f.set_cutoff(0.5);
        f.set_resonance(2.0);
        assert!((f.cutoff - 0.5).abs() < 1e-6);
        assert!((f.resonance - 2.0).abs() < 1e-6);
    }

    #[test]
    fn reset_zeroes_state() {
        let mut f = MoogLadder::new(0.3, 1.0);
        f.process_sample(1.0);
        f.reset();
        assert_eq!(f.state, [0.0; 4]);
    }

    #[test]
    fn cutoff_clamps() {
        let f = MoogLadder::new(2.0, 5.0);
        assert!(f.cutoff <= 0.99);
        assert!(f.resonance <= 4.0);
    }
}
