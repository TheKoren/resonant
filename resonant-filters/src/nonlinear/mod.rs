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

mod moog_ladder;
mod saturating_biquad;
mod svf;

pub use moog_ladder::MoogLadder;
pub use saturating_biquad::SaturatingBiquad;
pub use svf::{StateVariableFilter, SvfOutputs};

#[cfg(feature = "alloc")]
mod response_impls {
    extern crate alloc;
    use alloc::vec;

    use core::f32::consts::PI;

    #[allow(unused_imports)]
    use num_traits::float::Float as _;

    use super::*;
    use crate::response::{FilterError, FilterResponseExt, FrequencyResponse};

    /// Number of samples to let each frequency settle before measuring.
    const N_SETTLE: usize = 512;
    /// Number of samples over which to measure peak amplitude.
    const N_MEASURE: usize = 128;

    /// Estimate magnitude at a single frequency via sine sweep.
    fn sweep_magnitude<F: FnMut(f32) -> f32>(mut process: F, freq: f32, sample_rate: f32) -> f32 {
        if freq < 1.0 {
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
            // Phase is not computable from a peak-amplitude sweep measurement.
            let phases = vec![0.0_f32; n_points];
            for i in 0..n_points {
                let freq = nyquist * i as f32 / (n_points - 1) as f32;
                frequencies[i] = freq;
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
            // Phase is not computable from a peak-amplitude sweep measurement.
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
            // Phase is not computable from a peak-amplitude sweep measurement.
            let phases = vec![0.0_f32; n_points];
            for i in 0..n_points {
                let freq = nyquist * i as f32 / (n_points - 1) as f32;
                frequencies[i] = freq;
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

#[cfg(test)]
mod tests {
    #[cfg(feature = "alloc")]
    mod alloc_tests {
        extern crate std;
        use crate::design;
        use crate::nonlinear::{MoogLadder, SaturatingBiquad, StateVariableFilter};
        use crate::response::FilterResponseExt;
        use crate::BiquadCoeffs;

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
