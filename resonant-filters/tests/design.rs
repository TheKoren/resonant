use resonant_filters::design;
use resonant_filters::Biquad;

const SR: f64 = 48000.0;

#[test]
fn lowpass_steady_state_passes_dc() {
    let c = design::butterworth_lowpass(2000.0, SR).unwrap();
    let mut f = Biquad::new(c);
    let mut y = 0.0;
    for _ in 0..2000 {
        y = f.process_sample(1.0);
    }
    assert!((y - 1.0).abs() < 1e-3, "DC output = {y}");
}

#[test]
fn highpass_steady_state_blocks_dc() {
    let c = design::butterworth_highpass(2000.0, SR).unwrap();
    let mut f = Biquad::new(c);
    let mut y = 0.0;
    for _ in 0..2000 {
        y = f.process_sample(1.0);
    }
    assert!(y.abs() < 1e-3, "DC output = {y}");
}

#[test]
fn lowpass_and_highpass_complement() {
    // At the cutoff frequency, both filters should have ~-3dB gain
    let cutoff = 4000.0;
    let lp = design::butterworth_lowpass(cutoff, SR).unwrap();
    let hp = design::butterworth_highpass(cutoff, SR).unwrap();

    let mut f_lp = Biquad::new(lp);
    let mut f_hp = Biquad::new(hp);

    let freq = cutoff as f32;
    let mut max_lp = 0.0_f32;
    let mut max_hp = 0.0_f32;

    for i in 0..8000 {
        let x = (2.0 * core::f32::consts::PI * freq * i as f32 / SR as f32).sin();
        let y_lp = f_lp.process_sample(x);
        let y_hp = f_hp.process_sample(x);
        if i > 2000 {
            max_lp = max_lp.max(y_lp.abs());
            max_hp = max_hp.max(y_hp.abs());
        }
    }

    // Butterworth at cutoff: -3dB ≈ 0.707
    assert!((max_lp - 0.707).abs() < 0.1, "LP at cutoff = {max_lp}");
    assert!((max_hp - 0.707).abs() < 0.1, "HP at cutoff = {max_hp}");
}

#[test]
fn invalid_params_return_err() {
    use resonant_filters::DesignError;
    assert_eq!(
        design::butterworth_lowpass(0.0, SR),
        Err(DesignError::FrequencyOutOfRange)
    );
    assert_eq!(
        design::butterworth_lowpass(-1.0, SR),
        Err(DesignError::FrequencyOutOfRange)
    );
    assert_eq!(
        design::butterworth_lowpass(SR / 2.0, SR),
        Err(DesignError::FrequencyOutOfRange)
    );
    assert_eq!(
        design::butterworth_lowpass(100.0, 0.0),
        Err(DesignError::SampleRateOutOfRange)
    );
    assert_eq!(
        design::butterworth_highpass(0.0, SR),
        Err(DesignError::FrequencyOutOfRange)
    );
    assert_eq!(
        design::butterworth_highpass(SR, SR),
        Err(DesignError::FrequencyOutOfRange)
    );
}

#[test]
fn different_sample_rates_produce_different_coeffs() {
    let c1 = design::butterworth_lowpass(1000.0, 44100.0).unwrap();
    let c2 = design::butterworth_lowpass(1000.0, 96000.0).unwrap();
    assert_ne!(c1, c2);
}
