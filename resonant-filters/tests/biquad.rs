use resonant_filters::{Biquad, BiquadCoeffs};

#[test]
fn passthrough_preserves_signal() {
    let mut f = Biquad::new(BiquadCoeffs::PASSTHROUGH);
    let input = [0.1_f32, -0.5, 0.9, 0.0, -1.0];
    for &x in &input {
        assert!((f.process_sample(x) - x).abs() < 1e-6);
    }
}

#[test]
fn process_buf_filters_in_place() {
    let coeffs = BiquadCoeffs {
        b0: 0.5,
        b1: 0.25,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    };
    let mut f = Biquad::new(coeffs);
    let mut buf = [1.0_f32, 0.0, 0.0, 0.0];
    f.process_buf(&mut buf);
    // h[0]=b0=0.5, h[1]=b1=0.25, h[2]=0, h[3]=0
    assert!((buf[0] - 0.5).abs() < 1e-6);
    assert!((buf[1] - 0.25).abs() < 1e-6);
    assert!(buf[2].abs() < 1e-6);
}

#[test]
fn dc_gain_converges() {
    let coeffs = BiquadCoeffs {
        b0: 0.2,
        b1: 0.4,
        b2: 0.2,
        a1: -0.6,
        a2: 0.2,
    };
    let expected_dc = (coeffs.b0 + coeffs.b1 + coeffs.b2) / (1.0 + coeffs.a1 + coeffs.a2);
    let mut f = Biquad::new(coeffs);
    let mut y = 0.0;
    for _ in 0..2000 {
        y = f.process_sample(1.0);
    }
    assert!(
        (y - expected_dc).abs() < 1e-3,
        "expected {expected_dc}, got {y}"
    );
}

#[test]
fn reset_then_reprocess_matches_fresh() {
    let coeffs = BiquadCoeffs {
        b0: 0.3,
        b1: 0.5,
        b2: 0.2,
        a1: -0.4,
        a2: 0.1,
    };
    let input = [1.0_f32, 0.5, -0.3, 0.7];

    let mut f1 = Biquad::new(coeffs);
    let out1: Vec<f32> = input.iter().map(|&x| f1.process_sample(x)).collect();

    // Process some junk, reset, then reprocess
    let mut f2 = Biquad::new(coeffs);
    for _ in 0..50 {
        f2.process_sample(0.9);
    }
    f2.reset();
    let out2: Vec<f32> = input.iter().map(|&x| f2.process_sample(x)).collect();

    for (a, b) in out1.iter().zip(out2.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn empty_buf_is_noop() {
    let mut f = Biquad::new(BiquadCoeffs::PASSTHROUGH);
    let mut buf: [f32; 0] = [];
    f.process_buf(&mut buf);
}
