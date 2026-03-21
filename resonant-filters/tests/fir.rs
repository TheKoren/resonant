#![cfg(feature = "alloc")]

use resonant_filters::fir::Fir;

#[test]
fn impulse_response_is_coefficients() {
    let coeffs = vec![0.5, 0.3, 0.2];
    let mut f = Fir::new(coeffs.clone());

    for (i, &c) in coeffs.iter().enumerate() {
        let y = if i == 0 {
            f.process_sample(1.0)
        } else {
            f.process_sample(0.0)
        };
        assert!((y - c).abs() < 1e-6, "tap {i}: expected {c}, got {y}");
    }
    // After all taps: zero output
    assert!(f.process_sample(0.0).abs() < 1e-6);
}

#[test]
fn dc_gain_is_sum_of_coefficients() {
    let coeffs = vec![0.2, 0.3, 0.3, 0.2];
    let dc_gain: f32 = coeffs.iter().sum();
    let mut f = Fir::new(coeffs);

    // Feed constant 1.0 long enough to settle
    let mut y = 0.0;
    for _ in 0..20 {
        y = f.process_sample(1.0);
    }
    assert!((y - dc_gain).abs() < 1e-5, "expected {dc_gain}, got {y}");
}

#[test]
fn process_buf_in_place() {
    let mut f = Fir::new(vec![1.0, 0.0]);
    let mut buf = [1.0_f32, 2.0, 3.0, 4.0];
    f.process_buf(&mut buf);
    assert!((buf[0] - 1.0).abs() < 1e-6);
    assert!((buf[1] - 2.0).abs() < 1e-6);
}

#[test]
fn reset_produces_same_output_as_fresh() {
    let coeffs = vec![0.4, 0.3, 0.2, 0.1];
    let input = [1.0_f32, 0.5, -0.3, 0.7];

    let mut f1 = Fir::new(coeffs.clone());
    let out1: Vec<f32> = input.iter().map(|&x| f1.process_sample(x)).collect();

    let mut f2 = Fir::new(coeffs);
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
fn num_taps_matches() {
    let f = Fir::new(vec![0.0; 16]);
    assert_eq!(f.num_taps(), 16);
}
