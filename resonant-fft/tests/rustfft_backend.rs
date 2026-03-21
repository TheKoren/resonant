#![cfg(feature = "rustfft")]

use resonant_fft::{rustfft_backend, Complex, FftError};

fn c(re: f32, im: f32) -> Complex<f32> {
    Complex::new(re, im)
}

#[test]
fn roundtrip_arbitrary_size() {
    for n in [3, 5, 6, 7, 9, 10, 12, 13] {
        let original: Vec<_> = (0..n).map(|i| c(i as f32 / n as f32, 0.0)).collect();
        let mut buf = original.clone();
        rustfft_backend::fft(&mut buf).unwrap();
        rustfft_backend::ifft(&mut buf).unwrap();
        for (a, b) in buf.iter().zip(original.iter()) {
            assert!((a - b).norm() < 1e-3, "size {n}: mismatch {a} vs {b}");
        }
    }
}

#[test]
fn matches_radix2_on_power_of_two() {
    let original: Vec<_> = (0..8).map(|i| c(i as f32, 0.0)).collect();

    let mut radix2_buf = original.clone();
    resonant_fft::radix2::fft(&mut radix2_buf).unwrap();

    let mut rustfft_buf = original.clone();
    rustfft_backend::fft(&mut rustfft_buf).unwrap();

    for (a, b) in radix2_buf.iter().zip(rustfft_buf.iter()) {
        assert!(
            (a - b).norm() < 1e-3,
            "backends disagree: radix2={a} vs rustfft={b}"
        );
    }
}

#[test]
fn empty_returns_error() {
    let mut buf: Vec<Complex<f32>> = vec![];
    assert_eq!(rustfft_backend::fft(&mut buf), Err(FftError::Empty));
    assert_eq!(rustfft_backend::ifft(&mut buf), Err(FftError::Empty));
}

#[test]
fn plan_roundtrip() {
    let plan = rustfft_backend::FftPlan::new(6);
    let original: Vec<_> = (0..6).map(|i| c(i as f32, 0.0)).collect();
    let mut buf = original.clone();
    plan.fft(&mut buf).unwrap();
    plan.ifft(&mut buf).unwrap();
    for (a, b) in buf.iter().zip(original.iter()) {
        assert!((a - b).norm() < 1e-3);
    }
}

#[test]
fn dc_signal_non_power_of_two() {
    let mut buf = vec![c(1.0, 0.0); 5];
    rustfft_backend::fft(&mut buf).unwrap();
    assert!((buf[0].re - 5.0).abs() < 1e-4);
    for b in &buf[1..] {
        assert!(b.norm() < 1e-3);
    }
}
