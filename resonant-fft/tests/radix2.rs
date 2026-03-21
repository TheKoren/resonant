use resonant_fft::{fft, ifft, Complex, FftError};

fn c(re: f32, im: f32) -> Complex<f32> {
    Complex::new(re, im)
}

#[test]
fn forward_then_inverse_roundtrip() {
    let original = [c(1.0, 0.0), c(-0.5, 0.0), c(0.25, 0.0), c(-0.75, 0.0)];
    let mut buf = original;
    fft(&mut buf).ok();
    ifft(&mut buf).ok();
    for (a, b) in buf.iter().zip(original.iter()) {
        assert!((a - b).norm() < 1e-5, "roundtrip mismatch: {a} vs {b}");
    }
}

#[test]
fn dc_signal_produces_impulse_spectrum() {
    let mut buf = [c(1.0, 0.0); 16];
    fft(&mut buf).ok();
    assert!((buf[0].re - 16.0).abs() < 1e-3);
    for b in &buf[1..] {
        assert!(b.norm() < 1e-3, "expected near-zero, got {b}");
    }
}

#[test]
fn impulse_produces_flat_spectrum() {
    let mut buf = [c(0.0, 0.0); 16];
    buf[0] = c(1.0, 0.0);
    fft(&mut buf).ok();
    for b in &buf {
        assert!((b.re - 1.0).abs() < 1e-4, "expected 1.0, got {}", b.re);
        assert!(b.im.abs() < 1e-4);
    }
}

#[test]
fn error_on_non_power_of_two() {
    let mut buf = [c(0.0, 0.0); 8];
    assert_eq!(fft(&mut buf[..3]), Err(FftError::NotPowerOfTwo(3)));
    assert_eq!(ifft(&mut buf[..6]), Err(FftError::NotPowerOfTwo(6)));
}

#[test]
fn error_on_empty() {
    let mut buf: [Complex<f32>; 0] = [];
    assert_eq!(fft(&mut buf), Err(FftError::Empty));
    assert_eq!(ifft(&mut buf), Err(FftError::Empty));
}

#[test]
fn parseval_energy_conservation() {
    let original = [
        c(0.1, 0.0),
        c(-0.3, 0.0),
        c(0.5, 0.0),
        c(-0.7, 0.0),
        c(0.9, 0.0),
        c(-0.2, 0.0),
        c(0.4, 0.0),
        c(-0.6, 0.0),
    ];
    let time_energy: f32 = original.iter().map(|x| x.norm_sqr()).sum();

    let mut buf = original;
    fft(&mut buf).ok();
    let freq_energy: f32 = buf.iter().map(|x| x.norm_sqr()).sum();

    let n = original.len() as f32;
    assert!(
        (time_energy - freq_energy / n).abs() < 1e-3,
        "Parseval violation: time={time_energy}, freq/N={}",
        freq_energy / n
    );
}

#[test]
fn known_4point_result() {
    let mut buf = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
    fft(&mut buf).ok();

    assert!((buf[0].re - 10.0).abs() < 1e-4);
    assert!((buf[1].re - (-2.0)).abs() < 1e-4);
    assert!((buf[1].im - 2.0).abs() < 1e-4);
    assert!((buf[2].re - (-2.0)).abs() < 1e-4);
    assert!(buf[2].im.abs() < 1e-4);
    assert!((buf[3].re - (-2.0)).abs() < 1e-4);
    assert!((buf[3].im - (-2.0)).abs() < 1e-4);
}

#[test]
fn fft_error_display_format() {
    let e1 = format!("{}", FftError::Empty);
    let e2 = format!("{}", FftError::NotPowerOfTwo(7));
    assert_eq!(e1, "FFT input is empty");
    assert_eq!(e2, "FFT length 7 is not a power of two");
}

#[test]
fn roundtrip_large() {
    let mut buf: [Complex<f32>; 128] = [c(0.0, 0.0); 128];
    for (i, x) in buf.iter_mut().enumerate() {
        x.re = (i as f32) / 128.0;
    }
    let original = buf;
    fft(&mut buf).ok();
    ifft(&mut buf).ok();
    for (a, b) in buf.iter().zip(original.iter()) {
        assert!((a - b).norm() < 1e-3, "mismatch: {a} vs {b}");
    }
}
