use resonant_core::signal::Signal;
use resonant_fft::{FftError, SignalFftExt, SignalFreqExt, SignalIfftExt};

#[test]
fn fft_ifft_roundtrip_vec() {
    let original = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let sig = Signal::from_samples(original.clone());
    let freq = sig.fft().unwrap();
    let time = freq.ifft().unwrap();
    for (a, b) in time.data().iter().zip(original.iter()) {
        assert!((a - b).abs() < 1e-3, "mismatch: {a} vs {b}");
    }
}

#[test]
fn fft_ifft_roundtrip_array() {
    let sig = Signal::from_samples([0.5_f32, -0.5, 0.25, -0.25]);
    let freq = sig.fft().unwrap();
    let time = freq.ifft().unwrap();
    let expected = [0.5, -0.5, 0.25, -0.25];
    for (a, b) in time.data().iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-3);
    }
}

#[test]
fn fft_empty_signal_returns_error() {
    let sig = Signal::from_samples(Vec::<f32>::new());
    assert_eq!(sig.fft(), Err(FftError::Empty));
}

#[test]
fn magnitude_of_pure_cosine() {
    // 8-point cosine at bin 1: cos(2*pi*n/8) for n=0..7
    let n = 8;
    let samples: Vec<f32> = (0..n)
        .map(|i| (2.0 * core::f32::consts::PI * i as f32 / n as f32).cos())
        .collect();
    let sig = Signal::from_samples(samples);
    let freq = sig.fft().unwrap();
    let mags = freq.magnitude();
    // Bin 1 and bin 7 (mirror) should have magnitude N/2 = 4.0
    assert!((mags[1] - 4.0).abs() < 1e-2);
    assert!((mags[7] - 4.0).abs() < 1e-2);
    // DC and other bins ≈ 0
    assert!(mags[0] < 1e-2);
}

#[test]
fn magnitude_squared_matches_magnitude() {
    let sig = Signal::from_samples(vec![1.0_f32, 2.0, 3.0, 4.0]);
    let freq = sig.fft().unwrap();
    let mags = freq.magnitude();
    let mags_sq = freq.magnitude_squared();
    for (m, msq) in mags.iter().zip(mags_sq.iter()) {
        assert!((m * m - msq).abs() < 1e-3);
    }
}

#[test]
fn phase_real_positive_is_zero() {
    let sig = Signal::from_samples(vec![1.0_f32; 4]);
    let freq = sig.fft().unwrap();
    let phases = freq.phase();
    assert!(phases[0].abs() < 1e-4);
}

#[test]
fn fft_with_slice_ref() {
    let data = [1.0_f32, 0.0, -1.0, 0.0];
    let sig = Signal::<&[f32], resonant_core::signal::TimeDomain>::new(&data[..]);
    let freq = sig.fft().unwrap();
    assert_eq!(freq.data().len(), 4);
}
