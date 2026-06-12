use resonant_core::signal::Signal;
use resonant_fft::stft::Stft;
use resonant_fft::FftError;

#[test]
fn analyze_frame_count() {
    let sig = Signal::from_samples(vec![0.0_f32; 512]);
    let stft = Stft::builder(128, 64).build();
    let frames = stft.analyze(&sig).unwrap();
    // (512 - 128) / 64 + 1 = 7
    assert_eq!(frames.len(), 7);
}

#[test]
fn roundtrip_no_window() {
    let original: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
    let sig = Signal::from_samples(original.clone());
    let stft = Stft::builder(32, 32).build();
    let frames = stft.analyze(&sig).unwrap();
    let reconstructed = stft.synthesize(&frames).unwrap();
    for (a, b) in reconstructed.data().iter().zip(original.iter()) {
        assert!((a - b).abs() < 1e-3, "mismatch: {a} vs {b}");
    }
}

#[test]
fn analyze_empty_signal() {
    let sig = Signal::from_samples(Vec::<f32>::new());
    let stft = Stft::builder(16, 8).build();
    assert_eq!(stft.analyze(&sig), Err(FftError::Empty));
}

#[test]
fn synthesize_empty() {
    let stft = Stft::builder(16, 8).build();
    let result = stft.synthesize(&[]).unwrap();
    assert!(result.data().is_empty());
}

#[test]
fn with_hann_window() {
    let sig = Signal::from_samples(vec![1.0_f32; 128]);
    let stft = Stft::builder(32, 16)
        .window_fn(resonant_core::window::hann)
        .build();
    let frames = stft.analyze(&sig).unwrap();
    assert!(!frames.is_empty());
    // Frames should be windowed — DC magnitude should be less than
    // it would be without windowing (32.0 for rectangular)
    let dc_mag = frames[0].data()[0].norm();
    assert!(dc_mag < 32.0);
}

#[test]
fn accessors_match_config() {
    let stft = Stft::builder(1024, 256).build();
    assert_eq!(stft.window_size(), 1024);
    assert_eq!(stft.hop_size(), 256);
}

#[test]
fn roundtrip_hann_50pct_overlap() {
    // Hann + 50% hop: the per-sample squared-window normalization pass must
    // correct the amplitude. Without it, output ≈ 0.25 × original.
    // Sample 0 and the last sample land on the zero-valued Hann endpoint of a
    // single frame and cannot be reconstructed — only interior samples are checked.
    let original: Vec<f32> = (0..128)
        .map(|i| (i as f32 / 128.0 * 2.0 * core::f32::consts::PI).sin())
        .collect();
    let sig = Signal::from_samples(original.clone());
    let stft = Stft::builder(32, 16)
        .window_fn(resonant_core::window::hann)
        .build();
    let frames = stft.analyze(&sig).unwrap();
    let out = stft.synthesize(&frames).unwrap();
    let out_len = out.data().len();
    for (a, b) in out.data()[1..out_len - 1]
        .iter()
        .zip(original[1..out_len - 1].iter())
    {
        assert!((a - b).abs() < 1e-3, "mismatch: got {a}, expected {b}");
    }
}
