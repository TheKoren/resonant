use resonant_filters::resample::{self, PolyphaseResampler};

const SR: f64 = 48000.0;

#[test]
fn decimate_by_2_halves_length() {
    let input = vec![0.0_f32; 1000];
    let out = resample::decimate(&input, 2, SR).unwrap();
    assert_eq!(out.len(), 500);
}

#[test]
fn decimate_preserves_dc() {
    let input = vec![1.0_f32; 4000];
    let out = resample::decimate(&input, 4, SR).unwrap();
    let last = *out.last().unwrap();
    assert!((last - 1.0).abs() < 0.01, "DC = {last}");
}

#[test]
fn decimate_rejects_alias_frequency() {
    // 18 kHz tone decimated by 3 → output rate 16 kHz → Nyquist 8 kHz
    // 18 kHz is above output Nyquist, should be suppressed
    let n = 12000;
    let freq = 18000.0_f32;
    let input: Vec<f32> = (0..n)
        .map(|i| (2.0 * core::f32::consts::PI * freq * i as f32 / SR as f32).sin())
        .collect();

    let out = resample::decimate(&input, 3, SR).unwrap();
    let tail = &out[out.len() / 2..];
    let peak: f32 = tail.iter().map(|x| x.abs()).fold(0.0, f32::max);
    assert!(peak < 0.1, "Alias freq peak = {peak}");
}

#[test]
fn decimate_passes_low_tone() {
    // 200 Hz tone decimated by 2 → output rate 24 kHz
    // 200 Hz is well below Nyquist, should pass
    let n = 8000;
    let freq = 200.0_f32;
    let input: Vec<f32> = (0..n)
        .map(|i| (2.0 * core::f32::consts::PI * freq * i as f32 / SR as f32).sin())
        .collect();

    let out = resample::decimate(&input, 2, SR).unwrap();
    let tail = &out[out.len() / 2..];
    let peak: f32 = tail.iter().map(|x| x.abs()).fold(0.0, f32::max);
    assert!(peak > 0.85, "Low-freq peak = {peak}");
}

#[test]
fn decimate_invalid_inputs() {
    assert!(resample::decimate(&[1.0], 1, SR).is_none());
    assert!(resample::decimate(&[], 2, SR).is_none());
    assert!(resample::decimate(&[1.0], 2, 0.0).is_none());
}

#[test]
fn polyphase_upsample_output_length() {
    let mut r = PolyphaseResampler::new(2, 1).unwrap();
    let out = r.process(&vec![0.0_f32; 200]);
    assert_eq!(out.len(), 400);
}

#[test]
fn polyphase_rational_output_length_44100_to_48000() {
    // 44100 → 48000 Hz: exact ratio 160/147
    let mut r = PolyphaseResampler::new(160, 147).unwrap();
    let input = vec![0.0_f32; 44100];
    let out = r.process(&input);
    let expected = 44100 * 160 / 147; // = 48000
    assert!((out.len() as i64 - expected as i64).abs() <= 2);
}

#[test]
fn polyphase_streaming_matches_batch() {
    // Processing in two chunks must match a single batch call
    let mut r_batch = PolyphaseResampler::new(3, 2).unwrap();
    let mut r_stream = PolyphaseResampler::new(3, 2).unwrap();

    let full: Vec<f32> = (0..200).map(|i| (i as f32 * 0.05).sin()).collect();
    let batch = r_batch.process(&full);

    let mut streamed = r_stream.process(&full[..100]);
    streamed.extend(r_stream.process(&full[100..]));

    assert_eq!(batch.len(), streamed.len());
    for (a, b) in batch.iter().zip(streamed.iter()) {
        assert!((a - b).abs() < 1e-6, "streaming mismatch: {a} vs {b}");
    }
}
