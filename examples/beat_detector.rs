//! Beat detector stub — compute spectral flux per STFT frame.
//!
//! This is a simplified onset-strength estimator. A full beat detector
//!
//! Usage:
//!   cargo run -p resonant --example beat_detector -- assets/test.wav

use resonant::{AudioError, AudioFile};

fn main() -> Result<(), AudioError> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/test.wav".to_string());

    let audio = AudioFile::open(&path)?;
    let duration = audio.duration_secs();
    let sample_rate = audio.sample_rate() as f32;

    println!("Loaded: {} Hz, {:.2}s", audio.sample_rate(), duration,);

    let window_size = 1024_usize;
    let hop_size = 512_usize; // 50% overlap

    let frames: Vec<Vec<f32>> = audio
        .with_window_size(window_size)
        .fft_stream()?
        .filter_map(|r| r.ok())
        .map(|bins| bins.iter().map(|b| b.magnitude).collect())
        .collect();

    if frames.len() < 2 {
        println!("Audio too short for onset detection.");
        return Ok(());
    }

    // Spectral flux: sum of positive magnitude differences between frames
    let mut flux: Vec<f32> = Vec::with_capacity(frames.len() - 1);
    for i in 1..frames.len() {
        let sf: f32 = frames[i]
            .iter()
            .zip(frames[i - 1].iter())
            .map(|(curr, prev)| (curr - prev).max(0.0))
            .sum();
        flux.push(sf);
    }

    // Simple threshold: mean + 1.5 * std_dev
    let mean = flux.iter().sum::<f32>() / flux.len() as f32;
    let variance = flux.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / flux.len() as f32;
    let std_dev = variance.sqrt();
    let threshold = mean + 1.5 * std_dev;

    let secs_per_frame = hop_size as f32 / sample_rate;

    println!("\nOnset detection (spectral flux > {threshold:.1}):");
    println!("{:<10} {:>10} {:>10}", "Frame", "Time (s)", "Flux");
    println!("{}", "-".repeat(34));

    let mut onset_count = 0;
    for (i, &sf) in flux.iter().enumerate() {
        if sf > threshold {
            let time = (i + 1) as f32 * secs_per_frame;
            println!("{:<10} {:>10.3} {:>10.1}", i + 1, time, sf);
            onset_count += 1;
        }
    }

    println!("\nDetected {} onsets in {:.2}s", onset_count, duration);

    Ok(())
}
