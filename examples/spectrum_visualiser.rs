//! Spectrum visualiser — find the dominant frequencies in an audio file.
//!
//! Usage:
//!   cargo run -p resonant --example spectrum_visualiser -- assets/test.wav

use resonant::{AudioError, AudioFile};

fn main() -> Result<(), AudioError> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/test.wav".to_string());

    let audio = AudioFile::open(&path)?;
    println!(
        "Loaded: {} Hz, {} ch, {:.2}s ({} samples)",
        audio.sample_rate(),
        audio.channels(),
        audio.duration_secs(),
        audio.num_frames(),
    );

    // Run a 4096-point FFT with Hann window
    let bins = audio.with_window_size(4096).fft()?;

    // Find the top 10 peaks by magnitude
    let mut indexed: Vec<_> = bins.iter().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| {
        b.magnitude
            .partial_cmp(&a.magnitude)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("\nTop 10 frequency peaks:");
    println!("{:<6} {:>10} {:>10}", "Rank", "Freq (Hz)", "Magnitude");
    println!("{}", "-".repeat(30));
    for (rank, (_, bin)) in indexed.iter().take(10).enumerate() {
        println!(
            "{:<6} {:>10.1} {:>10.2}",
            rank + 1,
            bin.frequency_hz,
            bin.magnitude,
        );
    }

    Ok(())
}
