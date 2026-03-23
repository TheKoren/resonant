//! Spectrogram — visualise time × frequency as a heatmap PNG.
//!
//! Usage:
//!   cargo run -p resonant --example spectrogram -- assets/test.wav
//!
//! Outputs `spectrogram.png` in the current directory.

use plotters::prelude::*;
use resonant::AudioFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/test.wav".to_string());

    let audio = AudioFile::open(&path)?;
    let sample_rate = audio.sample_rate();
    let duration = audio.duration_secs() as f32;

    println!(
        "Loaded: {} Hz, {:.2}s ({} samples)",
        sample_rate,
        duration,
        audio.num_frames(),
    );

    let window_size = 2048_usize;
    let overlap = 0.75_f32;
    let hop_size = ((1.0 - overlap) * window_size as f32) as usize;

    // Collect all frames as magnitude vectors
    let frames: Vec<Vec<f32>> = audio
        .with_window_size(window_size)
        .with_overlap(overlap)
        .fft_stream()?
        .filter_map(|r| r.ok())
        .map(|bins| bins.iter().map(|b| b.magnitude).collect())
        .collect();

    if frames.is_empty() {
        println!("Audio too short for spectrogram.");
        return Ok(());
    }

    let num_frames = frames.len();
    let num_bins = frames[0].len();
    let max_freq = sample_rate as f32 / 2.0;

    // Clamp display to 8 kHz for readability
    let display_freq = max_freq.min(8000.0);
    let display_bins = ((display_freq / max_freq) * num_bins as f32) as usize;

    // Convert to dB, find range for colour mapping
    let db_frames: Vec<Vec<f32>> = frames
        .iter()
        .map(|f| {
            f[..display_bins]
                .iter()
                .map(|&m| 20.0 * m.max(1e-10).log10())
                .collect()
        })
        .collect();

    let db_min = db_frames
        .iter()
        .flat_map(|f| f.iter())
        .cloned()
        .fold(f32::INFINITY, f32::min)
        .max(-120.0);
    let db_max = db_frames
        .iter()
        .flat_map(|f| f.iter())
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let db_range = (db_max - db_min).max(1.0);

    println!(
        "Spectrogram: {} frames × {} bins, dB range: {:.0} to {:.0}",
        num_frames, display_bins, db_min, db_max
    );

    // Render
    let out_path = "spectrogram.png";
    let width = 1200_u32;
    let height = 600_u32;

    let root = BitMapBackend::new(out_path, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Spectrogram — {path}"),
            ("sans-serif", 20).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0_f32..duration, 0.0_f32..display_freq)?;

    chart
        .configure_mesh()
        .x_desc("Time (s)")
        .y_desc("Frequency (Hz)")
        .draw()?;

    // Draw each cell as a small rectangle
    let time_step = hop_size as f32 / sample_rate as f32;
    let freq_step = display_freq / display_bins as f32;

    for (fi, frame_db) in db_frames.iter().enumerate() {
        let t0 = fi as f32 * time_step;
        let t1 = t0 + time_step;
        for (bi, &db) in frame_db.iter().enumerate() {
            let f0 = bi as f32 * freq_step;
            let f1 = f0 + freq_step;

            // Map dB to 0.0–1.0 intensity
            let intensity = ((db - db_min) / db_range).clamp(0.0, 1.0);

            // Viridis-like: dark purple → blue → green → yellow
            let r = (255.0 * (intensity * 2.0 - 0.5).clamp(0.0, 1.0)) as u8;
            let g = (255.0
                * (intensity * 1.5 - 0.2)
                    .clamp(0.0, 1.0)
                    .min(1.0 - (intensity - 0.8).max(0.0) * 5.0)) as u8;
            let b_val = (255.0 * (1.0 - intensity * 1.2).clamp(0.0, 1.0)) as u8;
            let color = RGBColor(r, g, b_val);

            chart.draw_series(std::iter::once(Rectangle::new(
                [(t0, f0), (t1, f1)],
                color.filled(),
            )))?;
        }
    }

    root.present()?;
    println!("Saved to {out_path}");

    Ok(())
}
