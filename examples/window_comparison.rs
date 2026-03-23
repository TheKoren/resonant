//! Window comparison — plot the magnitude spectrum of a sine wave under
//! each window function, showing the leakage trade-off.
//!
//! Usage:
//!   cargo run -p resonant --example window_comparison
//!
//! Outputs `window_comparison.png` in the current directory.

use plotters::prelude::*;
use resonant::{window, AudioFile};

type WindowEntry<'a> = (&'a str, fn(&mut [f32]), RGBColor);
type SpectrumEntry<'a> = (&'a str, Vec<(f32, f32)>, RGBColor);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sample_rate = 44100_u32;
    let n = 4096_usize;
    // 1000 Hz sine — deliberately NOT bin-centred to show leakage
    let freq_hz = 1000.0_f32;

    let mut samples = vec![0.0_f32; n];
    for (i, s) in samples.iter_mut().enumerate() {
        *s = (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate as f32).sin();
    }

    let windows: Vec<WindowEntry> = vec![
        ("Rectangular", window::rectangular, RGBColor(220, 50, 50)),
        ("Hann", window::hann, RGBColor(50, 120, 220)),
        ("Hamming", window::hamming, RGBColor(50, 180, 80)),
        ("Blackman", window::blackman, RGBColor(160, 50, 200)),
        ("Bartlett", window::bartlett, RGBColor(220, 150, 30)),
    ];

    // For each window, build an AudioFile and run FFT, collect dB spectrum
    let bin_width = sample_rate as f32 / n as f32;
    let num_bins = n / 2 + 1;
    let freqs: Vec<f32> = (0..num_bins).map(|k| k as f32 * bin_width).collect();

    // Focus on 500–1500 Hz to see the main lobe and side lobes clearly
    let freq_lo = 500.0_f32;
    let freq_hi = 1500.0_f32;
    let display_range = freqs
        .iter()
        .enumerate()
        .filter(|(_, &f)| f >= freq_lo && f <= freq_hi)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();

    let mut spectra: Vec<SpectrumEntry> = Vec::new();

    for (name, wfn, color) in &windows {
        let audio = AudioFile::from_samples(samples.clone(), sample_rate, 1).with_window_fn(*wfn);

        let bins = audio.fft()?;

        // Normalise to peak = 0 dB for each window so we compare shapes
        let peak_mag = bins.iter().map(|b| b.magnitude).fold(0.0_f32, f32::max);
        let peak_db = 20.0 * peak_mag.max(1e-10).log10();

        let points: Vec<(f32, f32)> = display_range
            .iter()
            .map(|&i| {
                let db = 20.0 * bins[i].magnitude.max(1e-10).log10() - peak_db;
                (freqs[i], db)
            })
            .collect();

        spectra.push((name, points, *color));
    }

    // Render
    let out_path = "window_comparison.png";
    let root = BitMapBackend::new(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let db_lo = -100.0_f32;
    let db_hi = 5.0_f32;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Window comparison — {freq_hz:.0} Hz sine, {n}-point FFT"),
            ("sans-serif", 22).into_font(),
        )
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(freq_lo..freq_hi, db_lo..db_hi)?;

    chart
        .configure_mesh()
        .x_desc("Frequency (Hz)")
        .y_desc("Magnitude (dB, normalised)")
        .y_label_formatter(&|y| format!("{y:.0}"))
        .draw()?;

    for (name, points, color) in &spectra {
        chart
            .draw_series(LineSeries::new(
                points.iter().copied(),
                color.stroke_width(2),
            ))?
            .label(*name)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
            });
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    root.present()?;
    println!("Saved to {out_path}");
    println!("\nWhat to look for:");
    println!("  - Rectangular: narrowest main lobe, but highest side lobes (most leakage)");
    println!("  - Blackman: widest main lobe, but side lobes drop fastest (least leakage)");
    println!("  - Hann/Hamming: good middle ground for most applications");

    Ok(())
}
