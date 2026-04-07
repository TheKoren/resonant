//! Prints the frequency response of a Butterworth lowpass filter as an ASCII table.
//!
//! Run with:
//!
//! ```text
//! cargo run --example filter_response_plot
//! ```

use resonant_filters::{design, response::FilterResponseExt};

fn main() {
    let sample_rate = 44100.0_f64;
    let cutoff_hz = 1000.0_f64;
    let n_points = 32;

    let coeffs = design::butterworth_lowpass(cutoff_hz, sample_rate)
        .expect("valid cutoff and sample rate");

    let resp = coeffs
        .frequency_response(n_points, sample_rate as f32)
        .expect("valid parameters");

    let db = resp.magnitudes_db();

    println!(
        "Butterworth lowpass — cutoff {cutoff_hz:.0} Hz / {sample_rate:.0} Hz sample rate"
    );
    println!("{:<10}  {:>10}  {:>10}  {}", "Freq (Hz)", "Mag (lin)", "Mag (dB)", "Bar");
    println!("{}", "-".repeat(60));

    for i in 0..n_points {
        let freq = resp.frequencies[i];
        let mag = resp.magnitudes[i];
        let d = db[i];

        // ASCII bar: scale 0..1 to 0..30 chars
        let bar_len = (mag.min(1.0) * 30.0).round() as usize;
        let bar = "#".repeat(bar_len);

        println!("{:<10.1}  {:>10.4}  {:>10.2}  {}", freq, mag, d, bar);
    }
}
