//! End-to-end throughput test for a 3-node pipeline at 44100 Hz.
//!
//! Processes 1 second of mono audio through gain → lowpass → gain and
//! verifies correctness + measures wall-clock time.

use resonant_filters::design::butterworth_lowpass;
use resonant_filters::Biquad;
use resonant_stream::nodes::{FilterNode, GainNode};
use resonant_stream::{Chunk, Pipeline};

const SAMPLE_RATE: u32 = 44100;

#[test]
fn one_second_three_node_pipeline() {
    let coeffs = butterworth_lowpass(4000.0, f64::from(SAMPLE_RATE)).unwrap();

    let mut pipeline = Pipeline::builder()
        .sample_rate(SAMPLE_RATE)
        .channels(1)
        .node(GainNode::new(0.5))
        .node(FilterNode::new(Biquad::new(coeffs)))
        .node(GainNode::new(2.0))
        .build();

    assert_eq!(pipeline.len(), 3);

    // 1 second of DC at unity
    let samples = vec![1.0_f32; SAMPLE_RATE as usize];
    let chunk = Chunk::new(samples, SAMPLE_RATE, 1);

    let start = std::time::Instant::now();
    let out = pipeline.process(chunk).unwrap();
    let elapsed = start.elapsed();

    // Output should have same length
    assert_eq!(out.len(), SAMPLE_RATE as usize);
    assert_eq!(out.sample_rate(), SAMPLE_RATE);
    assert_eq!(out.channels(), 1);

    // DC should pass through the lowpass near-unity after transient settles.
    // Gain chain: 0.5 * 1.0 (filter ~unity for DC) * 2.0 = ~1.0
    let tail = &out.data()[out.len() - 100..];
    let avg: f32 = tail.iter().sum::<f32>() / tail.len() as f32;
    assert!(
        (avg - 1.0).abs() < 0.02,
        "DC pass-through expected ~1.0, got {avg}"
    );

    // Should complete well under real-time (1 second of audio)
    // On any modern machine this should take < 10ms
    assert!(
        elapsed.as_millis() < 1000,
        "Pipeline took {elapsed:?} for 1s of audio — slower than real-time"
    );

    eprintln!(
        "Pipeline throughput: {SAMPLE_RATE} samples in {elapsed:?} ({:.1}x real-time)",
        1.0 / elapsed.as_secs_f64()
    );
}

#[test]
fn ten_seconds_chunked_processing() {
    let coeffs = butterworth_lowpass(4000.0, f64::from(SAMPLE_RATE)).unwrap();

    let mut pipeline = Pipeline::builder()
        .sample_rate(SAMPLE_RATE)
        .channels(1)
        .node(GainNode::new(0.8))
        .node(FilterNode::new(Biquad::new(coeffs)))
        .node(GainNode::new(1.0))
        .build();

    let chunk_size = 1024_usize;
    let total_samples = SAMPLE_RATE as usize * 10;
    let num_chunks = total_samples.div_ceil(chunk_size);

    let start = std::time::Instant::now();
    let mut total_output = 0_usize;

    for _ in 0..num_chunks {
        let samples = vec![0.5_f32; chunk_size];
        let chunk = Chunk::new(samples, SAMPLE_RATE, 1);
        let out = pipeline.process(chunk).unwrap();
        total_output += out.len();
    }

    let elapsed = start.elapsed();

    assert_eq!(total_output, num_chunks * chunk_size);

    eprintln!(
        "Chunked throughput: {} samples in {elapsed:?} ({:.1}x real-time)",
        total_output,
        10.0 / elapsed.as_secs_f64()
    );
}
