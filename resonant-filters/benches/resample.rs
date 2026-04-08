use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use resonant_filters::resample::{decimate, PolyphaseResampler};

const SR: f64 = 44100.0;

/// Compare polyphase resampler against simple integer decimation.
fn bench_decimate_vs_polyphase(c: &mut Criterion) {
    let n = 44100; // 1 second of audio at 44.1 kHz
    let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut group = c.benchmark_group("resample");

    // ── Integer decimation (existing) ────────────────────────────────────────
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("decimate_2x", |b| {
        b.iter(|| decimate(&input, 2, SR).unwrap());
    });
    group.bench_function("decimate_4x", |b| {
        b.iter(|| decimate(&input, 4, SR).unwrap());
    });

    // ── Polyphase — common ratios ─────────────────────────────────────────────
    group.bench_function("polyphase_2x_down", |b| {
        let mut r = PolyphaseResampler::new(1, 2).unwrap();
        b.iter(|| {
            r.reset();
            r.process(&input)
        });
    });

    group.bench_function("polyphase_2x_up", |b| {
        let mut r = PolyphaseResampler::new(2, 1).unwrap();
        b.iter(|| {
            r.reset();
            r.process(&input)
        });
    });

    // 44100 → 48000 (up=160, down=147): real-world CD→DAC conversion
    group.bench_function("polyphase_44100_to_48000", |b| {
        let mut r = PolyphaseResampler::new(160, 147).unwrap();
        b.iter(|| {
            r.reset();
            r.process(&input)
        });
    });

    // 44100 → 16000 (up=160, down=441): downsampling for speech processing
    group.bench_function("polyphase_44100_to_16000", |b| {
        let mut r = PolyphaseResampler::new(160, 441).unwrap();
        b.iter(|| {
            r.reset();
            r.process(&input)
        });
    });

    group.finish();
}

/// Throughput across block sizes for the 44100→48000 ratio.
fn bench_polyphase_block_sizes(c: &mut Criterion) {
    let sizes: &[usize] = &[64, 256, 1024, 4096, 16384];
    let mut group = c.benchmark_group("polyphase_block_sizes");

    for &n in sizes {
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("44100_to_48000", n), &n, |b, _| {
            let mut r = PolyphaseResampler::new(160, 147).unwrap();
            b.iter(|| {
                r.reset();
                r.process(&input)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_decimate_vs_polyphase,
    bench_polyphase_block_sizes
);
criterion_main!(benches);
