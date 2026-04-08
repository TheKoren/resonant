use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use resonant::AudioFile;

const SR: u32 = 44100;
const SECONDS: usize = 10;
const N: usize = SR as usize * SECONDS;

fn make_mono_samples() -> Vec<f32> {
    (0..N).map(|i| (i as f32 * 0.01).sin()).collect()
}

/// End-to-end: AudioFile::from_samples → FFT.
///
/// Exercises the full path: mono normalisation, windowing, and the FFT call.
fn bench_audio_file_fft(c: &mut Criterion) {
    let samples = make_mono_samples();
    let mut group = c.benchmark_group("facade");
    group.throughput(Throughput::Elements(N as u64));
    // FFT over a 10-second buffer is expensive per iteration; use fewer samples.
    group.sample_size(10);

    group.bench_function("audio_file_fft_default", |b| {
        b.iter(|| {
            let file = AudioFile::from_samples(samples.clone(), SR, 1);
            criterion::black_box(file.fft().unwrap())
        });
    });

    group.bench_function("audio_file_fft_4096", |b| {
        b.iter(|| {
            let file = AudioFile::from_samples(samples.clone(), SR, 1)
                .with_window_size(4096);
            criterion::black_box(file.fft().unwrap())
        });
    });

    group.finish();
}

/// Streaming frame iterator throughput.
fn bench_fft_stream(c: &mut Criterion) {
    let samples = make_mono_samples();
    let mut group = c.benchmark_group("fft_stream");
    group.throughput(Throughput::Elements(N as u64));
    group.sample_size(10);

    group.bench_function("consume_all_frames_1024", |b| {
        b.iter(|| {
            let file = AudioFile::from_samples(samples.clone(), SR, 1)
                .with_window_size(1024);
            let count = file.fft_stream().unwrap().count();
            criterion::black_box(count)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_audio_file_fft, bench_fft_stream);
criterion_main!(benches);
