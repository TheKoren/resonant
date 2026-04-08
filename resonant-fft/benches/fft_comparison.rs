use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use resonant_core::signal::Signal;
use resonant_fft::{SignalFftExt, SignalRfftExt};

/// Compare radix-2 complex FFT, real FFT, and rustfft at power-of-two sizes.
fn bench_fft_comparison(c: &mut Criterion) {
    let sizes: &[usize] = &[64, 256, 1024, 4096, 16384, 65536];
    let mut group = c.benchmark_group("fft_comparison");

    for &n in sizes {
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();

        group.throughput(Throughput::Elements(n as u64));

        // rfft: N real → N/2+1 complex bins (~40% less work than complex FFT)
        group.bench_with_input(BenchmarkId::new("rfft", n), &n, |b, _| {
            b.iter(|| {
                let sig = Signal::from_samples(samples.clone());
                sig.rfft().unwrap()
            });
        });

        // Full complex FFT (rustfft backend if available, else radix-2)
        group.bench_with_input(BenchmarkId::new("complex_fft", n), &n, |b, _| {
            b.iter(|| {
                let sig = Signal::from_samples(samples.clone());
                sig.fft().unwrap()
            });
        });
    }

    group.finish();
}

/// Throughput in Msamples/sec for rfft at each size.
fn bench_rfft_throughput(c: &mut Criterion) {
    let sizes: &[usize] = &[512, 2048, 8192, 32768];
    let mut group = c.benchmark_group("rfft_throughput");

    for &n in sizes {
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos()).collect();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("rfft", n), &n, |b, _| {
            b.iter(|| {
                let sig = Signal::from_samples(samples.clone());
                sig.rfft().unwrap()
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_fft_comparison, bench_rfft_throughput);
criterion_main!(benches);
