use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use resonant_core::signal::Signal;
use resonant_fft::{Complex, SignalFftExt, SignalRfftExt};

fn bench_rfft_vs_complex(c: &mut Criterion) {
    let sizes: &[usize] = &[1024, 4096, 16384];

    let mut group = c.benchmark_group("rfft_vs_complex_fft");

    for &n in sizes {
        group.throughput(Throughput::Elements(n as u64));

        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();

        // Real-valued FFT via SignalRfftExt — produces N/2+1 bins
        group.bench_with_input(BenchmarkId::new("rfft", n), &n, |b, _| {
            b.iter(|| {
                let sig = Signal::from_samples(samples.clone());
                sig.rfft().unwrap()
            });
        });

        // Full complex FFT via SignalFftExt — produces N bins
        group.bench_with_input(BenchmarkId::new("complex_fft", n), &n, |b, _| {
            b.iter(|| {
                let sig = Signal::from_samples(samples.clone());
                sig.fft().unwrap()
            });
        });
    }

    group.finish();
}

fn bench_rfft_magnitude(c: &mut Criterion) {
    // Verify that magnitude() on N/2+1 rfft output follows the same hot path
    // as on full-spectrum FFT output (both are Vec<Complex<f32>> FreqDomain signals).
    use resonant_core::signal::FreqDomain;
    use resonant_fft::SignalFreqExt;

    let sizes: &[usize] = &[1024, 4096];

    let mut group = c.benchmark_group("rfft_magnitude");

    for &n in sizes {
        let m = n / 2 + 1;
        group.throughput(Throughput::Elements(m as u64));

        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let rfft_sig = Signal::from_samples(samples).rfft().unwrap();

        // Standalone N/2+1 bin signal for magnitude benchmarks
        let half_spectrum: Vec<Complex<f32>> = (0..m)
            .map(|i| Complex::new(i as f32 * 0.01, (i as f32 * 0.02).sin()))
            .collect();
        let half_sig = Signal::<_, FreqDomain>::new(half_spectrum);

        group.bench_with_input(BenchmarkId::new("rfft_then_magnitude", n), &n, |b, _| {
            b.iter(|| rfft_sig.magnitude());
        });

        group.bench_with_input(BenchmarkId::new("half_spectrum_magnitude", m), &m, |b, _| {
            b.iter(|| half_sig.magnitude());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_rfft_vs_complex, bench_rfft_magnitude);
criterion_main!(benches);
