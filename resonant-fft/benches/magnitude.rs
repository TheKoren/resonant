use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use resonant_core::signal::{FreqDomain, Signal};
use resonant_fft::{Complex, SignalFreqExt};

fn bench_magnitude(c: &mut Criterion) {
    let sizes: &[usize] = &[1024, 4096];

    let mut group = c.benchmark_group("magnitude");
    for &size in sizes {
        let data: Vec<Complex<f32>> = (0..size)
            .map(|i| Complex::new(i as f32 * 0.01, (i as f32 * 0.02).sin()))
            .collect();
        let sig = Signal::<_, FreqDomain>::new(data);

        group.bench_with_input(BenchmarkId::new("magnitude", size), &size, |b, &_| {
            b.iter(|| sig.magnitude());
        });

        group.bench_with_input(
            BenchmarkId::new("magnitude_squared", size),
            &size,
            |b, &_| {
                b.iter(|| sig.magnitude_squared());
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_magnitude);
criterion_main!(benches);
