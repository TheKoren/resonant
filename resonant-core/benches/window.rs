use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use resonant_core::window;

fn bench_window(c: &mut Criterion) {
    let sizes: &[usize] = &[1024, 4096, 16384];

    let mut group = c.benchmark_group("window");
    for &size in sizes {
        let mut buf = vec![1.0_f32; size];

        group.bench_with_input(BenchmarkId::new("hann", size), &size, |b, &_| {
            b.iter(|| {
                buf.fill(1.0);
                window::hann(&mut buf);
            });
        });

        group.bench_with_input(BenchmarkId::new("hamming", size), &size, |b, &_| {
            b.iter(|| {
                buf.fill(1.0);
                window::hamming(&mut buf);
            });
        });

        group.bench_with_input(BenchmarkId::new("blackman", size), &size, |b, &_| {
            b.iter(|| {
                buf.fill(1.0);
                window::blackman(&mut buf);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_window);
criterion_main!(benches);
