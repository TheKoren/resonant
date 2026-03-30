use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use resonant_filters::fir::Fir;

fn bench_fir(c: &mut Criterion) {
    let tap_counts: &[usize] = &[32, 128, 512];
    let buf_len = 4096;

    let mut group = c.benchmark_group("fir");
    for &taps in tap_counts {
        let coeffs = vec![1.0 / taps as f32; taps];
        let mut buf = vec![0.5_f32; buf_len];

        group.bench_with_input(
            BenchmarkId::new("process_buf", taps),
            &taps,
            |b, &_| {
                let mut filter = Fir::new(coeffs.clone());
                b.iter(|| {
                    buf.fill(0.5);
                    filter.process_buf(&mut buf);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_fir);
criterion_main!(benches);
