use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use resonant_filters::nonlinear::{MoogLadder, SaturatingBiquad, StateVariableFilter};
use resonant_filters::{design, Biquad};

const SR: f32 = 44100.0;
const BLOCK: usize = 4096;

fn make_input() -> Vec<f32> {
    (0..BLOCK).map(|i| (i as f32 * 0.1).sin()).collect()
}

/// Compare linear biquad against saturating biquad at various drive levels.
fn bench_biquad_vs_saturating(c: &mut Criterion) {
    let coeffs = design::butterworth_lowpass(1000.0, SR as f64).expect("valid design params");
    let input = make_input();
    let mut group = c.benchmark_group("biquad_vs_saturating");
    group.throughput(Throughput::Elements(BLOCK as u64));

    group.bench_function("biquad_linear", |b| {
        let mut f = Biquad::new(coeffs);
        b.iter(|| {
            for &x in &input {
                criterion::black_box(f.process_sample(x));
            }
        });
    });

    for drive in [0.0_f32, 0.3, 0.7, 1.0] {
        group.bench_with_input(
            BenchmarkId::new("saturating_biquad", drive),
            &drive,
            |b, &drv| {
                let mut f = SaturatingBiquad::new(coeffs, drv);
                b.iter(|| {
                    for &x in &input {
                        criterion::black_box(f.process_sample(x));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Moog ladder at various resonance settings.
fn bench_moog_ladder(c: &mut Criterion) {
    let input = make_input();
    let mut group = c.benchmark_group("moog_ladder");
    group.throughput(Throughput::Elements(BLOCK as u64));

    for resonance in [0.0_f32, 1.0, 2.0, 3.5] {
        group.bench_with_input(
            BenchmarkId::new("resonance", resonance),
            &resonance,
            |b, &res| {
                let mut f = MoogLadder::new(0.3, res);
                b.iter(|| {
                    for &x in &input {
                        criterion::black_box(f.process_sample(x));
                    }
                });
            },
        );
    }

    group.finish();
}

/// State-variable filter — all four outputs computed simultaneously.
fn bench_svf(c: &mut Criterion) {
    let input = make_input();
    let mut group = c.benchmark_group("svf");
    group.throughput(Throughput::Elements(BLOCK as u64));

    for q in [0.5_f32, 1.0, 5.0, 20.0] {
        group.bench_with_input(BenchmarkId::new("q", q), &q, |b, &q_val| {
            let mut f = StateVariableFilter::new(1000.0, q_val, SR);
            b.iter(|| {
                for &x in &input {
                    criterion::black_box(f.process(x));
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_biquad_vs_saturating,
    bench_moog_ladder,
    bench_svf
);
criterion_main!(benches);
