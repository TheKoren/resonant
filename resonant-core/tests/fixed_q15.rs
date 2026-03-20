use resonant_core::Q15;

#[test]
fn conversion_roundtrip_f32() {
    for &v in &[-1.0_f32, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75] {
        let q = Q15::from_f32(v);
        let back = q.to_f32();
        assert!(
            (back - v).abs() < 0.001,
            "roundtrip failed for {v}: got {back}"
        );
    }
}

#[test]
fn conversion_roundtrip_f64() {
    for &v in &[-1.0_f64, -0.5, 0.0, 0.5] {
        let q = Q15::from_f64(v);
        let back = q.to_f64();
        assert!(
            (back - v).abs() < 0.001,
            "roundtrip failed for {v}: got {back}"
        );
    }
}

#[test]
fn clamping_preserves_range() {
    let above = Q15::from_f32(5.0);
    let below = Q15::from_f32(-5.0);
    assert!(above.to_f32() < 1.0);
    assert!(below.to_f32() >= -1.0);
}

#[test]
fn arithmetic_chain() {
    // (0.5 + 0.25) * 0.5 = 0.375
    let result = Q15::from_f32(0.5)
        .saturating_add(Q15::from_f32(0.25))
        .saturating_mul(Q15::from_f32(0.5));
    assert!((result.to_f32() - 0.375).abs() < 0.01);
}

#[test]
fn subtraction_to_negative() {
    let a = Q15::from_f32(0.25);
    let b = Q15::from_f32(0.75);
    let c = a.saturating_sub(b);
    assert!((c.to_f32() - (-0.5)).abs() < 0.001);
}

#[test]
fn signal_with_q15_samples() {
    use resonant_core::{Signal, TimeDomain};

    let samples = [Q15::from_f32(0.0), Q15::from_f32(0.5), Q15::from_f32(-0.5)];
    let sig = Signal::<[Q15; 3], TimeDomain>::new(samples);
    assert_eq!(sig.into_inner().len(), 3);
}

#[test]
fn debug_format() {
    let q = Q15::from_f32(0.5);
    let s = format!("{q:?}");
    assert!(s.contains("Q15("));
}

#[test]
fn display_format() {
    let q = Q15::from_f32(0.5);
    let s = format!("{q}");
    assert!(s.contains("0.5"));
}

#[test]
fn from_into_traits() {
    let q: Q15 = 0.5_f32.into();
    let f: f32 = q.into();
    assert!((f - 0.5).abs() < 0.001);

    let q2: Q15 = 0.25_f64.into();
    let d: f64 = q2.into();
    assert!((d - 0.25).abs() < 0.001);
}
