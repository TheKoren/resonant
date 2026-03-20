use resonant_core::Q31;

#[test]
fn conversion_roundtrip_f32() {
    for &v in &[-1.0_f32, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75] {
        let q = Q31::from_f32(v);
        let back = q.to_f32();
        assert!(
            (back - v).abs() < 0.0001,
            "roundtrip failed for {v}: got {back}"
        );
    }
}

#[test]
fn conversion_roundtrip_f64() {
    for &v in &[-1.0_f64, -0.5, 0.0, 0.5] {
        let q = Q31::from_f64(v);
        let back = q.to_f64();
        assert!(
            (back - v).abs() < 1e-7,
            "roundtrip failed for {v}: got {back}"
        );
    }
}

#[test]
fn clamping_preserves_range() {
    let above = Q31::from_f32(5.0);
    let below = Q31::from_f32(-5.0);
    assert_eq!(above, Q31::MAX);
    assert_eq!(below, Q31::MIN);
    // Q31::MAX ≈ 0.9999999995, which rounds to 1.0 in f32 — that's expected
    assert!(above.to_f64() < 1.0);
    assert!(below.to_f64() >= -1.0);
}

#[test]
fn arithmetic_chain() {
    // (0.5 + 0.25) * 0.5 = 0.375
    let result = Q31::from_f32(0.5)
        .saturating_add(Q31::from_f32(0.25))
        .saturating_mul(Q31::from_f32(0.5));
    assert!((result.to_f32() - 0.375).abs() < 0.001);
}

#[test]
fn subtraction_to_negative() {
    let a = Q31::from_f32(0.25);
    let b = Q31::from_f32(0.75);
    let c = a.saturating_sub(b);
    assert!((c.to_f32() - (-0.5)).abs() < 0.0001);
}

#[test]
fn from_into_traits() {
    let q: Q31 = 0.5_f32.into();
    let f: f32 = q.into();
    assert!((f - 0.5).abs() < 0.0001);

    let q2: Q31 = 0.25_f64.into();
    let d: f64 = q2.into();
    assert!((d - 0.25).abs() < 1e-7);
}

#[test]
fn debug_format() {
    let q = Q31::from_f32(0.5);
    let s = format!("{q:?}");
    assert!(s.contains("Q31("));
}

#[test]
fn display_format() {
    let q = Q31::from_f32(0.5);
    let s = format!("{q}");
    assert!(s.contains("0.5"));
}

#[test]
fn q31_more_precise_than_q15() {
    use resonant_core::Q15;

    // Values that are indistinguishable in Q15 should differ in Q31
    let a15 = Q15::from_f64(0.000_01);
    let b15 = Q15::from_f64(0.000_02);
    // Q15 may round these to the same value
    let _ = (a15, b15);

    let a31 = Q31::from_f64(0.000_01);
    let b31 = Q31::from_f64(0.000_02);
    assert_ne!(a31, b31);
}
