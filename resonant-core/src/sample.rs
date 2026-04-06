//! The [`Sample`] trait — a common interface for scalar DSP sample types.
//!
//! Implementing types: [`f32`], [`f64`], [`i16`], [`i32`], [`Q15`](crate::Q15),
//! [`Q31`](crate::Q31).
//!
//! `Signal<T, D>` does **not** require `T: Sample` — the bound is opt-in for
//! generic algorithms that need float conversion.

use crate::fixed::{Q15, Q31};

/// A scalar sample type that can be losslessly converted to/from floating-point.
///
/// The trait is intentionally minimal: round-trip fidelity depends on the
/// representable range of the concrete type. For example, `i16` maps its full
/// integer range to \[−1.0, 1.0\], so converting `i16::MAX` to `f32` and back
/// saturates correctly.
///
/// # Implementing
///
/// All provided methods (`zero`, `one`) delegate to `from_f32`, so only the
/// four required methods need to be implemented.
pub trait Sample: Copy + Clone + PartialOrd + Default {
    /// Converts this sample to `f32`.
    fn to_f32(self) -> f32;

    /// Converts an `f32` value to this sample type, clamping if out of range.
    fn from_f32(v: f32) -> Self;

    /// Converts this sample to `f64`.
    fn to_f64(self) -> f64;

    /// Converts an `f64` value to this sample type, clamping if out of range.
    fn from_f64(v: f64) -> Self;

    /// Returns the additive identity (silence).
    #[inline]
    fn zero() -> Self {
        Self::from_f32(0.0)
    }

    /// Returns the multiplicative identity.
    #[inline]
    fn one() -> Self {
        Self::from_f32(1.0)
    }
}

// ── f32 ──────────────────────────────────────────────────────────────────────

impl Sample for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

// ── f64 ──────────────────────────────────────────────────────────────────────

impl Sample for f64 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v as f64
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
}

// ── i16 ──────────────────────────────────────────────────────────────────────
//
// Maps [i16::MIN, i16::MAX] → [-1.0, ~1.0) (same convention as Q15).

impl Sample for i16 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32 / 32768.0
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        let scaled = v * 32768.0;
        if scaled >= i16::MAX as f32 {
            i16::MAX
        } else if scaled <= i16::MIN as f32 {
            i16::MIN
        } else {
            scaled as i16
        }
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64 / 32768.0
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        let scaled = v * 32768.0;
        if scaled >= i16::MAX as f64 {
            i16::MAX
        } else if scaled <= i16::MIN as f64 {
            i16::MIN
        } else {
            scaled as i16
        }
    }
}

// ── i32 ──────────────────────────────────────────────────────────────────────
//
// Maps [i32::MIN, i32::MAX] → [-1.0, ~1.0) (same convention as Q31).

impl Sample for i32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        Self::from_f64(v as f64)
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64 / 2_147_483_648.0
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        let scaled = v * 2_147_483_648.0;
        if scaled >= i32::MAX as f64 {
            i32::MAX
        } else if scaled <= i32::MIN as f64 {
            i32::MIN
        } else {
            scaled as i32
        }
    }
}

// ── Q15 ──────────────────────────────────────────────────────────────────────

impl Sample for Q15 {
    #[inline]
    fn to_f32(self) -> f32 {
        Q15::to_f32(self)
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        Q15::from_f32(v)
    }

    #[inline]
    fn to_f64(self) -> f64 {
        Q15::to_f64(self)
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        Q15::from_f64(v)
    }
}

// ── Q31 ──────────────────────────────────────────────────────────────────────

impl Sample for Q31 {
    #[inline]
    fn to_f32(self) -> f32 {
        Q31::to_f32(self)
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        Q31::from_f32(v)
    }

    #[inline]
    fn to_f64(self) -> f64 {
        Q31::to_f64(self)
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        Q31::from_f64(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── zero / one ───────────────────────────────────────────────────────────

    #[test]
    fn f32_zero_one() {
        assert_eq!(f32::zero(), 0.0_f32);
        assert_eq!(f32::one(), 1.0_f32);
    }

    #[test]
    fn f64_zero_one() {
        assert_eq!(f64::zero(), 0.0_f64);
        assert_eq!(f64::one(), 1.0_f64);
    }

    #[test]
    fn i16_zero_one() {
        assert_eq!(i16::zero(), 0_i16);
        assert_eq!(i16::one(), i16::MAX); // 1.0 * 32768 clamps to i16::MAX
    }

    #[test]
    fn i32_zero_one() {
        assert_eq!(i32::zero(), 0_i32);
        assert_eq!(i32::one(), i32::MAX); // 1.0 * 2^31 clamps to i32::MAX
    }

    #[test]
    fn q15_zero_one() {
        assert_eq!(Q15::zero(), Q15::ZERO);
        assert_eq!(Q15::one(), Q15::MAX);
    }

    #[test]
    fn q31_zero_one() {
        assert_eq!(Q31::zero(), Q31::ZERO);
        assert_eq!(Q31::one(), Q31::MAX);
    }

    // ── f32 round-trip ───────────────────────────────────────────────────────

    #[test]
    fn f32_roundtrip() {
        let v = 0.5_f32;
        assert_eq!(f32::from_f32(f32::to_f32(v)), v);
        assert!((f32::from_f64(f32::to_f64(v)) - v).abs() < 1e-6);
    }

    // ── f64 round-trip ───────────────────────────────────────────────────────

    #[test]
    fn f64_roundtrip() {
        let v = 0.123_456_789_f64;
        assert!((f64::from_f64(f64::to_f64(v)) - v).abs() < 1e-12);
    }

    // ── i16 ──────────────────────────────────────────────────────────────────

    #[test]
    fn i16_to_f32_max() {
        // i16::MAX → just below 1.0
        let f = i16::MAX.to_f32();
        assert!(f > 0.999 && f < 1.0);
    }

    #[test]
    fn i16_to_f32_min() {
        // i16::MIN → exactly −1.0
        let f = i16::MIN.to_f32();
        assert!((f - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn i16_from_f32_saturates_positive() {
        assert_eq!(i16::from_f32(1.5), i16::MAX);
    }

    #[test]
    fn i16_from_f32_saturates_negative() {
        assert_eq!(i16::from_f32(-1.5), i16::MIN);
    }

    #[test]
    fn i16_roundtrip_f32() {
        let orig: i16 = 12345;
        let rt = i16::from_f32(orig.to_f32());
        // Precision loss possible at i16 resolution, allow 1 LSB
        assert!((rt as i32 - orig as i32).abs() <= 1);
    }

    // ── i32 ──────────────────────────────────────────────────────────────────

    #[test]
    fn i32_to_f64_max() {
        let f = i32::MAX.to_f64();
        assert!(f > 0.9999 && f < 1.0);
    }

    #[test]
    fn i32_to_f64_min() {
        let f = i32::MIN.to_f64();
        assert!((f - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn i32_from_f32_saturates_positive() {
        assert_eq!(i32::from_f32(1.5), i32::MAX);
    }

    #[test]
    fn i32_from_f32_saturates_negative() {
        assert_eq!(i32::from_f32(-1.5), i32::MIN);
    }

    // ── Q15 ──────────────────────────────────────────────────────────────────

    #[test]
    fn q15_roundtrip_f32() {
        let orig = Q15::from_f32(0.5);
        let rt = Q15::from_f32(orig.to_f32());
        assert_eq!(orig, rt);
    }

    #[test]
    fn q15_roundtrip_f64() {
        let orig = Q15::from_f64(0.25);
        let rt = Q15::from_f64(orig.to_f64());
        assert_eq!(orig, rt);
    }

    // ── Q31 ──────────────────────────────────────────────────────────────────

    #[test]
    fn q31_roundtrip_f32() {
        let orig = Q31::from_f32(0.5);
        let rt = Q31::from_f32(orig.to_f32());
        // f32 can't perfectly represent Q31 precision; allow small error
        assert!((orig.to_f64() - rt.to_f64()).abs() < 1e-6);
    }

    #[test]
    fn q31_roundtrip_f64() {
        let orig = Q31::from_f64(0.25);
        let rt = Q31::from_f64(orig.to_f64());
        assert_eq!(orig, rt);
    }

    // ── generic algorithm test ───────────────────────────────────────────────

    /// Demonstrate that a generic function works across all Sample impls.
    fn scale_by_half<S: Sample>(v: S) -> S {
        S::from_f32(v.to_f32() * 0.5)
    }

    #[test]
    fn generic_scale_f32() {
        assert!((scale_by_half(0.8_f32) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn generic_scale_f64() {
        assert!((scale_by_half(0.8_f64) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn generic_scale_q15() {
        let v = Q15::from_f32(0.8);
        let result = scale_by_half(v);
        assert!((result.to_f32() - 0.4).abs() < 0.001);
    }

    #[test]
    fn generic_scale_q31() {
        let v = Q31::from_f32(0.8);
        let result = scale_by_half(v);
        assert!((result.to_f32() - 0.4).abs() < 0.001);
    }
}
