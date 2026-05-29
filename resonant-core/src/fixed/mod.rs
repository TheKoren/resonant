//! Fixed-point arithmetic types for deterministic, FPU-free DSP.
//!
//! These types map a signed integer range onto the real interval \[-1.0, 1.0).
//! They are useful on targets without hardware floating-point or where
//! bit-exact reproducibility is required.

mod q15;
mod q31;

pub use q15::Q15;
pub use q31::Q31;

use crate::sample::Sample;

/// Common interface for fixed-point DSP sample types.
///
/// Extends [`Sample`] with saturating arithmetic and raw-value access. A
/// function bounded on `T: FixedPoint` works generically over both [`Q15`]
/// and [`Q31`] without knowing the underlying integer width.
///
/// # Examples
///
/// ```
/// use resonant_core::fixed::{FixedPoint, Q15, Q31};
///
/// fn saturating_sum<T: FixedPoint>(xs: &[T]) -> T {
///     xs.iter().fold(T::zero(), |acc, &x| acc.saturating_add(x))
/// }
///
/// assert!((saturating_sum(&[Q15::from_f32(0.25), Q15::from_f32(0.25)]).to_f32() - 0.5).abs() < 0.01);
/// assert!((saturating_sum(&[Q31::from_f32(0.25), Q31::from_f32(0.25)]).to_f32() - 0.5).abs() < 0.0001);
/// ```
pub trait FixedPoint: Sample {
    /// The underlying integer storage type: [`i16`] for [`Q15`], [`i32`] for [`Q31`].
    type Raw: Copy + Eq + Ord + Default;

    /// Wraps a raw integer without float conversion.
    #[must_use]
    fn from_raw(raw: Self::Raw) -> Self;

    /// Returns the underlying raw integer representation.
    #[must_use]
    fn to_raw(self) -> Self::Raw;

    /// Saturating addition. Returns [`Sample::MAX`] on overflow, [`Sample::MIN`] on underflow.
    #[must_use]
    fn saturating_add(self, rhs: Self) -> Self;

    /// Saturating subtraction. Returns [`Sample::MIN`] on underflow, [`Sample::MAX`] on overflow.
    #[must_use]
    fn saturating_sub(self, rhs: Self) -> Self;

    /// Saturating fixed-point multiplication using a wider intermediate.
    ///
    /// [`Q15`] widens to `i32`; [`Q31`] widens to `i64`. The product is
    /// shifted back to the original format and saturated to `[MIN, MAX]`.
    #[must_use]
    fn saturating_mul(self, rhs: Self) -> Self;

    /// Absolute value, saturating at [`Sample::MAX`] for the most negative representable value.
    #[must_use]
    fn abs(self) -> Self;

    /// Negation, saturating at [`Sample::MAX`] for the most negative representable value.
    #[must_use]
    fn saturating_neg(self) -> Self;
}

impl FixedPoint for Q15 {
    type Raw = i16;

    #[inline]
    fn from_raw(raw: i16) -> Self {
        Q15::from_raw(raw)
    }

    #[inline]
    fn to_raw(self) -> i16 {
        Q15::to_raw(self)
    }

    #[inline]
    fn saturating_add(self, rhs: Self) -> Self {
        Q15::saturating_add(self, rhs)
    }

    #[inline]
    fn saturating_sub(self, rhs: Self) -> Self {
        Q15::saturating_sub(self, rhs)
    }

    #[inline]
    fn saturating_mul(self, rhs: Self) -> Self {
        Q15::saturating_mul(self, rhs)
    }

    #[inline]
    fn abs(self) -> Self {
        Q15::abs(self)
    }

    #[inline]
    fn saturating_neg(self) -> Self {
        Q15::saturating_neg(self)
    }
}

impl FixedPoint for Q31 {
    type Raw = i32;

    #[inline]
    fn from_raw(raw: i32) -> Self {
        Q31::from_raw(raw)
    }

    #[inline]
    fn to_raw(self) -> i32 {
        Q31::to_raw(self)
    }

    #[inline]
    fn saturating_add(self, rhs: Self) -> Self {
        Q31::saturating_add(self, rhs)
    }

    #[inline]
    fn saturating_sub(self, rhs: Self) -> Self {
        Q31::saturating_sub(self, rhs)
    }

    #[inline]
    fn saturating_mul(self, rhs: Self) -> Self {
        Q31::saturating_mul(self, rhs)
    }

    #[inline]
    fn abs(self) -> Self {
        Q31::abs(self)
    }

    #[inline]
    fn saturating_neg(self) -> Self {
        Q31::saturating_neg(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn saturating_sum<T: FixedPoint>(xs: &[T]) -> T {
        xs.iter().fold(T::zero(), |acc, &x| acc.saturating_add(x))
    }

    #[test]
    fn q15_saturating_sum() {
        let vals = [Q15::from_f32(0.25), Q15::from_f32(0.25)];
        assert!((saturating_sum(&vals).to_f32() - 0.5).abs() < 0.01);
    }

    #[test]
    fn q31_saturating_sum() {
        let vals = [Q31::from_f32(0.25), Q31::from_f32(0.25)];
        assert!((saturating_sum(&vals).to_f32() - 0.5).abs() < 0.0001);
    }

    #[test]
    fn generic_saturating_mul() {
        fn scale<T: FixedPoint>(a: T, b: T) -> T {
            a.saturating_mul(b)
        }
        assert!((scale(Q15::from_f32(0.5), Q15::from_f32(0.5)).to_f32() - 0.25).abs() < 0.01);
        assert!((scale(Q31::from_f32(0.5), Q31::from_f32(0.5)).to_f32() - 0.25).abs() < 0.0001);
    }

    #[test]
    fn q15_from_to_raw() {
        let raw: i16 = 16384;
        let q = <Q15 as FixedPoint>::from_raw(raw);
        assert_eq!(<Q15 as FixedPoint>::to_raw(q), raw);
        assert!((q.to_f32() - 0.5).abs() < 0.001);
    }

    #[test]
    fn q31_from_to_raw() {
        let raw: i32 = 1_073_741_824;
        let q = <Q31 as FixedPoint>::from_raw(raw);
        assert_eq!(<Q31 as FixedPoint>::to_raw(q), raw);
        assert!((q.to_f32() - 0.5).abs() < 0.0001);
    }

    #[test]
    fn generic_abs_and_neg() {
        fn check<T: FixedPoint>(v: T) {
            assert!(v.abs().to_f32() >= 0.0);
            assert!(v.saturating_neg().saturating_neg() == v);
        }
        check(Q15::from_f32(0.5));
        check(Q31::from_f32(0.5));
    }

    #[test]
    fn saturation_clamps_at_bounds() {
        fn add_overflow<T: FixedPoint>() {
            assert!(T::MAX.saturating_add(T::MAX) == T::MAX);
        }
        fn sub_underflow<T: FixedPoint>() {
            assert!(T::MIN.saturating_sub(T::MAX) == T::MIN);
        }
        add_overflow::<Q15>();
        add_overflow::<Q31>();
        sub_underflow::<Q15>();
        sub_underflow::<Q31>();
    }

    #[test]
    fn raw_round_trip_both_types() {
        let q15_raw: <Q15 as FixedPoint>::Raw = 1000_i16;
        assert_eq!(Q15::from_raw(q15_raw).to_raw(), q15_raw);

        let q31_raw: <Q31 as FixedPoint>::Raw = 1_000_000_i32;
        assert_eq!(Q31::from_raw(q31_raw).to_raw(), q31_raw);
    }
}
