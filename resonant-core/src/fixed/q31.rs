/// Signed 1.31 fixed-point number representing values in \[-1.0, 1.0).
///
/// Internally stored as `i32` where `i32::MIN` maps to −1.0 and
/// `i32::MAX` maps to just below +1.0. Offers higher precision than
/// [`Q15`](super::Q15) at the cost of double the storage.
///
/// Arithmetic operations saturate rather than wrap, matching the
/// behaviour of hardware DSP accumulators.
///
/// # Examples
///
/// ```
/// use resonant_core::fixed::Q31;
///
/// let a = Q31::from_f32(0.5);
/// let b = Q31::from_f32(0.25);
/// let c = a.saturating_add(b);
/// assert!((c.to_f32() - 0.75).abs() < 0.0001);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Q31(i32);

/// Scale factor: 2^31 = 2147483648.
const SCALE: f64 = 2_147_483_648.0;

impl Q31 {
    /// Zero.
    pub const ZERO: Self = Self(0);
    /// Maximum representable value (just below +1.0).
    pub const MAX: Self = Self(i32::MAX);
    /// Minimum representable value (−1.0).
    pub const MIN: Self = Self(i32::MIN);

    /// Creates a `Q31` from a raw `i32` representation.
    #[must_use]
    #[inline]
    pub const fn from_raw(raw: i32) -> Self {
        Self(raw)
    }

    /// Returns the raw `i32` representation.
    #[must_use]
    #[inline]
    pub const fn to_raw(self) -> i32 {
        self.0
    }

    /// Converts an `f32` in \[-1.0, 1.0) to `Q31`, clamping out-of-range values.
    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        // Use f64 intermediate to avoid precision loss during scaling.
        Self::from_f64(value as f64)
    }

    /// Converts this `Q31` to `f32`.
    #[must_use]
    #[inline]
    pub fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    /// Converts an `f64` in \[-1.0, 1.0) to `Q31`, clamping out-of-range values.
    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        let scaled = value * SCALE;
        let clamped = if scaled > i32::MAX as f64 {
            i32::MAX
        } else if scaled < i32::MIN as f64 {
            i32::MIN
        } else {
            scaled as i32
        };
        Self(clamped)
    }

    /// Converts this `Q31` to `f64`.
    #[must_use]
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / SCALE
    }

    /// Saturating addition.
    #[must_use]
    #[inline]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    /// Saturating subtraction.
    #[must_use]
    #[inline]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }

    /// Saturating fixed-point multiplication.
    ///
    /// Uses a 64-bit intermediate to avoid precision loss, then shifts
    /// back to Q31 format.
    #[must_use]
    #[inline]
    pub const fn saturating_mul(self, rhs: Self) -> Self {
        let wide = (self.0 as i64) * (rhs.0 as i64);
        let shifted = wide >> 31;
        let clamped = if shifted > i32::MAX as i64 {
            i32::MAX
        } else if shifted < i32::MIN as i64 {
            i32::MIN
        } else {
            shifted as i32
        };
        Self(clamped)
    }

    /// Returns the absolute value, saturating at `Q31::MAX` for `Q31::MIN`.
    #[must_use]
    #[inline]
    pub const fn abs(self) -> Self {
        Self(self.0.saturating_abs())
    }

    /// Negates the value, saturating at `Q31::MAX` for `Q31::MIN`.
    #[must_use]
    #[inline]
    pub const fn saturating_neg(self) -> Self {
        Self(self.0.saturating_neg())
    }
}

impl From<f32> for Q31 {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl From<Q31> for f32 {
    fn from(q: Q31) -> Self {
        q.to_f32()
    }
}

impl From<f64> for Q31 {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl From<Q31> for f64 {
    fn from(q: Q31) -> Self {
        q.to_f64()
    }
}

impl core::fmt::Debug for Q31 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Q31({} ≈ {:.8})", self.0, self.to_f64())
    }
}

impl core::fmt::Display for Q31 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:.8}", self.to_f64())
    }
}

impl Default for Q31 {
    fn default() -> Self {
        Self::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_roundtrips() {
        let q = Q31::from_f32(0.0);
        assert_eq!(q.to_raw(), 0);
        assert!((q.to_f64()).abs() < 1e-9);
    }

    #[test]
    fn positive_half_roundtrips() {
        let q = Q31::from_f32(0.5);
        assert!((q.to_f32() - 0.5).abs() < 0.0001);
    }

    #[test]
    fn negative_one_roundtrips() {
        let q = Q31::from_f32(-1.0);
        assert_eq!(q.to_raw(), i32::MIN);
        assert!((q.to_f64() - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn clamps_above_one() {
        let q = Q31::from_f32(1.5);
        assert_eq!(q, Q31::MAX);
    }

    #[test]
    fn clamps_below_neg_one() {
        let q = Q31::from_f32(-1.5);
        assert_eq!(q, Q31::MIN);
    }

    #[test]
    fn f64_roundtrip() {
        let q = Q31::from_f64(0.25);
        assert!((q.to_f64() - 0.25).abs() < 1e-7);
    }

    #[test]
    fn f64_clamps() {
        assert_eq!(Q31::from_f64(2.0), Q31::MAX);
        assert_eq!(Q31::from_f64(-2.0), Q31::MIN);
    }

    #[test]
    fn saturating_add_normal() {
        let a = Q31::from_f32(0.25);
        let b = Q31::from_f32(0.25);
        let c = a.saturating_add(b);
        assert!((c.to_f32() - 0.5).abs() < 0.0001);
    }

    #[test]
    fn saturating_add_overflow() {
        let a = Q31::from_f32(0.9);
        let b = Q31::from_f32(0.9);
        assert_eq!(a.saturating_add(b), Q31::MAX);
    }

    #[test]
    fn saturating_sub_normal() {
        let a = Q31::from_f32(0.5);
        let b = Q31::from_f32(0.25);
        let c = a.saturating_sub(b);
        assert!((c.to_f32() - 0.25).abs() < 0.0001);
    }

    #[test]
    fn saturating_sub_underflow() {
        let a = Q31::from_f32(-0.9);
        let b = Q31::from_f32(0.9);
        assert_eq!(a.saturating_sub(b), Q31::MIN);
    }

    #[test]
    fn saturating_mul_normal() {
        let a = Q31::from_f32(0.5);
        let b = Q31::from_f32(0.5);
        let c = a.saturating_mul(b);
        assert!((c.to_f32() - 0.25).abs() < 0.0001);
    }

    #[test]
    fn saturating_mul_by_zero() {
        let a = Q31::from_f32(0.75);
        let c = a.saturating_mul(Q31::ZERO);
        assert_eq!(c, Q31::ZERO);
    }

    #[test]
    fn saturating_mul_negative() {
        let a = Q31::from_f32(0.5);
        let b = Q31::from_f32(-0.5);
        let c = a.saturating_mul(b);
        assert!((c.to_f32() - (-0.25)).abs() < 0.0001);
    }

    #[test]
    fn saturating_mul_overflow() {
        let c = Q31::MIN.saturating_mul(Q31::MIN);
        assert_eq!(c, Q31::MAX);
    }

    #[test]
    fn abs_positive() {
        let q = Q31::from_f32(0.5);
        assert_eq!(q.abs(), q);
    }

    #[test]
    fn abs_negative() {
        let q = Q31::from_f32(-0.5);
        assert!((q.abs().to_f32() - 0.5).abs() < 0.0001);
    }

    #[test]
    fn abs_min_saturates() {
        assert_eq!(Q31::MIN.abs(), Q31::MAX);
    }

    #[test]
    fn saturating_neg_normal() {
        let q = Q31::from_f32(0.5);
        let n = q.saturating_neg();
        assert!((n.to_f32() - (-0.5)).abs() < 0.0001);
    }

    #[test]
    fn saturating_neg_min_saturates() {
        assert_eq!(Q31::MIN.saturating_neg(), Q31::MAX);
    }

    #[test]
    fn from_trait_f32() {
        let q: Q31 = 0.5_f32.into();
        assert!((q.to_f32() - 0.5).abs() < 0.0001);
    }

    #[test]
    fn into_trait_f32() {
        let q = Q31::from_f32(0.5);
        let f: f32 = q.into();
        assert!((f - 0.5).abs() < 0.0001);
    }

    #[test]
    fn from_trait_f64() {
        let q: Q31 = 0.25_f64.into();
        assert!((q.to_f64() - 0.25).abs() < 1e-7);
    }

    #[test]
    fn into_trait_f64() {
        let q = Q31::from_f64(0.25);
        let f: f64 = q.into();
        assert!((f - 0.25).abs() < 1e-7);
    }

    #[test]
    fn default_is_zero() {
        assert_eq!(Q31::default(), Q31::ZERO);
    }

    #[test]
    fn from_raw_roundtrip() {
        let q = Q31::from_raw(1_073_741_824); // 2^30 ≈ 0.5
        assert_eq!(q.to_raw(), 1_073_741_824);
        assert!((q.to_f32() - 0.5).abs() < 0.0001);
    }

    #[test]
    fn ordering() {
        let a = Q31::from_f32(-0.5);
        let b = Q31::ZERO;
        let c = Q31::from_f32(0.5);
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn higher_precision_than_q15() {
        // Q31 should distinguish values that Q15 cannot
        let a = Q31::from_f64(0.000_001);
        let b = Q31::from_f64(0.000_002);
        assert_ne!(a, b);
    }
}
