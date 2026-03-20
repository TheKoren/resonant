/// Signed 1.15 fixed-point number representing values in \[-1.0, 1.0).
///
/// Internally stored as `i16` where `i16::MIN` maps to −1.0 and
/// `i16::MAX` maps to just below +1.0.
///
/// Arithmetic operations saturate rather than wrap, matching the
/// behaviour of hardware DSP accumulators.
///
/// # Examples
///
/// ```
/// use resonant_core::fixed::Q15;
///
/// let a = Q15::from_f32(0.5);
/// let b = Q15::from_f32(0.25);
/// let c = a.saturating_add(b);
/// assert!((c.to_f32() - 0.75).abs() < 0.001);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Q15(i16);

/// Scale factor: 2^15 = 32768.
const SCALE: f32 = 32768.0;
const SCALE_F64: f64 = 32768.0;

impl Q15 {
    /// Zero.
    pub const ZERO: Self = Self(0);
    /// Maximum representable value (just below +1.0).
    pub const MAX: Self = Self(i16::MAX);
    /// Minimum representable value (−1.0).
    pub const MIN: Self = Self(i16::MIN);

    /// Creates a `Q15` from a raw `i16` representation.
    #[must_use]
    #[inline]
    pub const fn from_raw(raw: i16) -> Self {
        Self(raw)
    }

    /// Returns the raw `i16` representation.
    #[must_use]
    #[inline]
    pub const fn to_raw(self) -> i16 {
        self.0
    }

    /// Converts an `f32` in \[-1.0, 1.0) to `Q15`, clamping out-of-range values.
    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        let scaled = value * SCALE;
        let clamped = if scaled > i16::MAX as f32 {
            i16::MAX
        } else if scaled < i16::MIN as f32 {
            i16::MIN
        } else {
            scaled as i16
        };
        Self(clamped)
    }

    /// Converts this `Q15` to `f32`.
    #[must_use]
    #[inline]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / SCALE
    }

    /// Converts an `f64` in \[-1.0, 1.0) to `Q15`, clamping out-of-range values.
    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        let scaled = value * SCALE_F64;
        let clamped = if scaled > i16::MAX as f64 {
            i16::MAX
        } else if scaled < i16::MIN as f64 {
            i16::MIN
        } else {
            scaled as i16
        };
        Self(clamped)
    }

    /// Converts this `Q15` to `f64`.
    #[must_use]
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / SCALE_F64
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
    /// Uses a 32-bit intermediate to avoid precision loss, then shifts
    /// back to Q15 format.
    #[must_use]
    #[inline]
    pub const fn saturating_mul(self, rhs: Self) -> Self {
        // Widen to i32, multiply, shift right by 15, then saturate back to i16.
        let wide = (self.0 as i32) * (rhs.0 as i32);
        let shifted = wide >> 15;
        let clamped = if shifted > i16::MAX as i32 {
            i16::MAX
        } else if shifted < i16::MIN as i32 {
            i16::MIN
        } else {
            shifted as i16
        };
        Self(clamped)
    }

    /// Returns the absolute value, saturating at `Q15::MAX` for `Q15::MIN`.
    #[must_use]
    #[inline]
    pub const fn abs(self) -> Self {
        Self(self.0.saturating_abs())
    }

    /// Negates the value, saturating at `Q15::MAX` for `Q15::MIN`.
    #[must_use]
    #[inline]
    pub const fn saturating_neg(self) -> Self {
        Self(self.0.saturating_neg())
    }
}

impl From<f32> for Q15 {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl From<Q15> for f32 {
    fn from(q: Q15) -> Self {
        q.to_f32()
    }
}

impl From<f64> for Q15 {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl From<Q15> for f64 {
    fn from(q: Q15) -> Self {
        q.to_f64()
    }
}

impl core::fmt::Debug for Q15 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Q15({} ≈ {:.5})", self.0, self.to_f32())
    }
}

impl core::fmt::Display for Q15 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:.5}", self.to_f32())
    }
}

impl Default for Q15 {
    fn default() -> Self {
        Self::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_roundtrips() {
        let q = Q15::from_f32(0.0);
        assert_eq!(q.to_raw(), 0);
        assert!((q.to_f32()).abs() < 1e-5);
    }

    #[test]
    fn positive_half_roundtrips() {
        let q = Q15::from_f32(0.5);
        assert!((q.to_f32() - 0.5).abs() < 0.001);
    }

    #[test]
    fn negative_one_roundtrips() {
        let q = Q15::from_f32(-1.0);
        assert_eq!(q.to_raw(), i16::MIN);
        assert!((q.to_f32() - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn clamps_above_one() {
        let q = Q15::from_f32(1.5);
        assert_eq!(q, Q15::MAX);
    }

    #[test]
    fn clamps_below_neg_one() {
        let q = Q15::from_f32(-1.5);
        assert_eq!(q, Q15::MIN);
    }

    #[test]
    fn f64_roundtrip() {
        let q = Q15::from_f64(0.25);
        assert!((q.to_f64() - 0.25).abs() < 0.001);
    }

    #[test]
    fn f64_clamps() {
        assert_eq!(Q15::from_f64(2.0), Q15::MAX);
        assert_eq!(Q15::from_f64(-2.0), Q15::MIN);
    }

    #[test]
    fn saturating_add_normal() {
        let a = Q15::from_f32(0.25);
        let b = Q15::from_f32(0.25);
        let c = a.saturating_add(b);
        assert!((c.to_f32() - 0.5).abs() < 0.001);
    }

    #[test]
    fn saturating_add_overflow() {
        let a = Q15::from_f32(0.9);
        let b = Q15::from_f32(0.9);
        assert_eq!(a.saturating_add(b), Q15::MAX);
    }

    #[test]
    fn saturating_sub_normal() {
        let a = Q15::from_f32(0.5);
        let b = Q15::from_f32(0.25);
        let c = a.saturating_sub(b);
        assert!((c.to_f32() - 0.25).abs() < 0.001);
    }

    #[test]
    fn saturating_sub_underflow() {
        let a = Q15::from_f32(-0.9);
        let b = Q15::from_f32(0.9);
        assert_eq!(a.saturating_sub(b), Q15::MIN);
    }

    #[test]
    fn saturating_mul_normal() {
        let a = Q15::from_f32(0.5);
        let b = Q15::from_f32(0.5);
        let c = a.saturating_mul(b);
        assert!((c.to_f32() - 0.25).abs() < 0.001);
    }

    #[test]
    fn saturating_mul_by_zero() {
        let a = Q15::from_f32(0.75);
        let c = a.saturating_mul(Q15::ZERO);
        assert_eq!(c, Q15::ZERO);
    }

    #[test]
    fn saturating_mul_negative() {
        let a = Q15::from_f32(0.5);
        let b = Q15::from_f32(-0.5);
        let c = a.saturating_mul(b);
        assert!((c.to_f32() - (-0.25)).abs() < 0.001);
    }

    #[test]
    fn saturating_mul_overflow() {
        // MIN * MIN would overflow without saturation
        let c = Q15::MIN.saturating_mul(Q15::MIN);
        assert_eq!(c, Q15::MAX);
    }

    #[test]
    fn abs_positive() {
        let q = Q15::from_f32(0.5);
        assert_eq!(q.abs(), q);
    }

    #[test]
    fn abs_negative() {
        let q = Q15::from_f32(-0.5);
        assert!((q.abs().to_f32() - 0.5).abs() < 0.001);
    }

    #[test]
    fn abs_min_saturates() {
        // -(-1.0) can't be represented, saturates to MAX
        assert_eq!(Q15::MIN.abs(), Q15::MAX);
    }

    #[test]
    fn saturating_neg_normal() {
        let q = Q15::from_f32(0.5);
        let n = q.saturating_neg();
        assert!((n.to_f32() - (-0.5)).abs() < 0.001);
    }

    #[test]
    fn saturating_neg_min_saturates() {
        assert_eq!(Q15::MIN.saturating_neg(), Q15::MAX);
    }

    #[test]
    fn from_trait_f32() {
        let q: Q15 = 0.5_f32.into();
        assert!((q.to_f32() - 0.5).abs() < 0.001);
    }

    #[test]
    fn into_trait_f32() {
        let q = Q15::from_f32(0.5);
        let f: f32 = q.into();
        assert!((f - 0.5).abs() < 0.001);
    }

    #[test]
    fn from_trait_f64() {
        let q: Q15 = 0.25_f64.into();
        assert!((q.to_f64() - 0.25).abs() < 0.001);
    }

    #[test]
    fn into_trait_f64() {
        let q = Q15::from_f64(0.25);
        let f: f64 = q.into();
        assert!((f - 0.25).abs() < 0.001);
    }

    #[test]
    fn default_is_zero() {
        assert_eq!(Q15::default(), Q15::ZERO);
    }

    #[test]
    fn from_raw_roundtrip() {
        let q = Q15::from_raw(16384);
        assert_eq!(q.to_raw(), 16384);
        assert!((q.to_f32() - 0.5).abs() < 0.001);
    }

    #[test]
    fn ordering() {
        let a = Q15::from_f32(-0.5);
        let b = Q15::ZERO;
        let c = Q15::from_f32(0.5);
        assert!(a < b);
        assert!(b < c);
    }
}
