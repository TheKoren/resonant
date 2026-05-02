//! Arithmetic operators for [`Signal`].
//!
//! All operations are element-wise and domain-preserving. The domain `D` must
//! be the same on both operands — mixing domains is a compile error.
//!
//! # Domain safety
//!
//! ```compile_fail
//! use resonant_core::signal::{Signal, TimeDomain, FreqDomain};
//!
//! let a = Signal::<[f32; 4], TimeDomain>::new([1.0; 4]);
//! let b = Signal::<[f32; 4], FreqDomain>::new([2.0; 4]);
//! let _ = a + b; // ERROR: mismatched domain types
//! ```

use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::signal::{Domain, Signal};

impl<T, D: Domain> Add for Signal<T, D>
where
    T: AsMut<[f32]> + AsRef<[f32]>,
{
    type Output = Self;
    /// Element-wise addition.
    ///
    /// # Panics
    ///
    /// Panics if the two signals have different lengths. For fixed-size array
    /// storage (`[f32; N]`) the lengths are equal by construction and the check
    /// is eliminated at compile time.
    ///
    /// # Examples
    ///
    /// ```
    /// use resonant_core::signal::{Signal, TimeDomain};
    ///
    /// let a = Signal::<[f32; 3], TimeDomain>::new([1.0, 2.0, 3.0]);
    /// let b = Signal::<[f32; 3], TimeDomain>::new([4.0, 5.0, 6.0]);
    /// assert_eq!((a + b).data(), &[5.0, 7.0, 9.0]);
    /// ```
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut lhs = self.into_inner();
        let rhs = rhs.into_inner();
        assert_eq!(
            lhs.as_ref().len(),
            rhs.as_ref().len(),
            "signal length mismatch"
        );
        for (a, b) in lhs.as_mut().iter_mut().zip(rhs.as_ref()) {
            *a += b;
        }
        Signal::new(lhs)
    }
}

impl<T, D: Domain> Sub for Signal<T, D>
where
    T: AsMut<[f32]> + AsRef<[f32]>,
{
    type Output = Self;
    /// Element-wise subtraction.
    ///
    /// # Panics
    ///
    /// Panics if the two signals have different lengths. For array storage the
    /// check is eliminated at compile time.
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut lhs = self.into_inner();
        let rhs = rhs.into_inner();
        assert_eq!(
            lhs.as_ref().len(),
            rhs.as_ref().len(),
            "signal length mismatch"
        );
        for (a, b) in lhs.as_mut().iter_mut().zip(rhs.as_ref()) {
            *a -= b;
        }
        Signal::new(lhs)
    }
}

impl<T, D: Domain> Mul<f32> for Signal<T, D>
where
    T: AsMut<[f32]>,
{
    type Output = Self;
    /// Scales every sample by `rhs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use resonant_core::signal::{Signal, TimeDomain};
    ///
    /// let sig = Signal::<[f32; 4], TimeDomain>::new([1.0, 2.0, 3.0, 4.0]);
    /// let scaled = sig * 0.5;
    /// assert_eq!(scaled.data(), &[0.5, 1.0, 1.5, 2.0]);
    /// ```
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        let mut data = self.into_inner();
        for a in data.as_mut().iter_mut() {
            *a *= rhs;
        }
        Signal::new(data)
    }
}

impl<T, D: Domain> Div<f32> for Signal<T, D>
where
    T: AsMut<[f32]>,
{
    type Output = Self;
    /// Divides every sample by `rhs`.
    #[inline]
    fn div(self, rhs: f32) -> Self {
        self * (1.0 / rhs)
    }
}

impl<T, D: Domain> Neg for Signal<T, D>
where
    T: AsMut<[f32]>,
{
    type Output = Self;
    /// Negates every sample.
    #[inline]
    fn neg(self) -> Self {
        self * -1.0
    }
}

impl<T, D: Domain> Signal<T, D>
where
    T: AsRef<[f32]> + AsMut<[f32]> + Clone,
{
    /// Returns a new signal equal to `self + other * gain`.
    ///
    /// Useful for wet/dry blending and cross-fading. Both signals must be in
    /// the same domain.
    ///
    /// # Panics
    ///
    /// Panics if the two signals have different lengths. For array storage the
    /// check is eliminated at compile time.
    ///
    /// # Examples
    ///
    /// ```
    /// use resonant_core::signal::{Signal, TimeDomain};
    ///
    /// let dry = Signal::<[f32; 4], TimeDomain>::new([1.0; 4]);
    /// let wet = Signal::<[f32; 4], TimeDomain>::new([0.0; 4]);
    /// let out = dry.mix(&wet, 0.5);
    /// assert_eq!(out.data(), &[1.0; 4]); // dry + wet*0.5 = 1.0 + 0.0 = 1.0
    /// ```
    #[must_use]
    pub fn mix(&self, other: &Self, gain: f32) -> Self {
        assert_eq!(self.len(), other.len(), "signal length mismatch");
        let mut data = self.data().clone();
        for (a, b) in data.as_mut().iter_mut().zip(other.data().as_ref()) {
            *a += b * gain;
        }
        Signal::new(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::{FreqDomain, TimeDomain};

    #[test]
    fn array_add_same_domain() {
        let a = Signal::<[f32; 4], TimeDomain>::new([1.0, 2.0, 3.0, 4.0]);
        let b = Signal::<[f32; 4], TimeDomain>::new([4.0, 3.0, 2.0, 1.0]);
        let c = a + b;
        assert_eq!(c.data(), &[5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn array_sub() {
        let a = Signal::<[f32; 3], TimeDomain>::new([3.0, 2.0, 1.0]);
        let b = Signal::<[f32; 3], TimeDomain>::new([1.0, 1.0, 1.0]);
        let c = a - b;
        assert_eq!(c.data(), &[2.0, 1.0, 0.0]);
    }

    #[test]
    fn array_mul_scalar() {
        let s = Signal::<[f32; 4], TimeDomain>::new([1.0, 2.0, 3.0, 4.0]);
        let scaled = s * 2.0;
        assert_eq!(scaled.data(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn array_div_scalar() {
        let s = Signal::<[f32; 4], TimeDomain>::new([2.0, 4.0, 6.0, 8.0]);
        let d = s / 2.0;
        assert_eq!(d.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn array_neg() {
        let s = Signal::<[f32; 3], TimeDomain>::new([1.0, -2.0, 3.0]);
        let n = -s;
        assert_eq!(n.data(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn array_mix_zero_gain() {
        let dry = Signal::<[f32; 4], TimeDomain>::new([1.0; 4]);
        let wet = Signal::<[f32; 4], TimeDomain>::new([99.0; 4]);
        let out = dry.mix(&wet, 0.0);
        assert_eq!(out.data(), &[1.0; 4]);
    }

    #[test]
    fn array_mix_full_gain() {
        let dry = Signal::<[f32; 4], TimeDomain>::new([1.0; 4]);
        let wet = Signal::<[f32; 4], TimeDomain>::new([1.0; 4]);
        let out = dry.mix(&wet, 1.0);
        assert_eq!(out.data(), &[2.0; 4]);
    }

    #[test]
    fn array_operators_work_on_freq_domain() {
        let a = Signal::<[f32; 2], FreqDomain>::new([1.0, 0.5]);
        let b = Signal::<[f32; 2], FreqDomain>::new([0.5, 0.5]);
        let c = a + b;
        assert_eq!(c.data(), &[1.5, 1.0]);
    }

    #[cfg(feature = "alloc")]
    mod vec_tests {
        extern crate alloc;
        use alloc::vec;

        use super::*;

        #[test]
        fn vec_add() {
            let a = Signal::<alloc::vec::Vec<f32>, TimeDomain>::new(vec![1.0, 2.0, 3.0]);
            let b = Signal::<alloc::vec::Vec<f32>, TimeDomain>::new(vec![4.0, 5.0, 6.0]);
            let c = a + b;
            assert_eq!(c.data(), &[5.0, 7.0, 9.0]);
        }

        #[test]
        fn vec_sub() {
            let a = Signal::<alloc::vec::Vec<f32>, TimeDomain>::new(vec![5.0, 5.0]);
            let b = Signal::<alloc::vec::Vec<f32>, TimeDomain>::new(vec![3.0, 2.0]);
            let c = a - b;
            assert_eq!(c.data(), &[2.0, 3.0]);
        }

        #[test]
        fn vec_mul_scalar() {
            let s = Signal::<alloc::vec::Vec<f32>, TimeDomain>::new(vec![1.0, 2.0, 4.0]);
            let scaled = s * 3.0;
            assert_eq!(scaled.data(), &[3.0, 6.0, 12.0]);
        }

        #[test]
        fn vec_div_scalar() {
            let s = Signal::<alloc::vec::Vec<f32>, TimeDomain>::new(vec![6.0, 9.0]);
            let d = s / 3.0;
            assert_eq!(d.data(), &[2.0, 3.0]);
        }

        #[test]
        fn vec_neg() {
            let s = Signal::<alloc::vec::Vec<f32>, TimeDomain>::new(vec![1.0, -1.0, 0.0]);
            let n = -s;
            assert_eq!(n.data(), &[-1.0, 1.0, 0.0]);
        }

        #[test]
        fn vec_mix() {
            let dry = Signal::<alloc::vec::Vec<f32>, TimeDomain>::new(vec![1.0, 2.0]);
            let wet = Signal::<alloc::vec::Vec<f32>, TimeDomain>::new(vec![1.0, 1.0]);
            let out = dry.mix(&wet, 0.5);
            assert_eq!(out.data(), &[1.5, 2.5]);
        }

        #[test]
        #[should_panic(expected = "signal length mismatch")]
        fn vec_add_length_mismatch_panics() {
            let a = Signal::<alloc::vec::Vec<f32>, TimeDomain>::new(vec![1.0, 2.0]);
            let b = Signal::<alloc::vec::Vec<f32>, TimeDomain>::new(vec![1.0]);
            let _ = a + b;
        }
    }
}
