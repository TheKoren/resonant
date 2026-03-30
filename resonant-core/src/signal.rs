use core::marker::PhantomData;

/// Marker trait for signal domains.
///
/// Implemented by [`TimeDomain`] and [`FreqDomain`]. Downstream crates can
/// implement this for custom domains (e.g. cepstral, mel-frequency).
pub trait Domain {}

/// Marks a signal as existing in the time domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimeDomain;
impl Domain for TimeDomain {}

/// Marks a signal as existing in the frequency domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FreqDomain;
impl Domain for FreqDomain {}

/// A signal tagged with its domain at the type level.
///
/// `Signal` is the core abstraction of the resonant ecosystem. The `D` parameter
/// tracks which domain the data lives in, so operations that are only valid in one
/// domain (e.g. FFT on time-domain data) are enforced at compile time.
///
/// `T` is the storage type — typically `&[f32]`, `Vec<f32>`, or `[f32; N]`.
///
/// # Examples
///
/// ```
/// use resonant_core::signal::{Signal, TimeDomain, FreqDomain};
///
/// let time: Signal<[f32; 4], TimeDomain> = Signal::new([0.0, 0.5, 1.0, 0.5]);
/// assert_eq!(time.data(), &[0.0, 0.5, 1.0, 0.5]);
///
/// let freq: Signal<[f32; 4], FreqDomain> = Signal::new([1.0, 0.0, 0.0, 0.0]);
/// assert_eq!(freq.data(), &[1.0, 0.0, 0.0, 0.0]);
/// ```
///
/// A function that accepts only time-domain signals rejects frequency-domain ones:
///
/// ```compile_fail
/// use resonant_core::signal::{Signal, TimeDomain, FreqDomain};
///
/// fn process_time(sig: Signal<Vec<f32>, TimeDomain>) {}
///
/// let freq: Signal<Vec<f32>, FreqDomain> = Signal::new(vec![1.0]);
/// process_time(freq); // ERROR: expected TimeDomain, found FreqDomain
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signal<T, D: Domain> {
    data: T,
    _domain: PhantomData<D>,
}

impl<T, D: Domain> Signal<T, D> {
    /// Wraps `data` in a signal tagged with domain `D`.
    #[inline]
    #[must_use]
    pub fn new(data: T) -> Self {
        Self {
            data,
            _domain: PhantomData,
        }
    }

    /// Returns a reference to the underlying data.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Returns a mutable reference to the underlying data.
    #[doc(hidden)]
    #[inline]
    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }

    /// Consumes the signal, returning the underlying data.
    #[inline]
    #[must_use]
    pub fn into_inner(self) -> T {
        self.data
    }

    /// Re-tags this signal with a different domain, consuming the original.
    ///
    /// This is the mechanism by which transforms (FFT, IFFT) transition
    /// between domains. Typically called by transform implementations,
    /// not end users.
    #[doc(hidden)]
    #[inline]
    #[must_use]
    pub fn map_domain<D2: Domain>(self) -> Signal<T, D2> {
        Signal {
            data: self.data,
            _domain: PhantomData,
        }
    }
}

impl<T, D: Domain> Signal<T, D>
where
    T: AsRef<[f32]>,
{
    /// Returns the number of samples in the signal.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.as_ref().len()
    }

    /// Returns `true` if the signal contains no samples.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.as_ref().is_empty()
    }

    /// Returns the underlying data as a sample slice.
    #[inline]
    #[must_use]
    pub fn as_samples(&self) -> &[f32] {
        self.data.as_ref()
    }
}

/// Convenience constructor for time-domain signals.
impl<T> Signal<T, TimeDomain> {
    /// Creates a time-domain signal from sample data.
    ///
    /// # Examples
    ///
    /// ```
    /// use resonant_core::signal::Signal;
    ///
    /// let sig = Signal::from_samples([0.0_f32, 0.5, 1.0, 0.5]);
    /// assert_eq!(sig.data(), &[0.0, 0.5, 1.0, 0.5]);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_samples(data: T) -> Self {
        Self::new(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_domain_construction() {
        let sig: Signal<[f32; 4], TimeDomain> = Signal::new([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(sig.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn freq_domain_construction() {
        let sig: Signal<[f32; 2], FreqDomain> = Signal::new([0.5, -0.5]);
        assert_eq!(sig.data(), &[0.5, -0.5]);
    }

    #[test]
    fn from_samples_creates_time_domain() {
        let sig = Signal::from_samples([0.0_f32, 1.0]);
        let _: &Signal<[f32; 2], TimeDomain> = &sig;
        assert_eq!(sig.data(), &[0.0, 1.0]);
    }

    #[test]
    fn into_inner_returns_data() {
        let sig = Signal::<_, TimeDomain>::new([1.0_f32, 2.0]);
        let data = sig.into_inner();
        assert_eq!(data, [1.0, 2.0]);
    }

    #[test]
    fn data_mut_allows_modification() {
        let mut sig = Signal::<_, TimeDomain>::new([0.0_f32; 4]);
        sig.data_mut()[0] = 42.0;
        assert_eq!(sig.data()[0], 42.0);
    }

    #[test]
    fn map_domain_transitions() {
        let time = Signal::<_, TimeDomain>::new([1.0_f32, 2.0]);
        let freq: Signal<[f32; 2], FreqDomain> = time.map_domain();
        assert_eq!(freq.data(), &[1.0, 2.0]);
    }

    #[test]
    fn clone_produces_equal_signal() {
        let sig = Signal::<_, TimeDomain>::new([1.0_f32, 2.0]);
        let cloned = sig.clone();
        assert_eq!(sig, cloned);
    }

    #[test]
    fn len_and_is_empty() {
        let sig = Signal::from_samples([1.0_f32, 2.0, 3.0]);
        assert_eq!(sig.len(), 3);
        assert!(!sig.is_empty());

        let empty = Signal::from_samples([0.0_f32; 0]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn as_samples_returns_slice() {
        let sig = Signal::from_samples([1.0_f32, 2.0, 3.0]);
        assert_eq!(sig.as_samples(), &[1.0, 2.0, 3.0]);
    }
}
