//! Heap-backed ring buffer for runtime-determined capacities.
//!
//! Requires the `alloc` feature. For `no_std`/`no_alloc` contexts use
//! [`RingBuf<T, N>`](crate::RingBuf) instead.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::ring_buf::ops;

/// A heap-allocated circular buffer with a runtime-specified capacity.
///
/// Semantics mirror [`RingBuf<T, N>`](crate::RingBuf): when full, [`push`](HeapRingBuf::push)
/// overwrites the oldest element. Useful where the capacity is not known at
/// compile time (e.g. a pipeline node whose buffer size is set from a config).
///
/// # Examples
///
/// ```
/// use resonant_core::HeapRingBuf;
///
/// let mut buf = HeapRingBuf::new(3);
/// buf.push(1.0_f32);
/// buf.push(2.0);
/// buf.push(3.0);
/// assert!(buf.is_full());
///
/// buf.push(4.0); // overwrites oldest
/// assert_eq!(buf.pop(), Some(2.0));
/// ```
#[derive(Debug, Clone)]
pub struct HeapRingBuf<T: Copy> {
    buf: Vec<T>,
    head: usize,
    len: usize,
    capacity: usize,
}

impl<T: Copy + Default> HeapRingBuf<T> {
    /// Creates an empty ring buffer with the given capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(
            capacity > 0,
            "HeapRingBuf capacity must be greater than zero"
        );
        Self {
            buf: vec![T::default(); capacity],
            head: 0,
            len: 0,
            capacity,
        }
    }
}

impl<T: Copy> HeapRingBuf<T> {
    /// Appends a value, overwriting the oldest element if the buffer is full.
    #[inline]
    pub fn push(&mut self, value: T) {
        ops::push(&mut self.buf, &mut self.head, &mut self.len, value);
    }

    /// Removes and returns the oldest element, or `None` if empty.
    #[inline]
    #[must_use]
    pub fn pop(&mut self) -> Option<T> {
        ops::pop(&self.buf, &mut self.head, &mut self.len)
    }

    /// Returns a reference to the oldest element without removing it.
    #[inline]
    #[must_use]
    pub fn peek(&self) -> Option<&T> {
        ops::peek(&self.buf, self.head, self.len)
    }

    /// Returns the contents as two contiguous slices, oldest element first.
    ///
    /// The concatenation of the two slices is always the full sequence from
    /// oldest to newest. The second slice is empty when the buffer has not
    /// wrapped around.
    #[inline]
    #[must_use]
    pub fn as_slices(&self) -> (&[T], &[T]) {
        ops::as_slices(&self.buf, self.head, self.len)
    }

    /// Removes and returns all elements from oldest to newest.
    ///
    /// The buffer is empty after the returned iterator is consumed.
    #[inline]
    pub fn drain(&mut self) -> HeapRingBufDrain<'_, T> {
        HeapRingBufDrain { buf: self }
    }

    /// Returns `true` if the buffer contains `capacity` elements.
    #[inline]
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Returns `true` if the buffer contains no elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of elements currently stored.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the maximum number of elements the buffer can hold.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Removes all elements without deallocating.
    #[inline]
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Returns an iterator over elements from oldest to newest.
    ///
    /// # Examples
    ///
    /// ```
    /// use resonant_core::HeapRingBuf;
    ///
    /// let mut buf = HeapRingBuf::new(4);
    /// buf.push(1_i32); buf.push(2); buf.push(3);
    /// let vals: Vec<i32> = buf.iter().copied().collect();
    /// assert_eq!(vals, vec![1, 2, 3]);
    /// ```
    #[inline]
    pub fn iter(&self) -> HeapRingBufIter<'_, T> {
        HeapRingBufIter {
            buf: self,
            offset: 0,
        }
    }
}

/// Iterator over [`HeapRingBuf`] elements from oldest to newest.
#[derive(Debug)]
pub struct HeapRingBufIter<'a, T: Copy> {
    buf: &'a HeapRingBuf<T>,
    offset: usize,
}

impl<'a, T: Copy> Iterator for HeapRingBufIter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        ops::iter_next(&self.buf.buf, self.buf.head, &mut self.offset, self.buf.len)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        ops::size_hint(self.buf.len, self.offset)
    }
}

impl<T: Copy> ExactSizeIterator for HeapRingBufIter<'_, T> {}

/// Draining iterator produced by [`HeapRingBuf::drain`].
#[derive(Debug)]
pub struct HeapRingBufDrain<'a, T: Copy> {
    buf: &'a mut HeapRingBuf<T>,
}

impl<T: Copy> Iterator for HeapRingBufDrain<'_, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.buf.pop()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.buf.len, Some(self.buf.len))
    }
}

impl<T: Copy> ExactSizeIterator for HeapRingBufDrain<'_, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_empty() {
        let buf = HeapRingBuf::<f32>::new(4);
        assert!(buf.is_empty());
        assert!(!buf.is_full());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), 4);
    }

    #[test]
    fn push_and_pop_fifo() {
        let mut buf = HeapRingBuf::new(4);
        buf.push(1_i32);
        buf.push(2);
        buf.push(3);
        assert_eq!(buf.pop(), Some(1));
        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.pop(), Some(3));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn overwrites_oldest_when_full() {
        let mut buf = HeapRingBuf::new(3);
        buf.push(1_i32);
        buf.push(2);
        buf.push(3);
        buf.push(4); // overwrites 1
        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.pop(), Some(3));
        assert_eq!(buf.pop(), Some(4));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn as_slices_no_wrap() {
        let mut buf = HeapRingBuf::new(4);
        buf.push(1_i32);
        buf.push(2);
        buf.push(3);
        let (a, b) = buf.as_slices();
        assert_eq!(a, &[1, 2, 3]);
        assert!(b.is_empty());
    }

    #[test]
    fn as_slices_wrapped() {
        let mut buf = HeapRingBuf::new(4);
        buf.push(1_i32);
        buf.push(2);
        buf.push(3);
        buf.push(4);
        let _ = buf.pop();
        buf.push(5); // wraps
        let (a, b) = buf.as_slices();
        let combined: Vec<i32> = a.iter().chain(b).copied().collect();
        assert_eq!(combined, vec![2, 3, 4, 5]);
    }

    #[test]
    fn iter_oldest_first() {
        let mut buf = HeapRingBuf::new(4);
        buf.push(10_i32);
        buf.push(20);
        buf.push(30);
        let vals: Vec<i32> = buf.iter().copied().collect();
        assert_eq!(vals, vec![10, 20, 30]);
    }

    #[test]
    fn iter_after_wraparound() {
        let mut buf = HeapRingBuf::new(3);
        for i in 0..5_i32 {
            buf.push(i);
        }
        let vals: Vec<i32> = buf.iter().copied().collect();
        assert_eq!(vals, vec![2, 3, 4]);
    }

    #[test]
    fn drain_yields_all_and_empties() {
        let mut buf = HeapRingBuf::new(4);
        buf.push(1_i32);
        buf.push(2);
        buf.push(3);
        let drained: Vec<i32> = buf.drain().collect();
        assert_eq!(drained, vec![1, 2, 3]);
        assert!(buf.is_empty());
    }

    #[test]
    fn clear_resets() {
        let mut buf = HeapRingBuf::new(4);
        buf.push(1_i32);
        buf.push(2);
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.pop(), None);
    }

    #[test]
    #[should_panic(expected = "capacity must be greater than zero")]
    fn zero_capacity_panics() {
        let _ = HeapRingBuf::<f32>::new(0);
    }

    #[test]
    fn matches_ring_buf_behaviour() {
        // Same sequence of operations on both types should yield identical results.
        use crate::RingBuf;
        let mut fixed = RingBuf::<i32, 3>::new();
        let mut heap = HeapRingBuf::new(3);

        for i in 0..5_i32 {
            fixed.push(i);
            heap.push(i);
        }

        let fixed_vals: Vec<i32> = fixed.iter().copied().collect();
        let heap_vals: Vec<i32> = heap.iter().copied().collect();
        assert_eq!(fixed_vals, heap_vals);
    }
}
