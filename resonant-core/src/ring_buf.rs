/// A fixed-capacity circular buffer backed by a stack-allocated array.
///
/// `RingBuf` stores up to `N` elements of type `T`. When full, [`push`](RingBuf::push)
/// overwrites the oldest element. This is the standard behaviour for streaming DSP
/// buffers where only the most recent N samples matter.
///
/// # Examples
///
/// ```
/// use resonant_core::RingBuf;
///
/// let mut buf = RingBuf::<f32, 3>::new();
/// buf.push(1.0);
/// buf.push(2.0);
/// buf.push(3.0);
/// assert!(buf.is_full());
///
/// // Pushing when full overwrites the oldest element
/// buf.push(4.0);
/// assert_eq!(buf.pop(), Some(2.0));
/// ```
#[derive(Debug, Clone)]
pub struct RingBuf<T: Copy + Default, const N: usize> {
    buf: [T; N],
    head: usize,
    len: usize,
}

impl<T: Copy + Default, const N: usize> RingBuf<T, N> {
    /// Creates an empty ring buffer.
    ///
    /// # Panics
    ///
    /// Compile-time panic if `N` is zero.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        const { assert!(N > 0, "RingBuf capacity must be greater than zero") }
        Self {
            buf: [T::default(); N],
            head: 0,
            len: 0,
        }
    }

    /// Appends a value, overwriting the oldest element if the buffer is full.
    #[inline]
    pub fn push(&mut self, value: T) {
        let write_idx = (self.head + self.len) % N;
        if self.len == N {
            // full — overwrite oldest, advance head
            self.buf[write_idx] = value;
            self.head = (self.head + 1) % N;
        } else {
            self.buf[write_idx] = value;
            self.len += 1;
        }
    }

    /// Removes and returns the oldest element, or `None` if empty.
    #[inline]
    #[must_use]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let value = self.buf[self.head];
        self.head = (self.head + 1) % N;
        self.len -= 1;
        Some(value)
    }

    /// Returns a reference to the oldest element without removing it.
    #[inline]
    #[must_use]
    pub fn peek(&self) -> Option<&T> {
        if self.len == 0 {
            return None;
        }
        Some(&self.buf[self.head])
    }

    /// Returns `true` if the buffer contains `N` elements.
    #[inline]
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len == N
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

    /// Returns the fixed capacity `N`.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        N
    }

    /// Removes all elements, resetting the buffer to its initial state.
    #[inline]
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Returns an iterator over the elements from oldest to newest.
    ///
    /// # Examples
    ///
    /// ```
    /// use resonant_core::RingBuf;
    ///
    /// let mut buf = RingBuf::<i32, 4>::new();
    /// for i in 0..6 {
    ///     buf.push(i);
    /// }
    /// let vals: Vec<i32> = buf.iter().copied().collect();
    /// assert_eq!(vals, vec![2, 3, 4, 5]);
    /// ```
    #[inline]
    pub fn iter(&self) -> RingBufIter<'_, T, N> {
        RingBufIter {
            buf: self,
            offset: 0,
        }
    }
}

impl<T: Copy + Default, const N: usize> Default for RingBuf<T, N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over [`RingBuf`] elements from oldest to newest.
#[derive(Debug)]
pub struct RingBufIter<'a, T: Copy + Default, const N: usize> {
    buf: &'a RingBuf<T, N>,
    offset: usize,
}

impl<'a, T: Copy + Default, const N: usize> Iterator for RingBufIter<'a, T, N> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.buf.len {
            return None;
        }
        let idx = (self.buf.head + self.offset) % N;
        self.offset += 1;
        Some(&self.buf.buf[idx])
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.buf.len - self.offset;
        (remaining, Some(remaining))
    }
}

impl<T: Copy + Default, const N: usize> ExactSizeIterator for RingBufIter<'_, T, N> {}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec::Vec;

    use super::*;

    #[test]
    fn new_buffer_is_empty() {
        let buf = RingBuf::<f32, 4>::new();
        assert!(buf.is_empty());
        assert!(!buf.is_full());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), 4);
    }

    #[test]
    fn push_increases_len() {
        let mut buf = RingBuf::<f32, 4>::new();
        buf.push(1.0);
        assert_eq!(buf.len(), 1);
        buf.push(2.0);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn push_until_full() {
        let mut buf = RingBuf::<u8, 3>::new();
        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert!(buf.is_full());
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn push_overwrites_oldest_when_full() {
        let mut buf = RingBuf::<i32, 3>::new();
        buf.push(1);
        buf.push(2);
        buf.push(3);
        buf.push(4); // overwrites 1
        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.pop(), Some(3));
        assert_eq!(buf.pop(), Some(4));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn pop_returns_oldest_first() {
        let mut buf = RingBuf::<i32, 4>::new();
        buf.push(10);
        buf.push(20);
        buf.push(30);
        assert_eq!(buf.pop(), Some(10));
        assert_eq!(buf.pop(), Some(20));
        assert_eq!(buf.pop(), Some(30));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn pop_empty_returns_none() {
        let mut buf = RingBuf::<f32, 2>::new();
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn peek_returns_oldest() {
        let mut buf = RingBuf::<i32, 3>::new();
        assert_eq!(buf.peek(), None);
        buf.push(5);
        buf.push(10);
        assert_eq!(buf.peek(), Some(&5));
    }

    #[test]
    fn peek_does_not_remove() {
        let mut buf = RingBuf::<i32, 3>::new();
        buf.push(42);
        assert_eq!(buf.peek(), Some(&42));
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.peek(), Some(&42));
    }

    #[test]
    fn clear_resets_state() {
        let mut buf = RingBuf::<f32, 4>::new();
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn iter_oldest_to_newest() {
        let mut buf = RingBuf::<i32, 4>::new();
        buf.push(1);
        buf.push(2);
        buf.push(3);
        let vals: &[i32] = &buf.iter().copied().collect::<Vec<i32>>();
        assert_eq!(vals, &[1, 2, 3]);
    }

    #[test]
    fn iter_after_wraparound() {
        let mut buf = RingBuf::<i32, 3>::new();
        for i in 0..5 {
            buf.push(i);
        }
        // contains [2, 3, 4]
        let vals: &[i32] = &buf.iter().copied().collect::<Vec<i32>>();
        assert_eq!(vals, &[2, 3, 4]);
    }

    #[test]
    fn iter_empty() {
        let buf = RingBuf::<f32, 4>::new();
        assert_eq!(buf.iter().count(), 0);
    }

    #[test]
    fn iter_exact_size() {
        let mut buf = RingBuf::<i32, 4>::new();
        buf.push(1);
        buf.push(2);
        let iter = buf.iter();
        assert_eq!(iter.len(), 2);
    }

    #[test]
    fn interleaved_push_pop() {
        let mut buf = RingBuf::<i32, 3>::new();
        buf.push(1);
        buf.push(2);
        assert_eq!(buf.pop(), Some(1));
        buf.push(3);
        buf.push(4);
        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.pop(), Some(3));
        assert_eq!(buf.pop(), Some(4));
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn works_with_f32() {
        let mut buf = RingBuf::<f32, 2>::new();
        buf.push(0.5);
        buf.push(-0.5);
        assert_eq!(buf.pop(), Some(0.5));
        assert_eq!(buf.pop(), Some(-0.5));
    }

    #[test]
    fn works_with_i16() {
        let mut buf = RingBuf::<i16, 2>::new();
        buf.push(i16::MAX);
        buf.push(i16::MIN);
        assert_eq!(buf.pop(), Some(i16::MAX));
        assert_eq!(buf.pop(), Some(i16::MIN));
    }

    #[test]
    fn default_is_new() {
        let a = RingBuf::<f32, 4>::new();
        let b = RingBuf::<f32, 4>::default();
        assert_eq!(a.len(), b.len());
        assert_eq!(a.capacity(), b.capacity());
    }

    #[test]
    fn capacity_one() {
        let mut buf = RingBuf::<i32, 1>::new();
        buf.push(1);
        assert!(buf.is_full());
        buf.push(2); // overwrites
        assert_eq!(buf.pop(), Some(2));
        assert!(buf.is_empty());
    }
}
