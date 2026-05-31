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

    /// Returns the contents as two contiguous slices, oldest element first.
    ///
    /// The concatenation of the two slices is always the full sequence from
    /// oldest to newest. The second slice is empty when the buffer has not
    /// wrapped around.
    ///
    /// # Examples
    ///
    /// ```
    /// use resonant_core::RingBuf;
    ///
    /// let mut buf = RingBuf::<i32, 4>::new();
    /// buf.push(1); buf.push(2); buf.push(3); buf.push(4);
    /// buf.pop(); // discard 1, head advances
    /// buf.push(5); // wraps around
    ///
    /// let (a, b) = buf.as_slices();
    /// let combined: Vec<i32> = a.iter().chain(b).copied().collect();
    /// assert_eq!(combined, vec![2, 3, 4, 5]);
    /// ```
    #[inline]
    #[must_use]
    pub fn as_slices(&self) -> (&[T], &[T]) {
        ops::as_slices(&self.buf, self.head, self.len)
    }

    /// Removes and returns all elements from oldest to newest.
    ///
    /// The buffer is empty after this call.
    ///
    /// # Examples
    ///
    /// ```
    /// use resonant_core::RingBuf;
    ///
    /// let mut buf = RingBuf::<i32, 4>::new();
    /// buf.push(10); buf.push(20); buf.push(30);
    /// let drained: Vec<i32> = buf.drain().collect();
    /// assert_eq!(drained, vec![10, 20, 30]);
    /// assert!(buf.is_empty());
    /// ```
    #[inline]
    pub fn drain(&mut self) -> RingBufDrain<'_, T, N> {
        RingBufDrain { buf: self }
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
        ops::iter_next(&self.buf.buf, self.buf.head, &mut self.offset, self.buf.len)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        ops::size_hint(self.buf.len, self.offset)
    }
}

impl<T: Copy + Default, const N: usize> ExactSizeIterator for RingBufIter<'_, T, N> {}

/// Draining iterator produced by [`RingBuf::drain`].
///
/// Yields elements from oldest to newest, removing each from the buffer.
#[derive(Debug)]
pub struct RingBufDrain<'a, T: Copy + Default, const N: usize> {
    buf: &'a mut RingBuf<T, N>,
}

impl<T: Copy + Default, const N: usize> Iterator for RingBufDrain<'_, T, N> {
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

impl<T: Copy + Default, const N: usize> ExactSizeIterator for RingBufDrain<'_, T, N> {}

/// Shared ring-buffer algorithm operating on a plain slice.
///
/// Both [`RingBuf`] and [`HeapRingBuf`](crate::HeapRingBuf) delegate every
/// method to these free functions. The only structural difference between the
/// two types is their backing storage (`[T; N]` vs `Vec<T>`); the algorithm
/// is identical.
///
/// All functions accept the storage as `&[T]` / `&mut [T]` and derive the
/// capacity from `buf.len()`, so no separate capacity parameter is needed.
pub(crate) mod ops {
    /// Appends `value`, overwriting the oldest element when `len == buf.len()`.
    #[inline]
    pub(crate) fn push<T: Copy>(buf: &mut [T], head: &mut usize, len: &mut usize, value: T) {
        let cap = buf.len();
        let write_idx = (*head + *len) % cap;
        buf[write_idx] = value;
        if *len == cap {
            *head = (*head + 1) % cap;
        } else {
            *len += 1;
        }
    }

    /// Removes and returns the oldest element, or `None` if empty.
    #[inline]
    pub(crate) fn pop<T: Copy>(buf: &[T], head: &mut usize, len: &mut usize) -> Option<T> {
        if *len == 0 {
            return None;
        }
        let value = buf[*head];
        *head = (*head + 1) % buf.len();
        *len -= 1;
        Some(value)
    }

    /// Returns a reference to the oldest element without removing it.
    #[inline]
    pub(crate) fn peek<T>(buf: &[T], head: usize, len: usize) -> Option<&T> {
        if len == 0 {
            return None;
        }
        Some(&buf[head])
    }

    /// Returns the logical contents as two contiguous slices (oldest-first).
    #[inline]
    pub(crate) fn as_slices<T>(buf: &[T], head: usize, len: usize) -> (&[T], &[T]) {
        let cap = buf.len();
        let tail = head + len;
        if tail <= cap {
            (&buf[head..tail], &[])
        } else {
            (&buf[head..cap], &buf[..tail - cap])
        }
    }

    /// Advances a by-reference iterator one step.
    ///
    /// `offset` tracks how many elements have already been yielded. Returns
    /// `None` when `offset >= len`.
    #[inline]
    pub(crate) fn iter_next<'a, T>(
        buf: &'a [T],
        head: usize,
        offset: &mut usize,
        len: usize,
    ) -> Option<&'a T> {
        if *offset >= len {
            return None;
        }
        let idx = (head + *offset) % buf.len();
        *offset += 1;
        Some(&buf[idx])
    }

    /// Returns the `(lower, upper)` size hint for an iterator that has yielded
    /// `offset` elements out of a buffer currently holding `len`.
    #[inline]
    pub(crate) fn size_hint(len: usize, offset: usize) -> (usize, Option<usize>) {
        let remaining = len - offset;
        (remaining, Some(remaining))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec;
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

    #[test]
    fn as_slices_no_wraparound() {
        let mut buf = RingBuf::<i32, 4>::new();
        buf.push(1);
        buf.push(2);
        buf.push(3);
        let (a, b) = buf.as_slices();
        assert_eq!(a, &[1, 2, 3]);
        assert!(b.is_empty());
    }

    #[test]
    fn as_slices_wraparound() {
        let mut buf = RingBuf::<i32, 4>::new();
        buf.push(1);
        buf.push(2);
        buf.push(3);
        buf.push(4);
        let _ = buf.pop(); // head → 1, buf = [2,3,4]
        buf.push(5); // wraps: buf internal = [5,2,3,4] with head=1, len=4
        let (a, b) = buf.as_slices();
        let combined: Vec<i32> = a.iter().chain(b).copied().collect();
        assert_eq!(combined, vec![2, 3, 4, 5]);
    }

    #[test]
    fn as_slices_empty() {
        let buf = RingBuf::<i32, 4>::new();
        let (a, b) = buf.as_slices();
        assert!(a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn as_slices_full_no_wrap() {
        let mut buf = RingBuf::<i32, 3>::new();
        buf.push(7);
        buf.push(8);
        buf.push(9);
        let (a, b) = buf.as_slices();
        assert_eq!(a, &[7, 8, 9]);
        assert!(b.is_empty());
    }

    #[test]
    fn drain_yields_oldest_first() {
        let mut buf = RingBuf::<i32, 4>::new();
        buf.push(10);
        buf.push(20);
        buf.push(30);
        let drained: Vec<i32> = buf.drain().collect();
        assert_eq!(drained, vec![10, 20, 30]);
    }

    #[test]
    fn drain_empties_buffer() {
        let mut buf = RingBuf::<i32, 4>::new();
        buf.push(1);
        buf.push(2);
        let _ = buf.drain().count();
        assert!(buf.is_empty());
    }

    #[test]
    fn drain_empty_buffer() {
        let mut buf = RingBuf::<f32, 4>::new();
        assert_eq!(buf.drain().count(), 0);
    }

    #[test]
    fn drain_exact_size() {
        let mut buf = RingBuf::<i32, 4>::new();
        buf.push(1);
        buf.push(2);
        buf.push(3);
        let drain = buf.drain();
        assert_eq!(drain.len(), 3);
    }

    #[test]
    fn drain_after_wraparound() {
        let mut buf = RingBuf::<i32, 3>::new();
        for i in 0..5 {
            buf.push(i);
        }
        // contains [2, 3, 4]
        let drained: Vec<i32> = buf.drain().collect();
        assert_eq!(drained, vec![2, 3, 4]);
        assert!(buf.is_empty());
    }
}
