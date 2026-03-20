use resonant_core::RingBuf;

#[test]
fn push_and_pop_fifo_order() {
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
fn overwrite_oldest_when_full() {
    let mut buf = RingBuf::<i32, 3>::new();
    buf.push(1);
    buf.push(2);
    buf.push(3);
    buf.push(4);
    buf.push(5);
    // oldest two (1, 2) overwritten; contains [3, 4, 5]
    assert_eq!(buf.pop(), Some(3));
    assert_eq!(buf.pop(), Some(4));
    assert_eq!(buf.pop(), Some(5));
}

#[test]
fn iter_preserves_order_after_wraparound() {
    let mut buf = RingBuf::<i32, 3>::new();
    for i in 0..7 {
        buf.push(i);
    }
    // contains [4, 5, 6]
    let vals: Vec<i32> = buf.iter().copied().collect();
    assert_eq!(vals, vec![4, 5, 6]);
}

#[test]
fn interleaved_push_pop_stress() {
    let mut buf = RingBuf::<i32, 4>::new();
    for i in 0..100 {
        buf.push(i);
        if i % 3 == 0 {
            let _ = buf.pop();
        }
    }
    // just verify it doesn't panic and length is sane
    assert!(buf.len() <= 4);
    assert!(!buf.is_empty());
}

#[test]
fn capacity_one_buffer() {
    let mut buf = RingBuf::<f32, 1>::new();
    buf.push(1.0);
    assert!(buf.is_full());
    assert_eq!(buf.peek(), Some(&1.0));
    buf.push(2.0);
    assert_eq!(buf.pop(), Some(2.0));
    assert!(buf.is_empty());
}

#[test]
fn clear_allows_reuse() {
    let mut buf = RingBuf::<i32, 3>::new();
    buf.push(1);
    buf.push(2);
    buf.push(3);
    buf.clear();

    assert!(buf.is_empty());
    buf.push(10);
    buf.push(20);
    assert_eq!(buf.pop(), Some(10));
    assert_eq!(buf.pop(), Some(20));
}

#[test]
fn iter_exact_size_hint() {
    let mut buf = RingBuf::<i32, 8>::new();
    buf.push(1);
    buf.push(2);
    buf.push(3);
    let iter = buf.iter();
    assert_eq!(iter.len(), 3);
    assert_eq!(iter.size_hint(), (3, Some(3)));
}

#[test]
fn default_trait() {
    let buf = RingBuf::<f32, 4>::default();
    assert!(buf.is_empty());
    assert_eq!(buf.capacity(), 4);
}
