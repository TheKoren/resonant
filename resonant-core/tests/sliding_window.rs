#![cfg(feature = "alloc")]

use resonant_core::SlidingWindow;

#[test]
fn basic_overlapping_frames() {
    let mut sw = SlidingWindow::new(4, 2);
    sw.push_slice(&[1.0_f32, 2.0, 3.0, 4.0]);

    assert!(sw.is_ready());
    assert_eq!(sw.current_frame(), Some([1.0, 2.0, 3.0, 4.0].as_slice()));

    sw.advance();
    sw.push_slice(&[5.0, 6.0]);
    assert_eq!(sw.current_frame(), Some([3.0, 4.0, 5.0, 6.0].as_slice()));
}

#[test]
fn non_overlapping_hop() {
    let mut sw = SlidingWindow::new(3, 3);
    sw.push_slice(&[10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0]);

    assert_eq!(sw.current_frame(), Some([10.0, 20.0, 30.0].as_slice()));
    sw.advance();
    assert_eq!(sw.current_frame(), Some([40.0, 50.0, 60.0].as_slice()));
    sw.advance();
    assert!(!sw.is_ready());
}

#[test]
fn single_sample_push_workflow() {
    let mut sw = SlidingWindow::new(3, 1);
    for &s in &[1.0_f32, 2.0, 3.0] {
        sw.push(s);
    }
    assert_eq!(sw.current_frame(), Some([1.0, 2.0, 3.0].as_slice()));

    sw.advance();
    sw.push(4.0);
    assert_eq!(sw.current_frame(), Some([2.0, 3.0, 4.0].as_slice()));
}

#[test]
fn large_hop_skips_past_buffer() {
    let mut sw = SlidingWindow::new(4, 10);
    sw.push_slice(&[1.0_f32, 2.0]);
    sw.advance();
    assert!(sw.is_empty());
}

#[test]
fn clear_and_reuse() {
    let mut sw = SlidingWindow::new(2, 1);
    sw.push_slice(&[1.0_f32, 2.0]);
    assert!(sw.is_ready());

    sw.clear();
    assert!(!sw.is_ready());
    assert!(sw.is_empty());

    sw.push_slice(&[3.0, 4.0]);
    assert_eq!(sw.current_frame(), Some([3.0, 4.0].as_slice()));
}

#[test]
fn streaming_multiple_frames() {
    let mut sw = SlidingWindow::new(4, 2);
    let signal = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut frames = Vec::new();

    sw.push_slice(&signal);
    while sw.is_ready() {
        let frame: Vec<f32> = sw.current_frame().map(|f| f.to_vec()).unwrap_or_default();
        frames.push(frame);
        sw.advance();
    }

    assert_eq!(frames.len(), 3);
    assert_eq!(frames[0], vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(frames[1], vec![3.0, 4.0, 5.0, 6.0]);
    assert_eq!(frames[2], vec![5.0, 6.0, 7.0, 8.0]);
}
