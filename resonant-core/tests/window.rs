use resonant_core::window;

#[test]
fn hann_applied_to_dc_signal() {
    let mut buf = [1.0_f32; 8];
    window::hann(&mut buf);

    // endpoints zero, interior positive
    assert!((buf[0]).abs() < 1e-6);
    assert!((buf[7]).abs() < 1e-6);
    for &v in &buf[1..7] {
        assert!(v > 0.0);
    }
}

#[test]
fn hamming_applied_to_dc_signal() {
    let mut buf = [1.0_f32; 8];
    window::hamming(&mut buf);

    assert!((buf[0] - 0.08).abs() < 1e-6);
    assert!((buf[7] - 0.08).abs() < 1e-6);
}

#[test]
fn blackman_applied_to_dc_signal() {
    let mut buf = [1.0_f32; 8];
    window::blackman(&mut buf);

    assert!(buf[0].abs() < 0.01);
    assert!(buf[7].abs() < 0.01);
    // peak near centre
    assert!(buf[4] > 0.5);
}

#[test]
fn rectangular_preserves_signal() {
    let original = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let mut buf = original;
    window::rectangular(&mut buf);
    assert_eq!(buf, original);
}

#[test]
fn bartlett_linear_taper() {
    let mut buf = [1.0_f32; 9];
    window::bartlett(&mut buf);

    // should increase linearly to centre then decrease
    for i in 0..4 {
        assert!(buf[i] < buf[i + 1], "expected increasing at index {i}");
    }
    for i in 4..8 {
        assert!(buf[i] > buf[i + 1], "expected decreasing at index {i}");
    }
}

#[test]
fn window_scales_signal_amplitude() {
    let mut buf = [2.0_f32; 8];
    window::hann(&mut buf);

    // all values should be <= 2.0 (original amplitude)
    for &v in &buf {
        assert!(v <= 2.0 + 1e-6);
        assert!(v >= 0.0 - 1e-6);
    }
}

#[test]
fn all_windows_symmetric_on_power_of_two() {
    let fns: &[fn(&mut [f32])] = &[
        window::hann,
        window::hamming,
        window::blackman,
        window::bartlett,
    ];
    for apply in fns {
        let mut buf = [1.0_f32; 32];
        apply(&mut buf);
        for i in 0..16 {
            assert!(
                (buf[i] - buf[31 - i]).abs() < 1e-5,
                "asymmetry at index {i}"
            );
        }
    }
}
