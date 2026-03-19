use resonant_core::{FreqDomain, Signal, TimeDomain};

#[test]
fn construct_time_domain_signal() {
    let sig = Signal::from_samples(vec![0.0_f32, 0.5, 1.0, 0.5]);
    assert_eq!(sig.as_samples(), &[0.0, 0.5, 1.0, 0.5]);
}

#[test]
fn construct_freq_domain_signal() {
    let sig: Signal<Vec<f32>, FreqDomain> = Signal::new(vec![1.0, 0.0, 0.0]);
    assert_eq!(sig.data(), &vec![1.0, 0.0, 0.0]);
}

#[test]
fn into_inner_round_trip() {
    let original = vec![1.0_f32, 2.0, 3.0];
    let sig = Signal::from_samples(original.clone());
    let recovered = sig.into_inner();
    assert_eq!(recovered, original);
}

#[test]
fn map_domain_preserves_data() {
    let time = Signal::from_samples(vec![1.0_f32, 2.0, 3.0]);
    let freq: Signal<Vec<f32>, FreqDomain> = time.map_domain();
    assert_eq!(freq.data(), &vec![1.0, 2.0, 3.0]);
}

#[test]
fn map_domain_round_trip() {
    let original = vec![1.0_f32, 2.0];
    let time = Signal::from_samples(original.clone());
    let freq: Signal<Vec<f32>, FreqDomain> = time.map_domain();
    let back: Signal<Vec<f32>, TimeDomain> = freq.map_domain();
    assert_eq!(back.into_inner(), original);
}

#[test]
fn data_mut_modification() {
    let mut sig = Signal::from_samples(vec![0.0_f32; 3]);
    sig.data_mut()[1] = 42.0;
    assert_eq!(sig.as_samples(), &[0.0, 42.0, 0.0]);
}

#[test]
fn signals_with_same_data_are_equal() {
    let a = Signal::from_samples(vec![1.0_f32, 2.0]);
    let b = Signal::from_samples(vec![1.0_f32, 2.0]);
    assert_eq!(a, b);
}

#[test]
fn signals_with_different_data_are_not_equal() {
    let a = Signal::from_samples(vec![1.0_f32]);
    let b = Signal::from_samples(vec![2.0_f32]);
    assert_ne!(a, b);
}

#[test]
fn empty_signal() {
    let sig = Signal::from_samples(Vec::<f32>::new());
    assert!(sig.is_empty());
    assert_eq!(sig.len(), 0);
}

#[test]
fn len_matches_data() {
    let sig = Signal::from_samples(vec![0.0_f32; 1024]);
    assert_eq!(sig.len(), 1024);
    assert!(!sig.is_empty());
}

#[test]
fn clone_is_independent() {
    let sig = Signal::from_samples(vec![1.0_f32, 2.0]);
    let mut cloned = sig.clone();
    cloned.data_mut()[0] = 99.0;
    assert_eq!(sig.as_samples()[0], 1.0);
    assert_eq!(cloned.as_samples()[0], 99.0);
}

#[test]
fn works_with_array_storage() {
    let sig: Signal<[f32; 3], TimeDomain> = Signal::new([1.0, 2.0, 3.0]);
    assert_eq!(sig.len(), 3);
    assert_eq!(sig.as_samples(), &[1.0, 2.0, 3.0]);
}
