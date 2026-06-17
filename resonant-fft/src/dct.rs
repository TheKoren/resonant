//! Discrete Cosine Transform — types II and III.
//!
//! Direct O(N²) implementation suitable for the small sizes typical in audio
//! analysis (e.g. 13–40 coefficients for MFCCs). For large-N use cases an
//! FFT-based DCT would be faster, but is not needed here.
//!
//! Both transforms are `no_std` compatible and allocate nothing.

use core::f32::consts::PI;

// Brings f32::cos into scope on no_std via libm.
#[allow(unused_imports)]
use num_traits::float::Float as _;

use crate::FftError;

/// Computes the DCT type II (the "DCT") of `input`, writing results to `output`.
///
/// Formula: `X[k] = Σ_{n=0}^{N-1} x[n] · cos(π/N · (n + 0.5) · k)`
///
/// Both slices must have the same non-zero length.
///
/// # Errors
///
/// Returns [`FftError::Empty`] if `input` is empty, or
/// [`FftError::LengthMismatch`] if `input` and `output` lengths differ.
///
/// # Examples
///
/// ```
/// use resonant_fft::dct;
///
/// let input = [1.0_f32, 1.0, 1.0, 1.0];
/// let mut output = [0.0_f32; 4];
/// dct::dct_ii(&input, &mut output).unwrap();
/// // DC coefficient = sum of all samples
/// assert!((output[0] - 4.0).abs() < 1e-4);
/// ```
pub fn dct_ii(input: &[f32], output: &mut [f32]) -> Result<(), FftError> {
    validate(input, output)?;
    let n = input.len();
    let pi_over_n = PI / n as f32;

    for (k, out) in output.iter_mut().enumerate() {
        let mut sum = 0.0_f32;
        for (i, &x) in input.iter().enumerate() {
            sum += x * (pi_over_n * (i as f32 + 0.5) * k as f32).cos();
        }
        *out = sum;
    }
    Ok(())
}

/// Computes the DCT type III (the "inverse DCT") of `input`, writing to `output`.
///
/// Formula: `x[n] = 0.5·X[0] + Σ_{k=1}^{N-1} X[k] · cos(π/N · k · (n + 0.5))`
///
/// This is the inverse of [`dct_ii`] up to a scale factor of `N/2`:
/// `dct_iii(dct_ii(x)) = x · N/2`.
///
/// # Errors
///
/// Same as [`dct_ii`].
///
/// # Examples
///
/// ```
/// use resonant_fft::dct;
///
/// let original = [1.0_f32, 2.0, 3.0, 4.0];
/// let mut freq = [0.0_f32; 4];
/// let mut recovered = [0.0_f32; 4];
///
/// dct::dct_ii(&original, &mut freq).unwrap();
/// dct::dct_iii(&freq, &mut recovered).unwrap();
///
/// // Round-trip: recovered = original * N/2
/// let n = original.len() as f32;
/// for (r, &o) in recovered.iter().zip(&original) {
///     assert!((*r / (n / 2.0) - o).abs() < 1e-4);
/// }
/// ```
pub fn dct_iii(input: &[f32], output: &mut [f32]) -> Result<(), FftError> {
    validate(input, output)?;
    let n = input.len();
    let pi_over_n = PI / n as f32;
    let half_x0 = 0.5 * input[0];

    for (i, out) in output.iter_mut().enumerate() {
        let mut sum = half_x0;
        for (k, &x) in input.iter().enumerate().skip(1) {
            sum += x * (pi_over_n * k as f32 * (i as f32 + 0.5)).cos();
        }
        *out = sum;
    }
    Ok(())
}

fn validate(input: &[f32], output: &[f32]) -> Result<(), FftError> {
    if input.is_empty() {
        return Err(FftError::Empty);
    }
    if input.len() != output.len() {
        return Err(FftError::LengthMismatch {
            input: input.len(),
            output: output.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct_ii_dc_signal() {
        let input = [1.0_f32; 4];
        let mut output = [0.0_f32; 4];
        dct_ii(&input, &mut output).ok();
        // DC coefficient = N
        assert!((output[0] - 4.0).abs() < 1e-4);
        // AC coefficients ≈ 0
        for &v in &output[1..] {
            assert!(v.abs() < 1e-4, "AC bin not zero: {v}");
        }
    }

    #[test]
    fn dct_ii_single_element() {
        let input = [3.5_f32];
        let mut output = [0.0_f32; 1];
        dct_ii(&input, &mut output).ok();
        assert!((output[0] - 3.5).abs() < 1e-4);
    }

    #[test]
    fn dct_ii_empty_returns_error() {
        let mut output = [0.0_f32; 0];
        let err = dct_ii(&[], &mut output);
        assert_eq!(err, Err(FftError::Empty));
    }

    #[test]
    fn dct_ii_length_mismatch() {
        let input = [1.0_f32; 4];
        let mut output = [0.0_f32; 3];
        let err = dct_ii(&input, &mut output);
        assert_eq!(
            err,
            Err(FftError::LengthMismatch {
                input: 4,
                output: 3
            })
        );
    }

    #[test]
    fn dct_iii_single_element() {
        let input = [2.0_f32];
        let mut output = [0.0_f32; 1];
        dct_iii(&input, &mut output).ok();
        // half_x0 = 1.0
        assert!((output[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn dct_iii_empty_returns_error() {
        let mut output = [0.0_f32; 0];
        let err = dct_iii(&[], &mut output);
        assert_eq!(err, Err(FftError::Empty));
    }

    #[test]
    fn round_trip_dct_ii_iii() {
        let original = [1.0_f32, 2.0, 3.0, 4.0];
        let mut freq = [0.0_f32; 4];
        let mut recovered = [0.0_f32; 4];

        dct_ii(&original, &mut freq).ok();
        dct_iii(&freq, &mut recovered).ok();

        // dct_iii(dct_ii(x)) = x * N/2
        let scale = original.len() as f32 / 2.0;
        for (r, &o) in recovered.iter().zip(&original) {
            assert!(
                (r / scale - o).abs() < 1e-4,
                "round-trip failed: {r}/{scale} != {o}"
            );
        }
    }

    #[test]
    fn round_trip_larger() {
        let original: [f32; 8] = [0.1, -0.5, 0.3, 0.8, -0.2, 0.0, 0.6, -0.4];
        let mut freq = [0.0_f32; 8];
        let mut recovered = [0.0_f32; 8];

        dct_ii(&original, &mut freq).ok();
        dct_iii(&freq, &mut recovered).ok();

        let scale = 8.0_f32 / 2.0;
        for (r, &o) in recovered.iter().zip(&original) {
            assert!(
                (r / scale - o).abs() < 1e-4,
                "round-trip failed: {r}/{scale} != {o}"
            );
        }
    }

    #[test]
    fn dct_ii_known_two_point() {
        // x = [1, -1], N=2
        // X[k] = Σ x[n]*cos(π/N*(n+0.5)*k)
        // X[0] = 1*cos(0) + (-1)*cos(0) = 0
        // X[1] = cos(π/4) + (-1)*cos(3π/4) = √2/2 + √2/2 = √2
        let input = [1.0_f32, -1.0];
        let mut output = [0.0_f32; 2];
        dct_ii(&input, &mut output).ok();

        let sqrt2 = core::f32::consts::SQRT_2;
        assert!(
            output[0].abs() < 1e-4,
            "X[0] should be 0, got {}",
            output[0]
        );
        assert!(
            (output[1] - sqrt2).abs() < 1e-4,
            "X[1] should be √2, got {}",
            output[1]
        );
    }

    #[test]
    fn dct_ii_linearity() {
        // DCT(a*x + b*y) = a*DCT(x) + b*DCT(y)
        let x = [1.0_f32, 2.0, 3.0, 4.0];
        let y = [4.0_f32, 3.0, 2.0, 1.0];
        let a = 2.0_f32;
        let b = -0.5_f32;

        let mut dct_x = [0.0_f32; 4];
        let mut dct_y = [0.0_f32; 4];
        dct_ii(&x, &mut dct_x).ok();
        dct_ii(&y, &mut dct_y).ok();

        let combined: [f32; 4] = core::array::from_fn(|i| a * x[i] + b * y[i]);
        let mut dct_combined = [0.0_f32; 4];
        dct_ii(&combined, &mut dct_combined).ok();

        for i in 0..4 {
            let expected = a * dct_x[i] + b * dct_y[i];
            assert!(
                (dct_combined[i] - expected).abs() < 1e-3,
                "linearity failed at bin {i}: {} != {expected}",
                dct_combined[i]
            );
        }
    }
}
