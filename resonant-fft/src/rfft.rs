//! Real-valued FFT — exploits conjugate symmetry of real inputs.
//!
//! For N real samples the DFT has conjugate symmetry: `X[N-k] = X*[k]`. Only
//! N/2+1 unique complex bins exist. This module computes exactly those bins at
//! roughly half the cost of a full complex FFT.
//!
//! Both functions are `no_std`, `no_alloc` — they operate entirely on
//! caller-provided slices. For an ergonomic `Signal`-based API see
//! [`SignalRfftExt`](crate::SignalRfftExt) (requires `alloc` feature).
//!
//! # Algorithm
//!
//! Forward: pack N real samples as N/2 complex values `z[k] = x[2k] + j·x[2k+1]`,
//! run the N/2-point complex FFT, then apply a split-radix post-processing step to
//! recover the N/2+1 unique bins.
//!
//! Inverse: reverse the post-processing step to rebuild the N/2-point complex
//! sequence, run the complex IFFT, then deinterleave into N real samples. The
//! output buffer is temporarily reinterpreted as `Complex<f32>` scratch — no
//! extra allocation is needed.

use core::f32::consts::PI;

use num_complex::Complex;
// Brings f32::cos/sin into scope on no_std targets via libm.
#[allow(unused_imports)]
use num_traits::float::Float as _;

use crate::FftError;

/// Computes the real-valued forward FFT of `input`, writing N/2+1 complex bins
/// into `out`.
///
/// `input.len()` must be a non-zero power of two.
/// `out.len()` must equal `input.len() / 2 + 1`.
///
/// The output is not normalised (same convention as [`radix2::fft`](crate::fft)).
/// Apply a `1/N` scale manually if you need a unitary transform.
///
/// # Errors
///
/// - [`FftError::Empty`] — `input` is empty.
/// - [`FftError::NotPowerOfTwo`] — `input.len()` is not a power of two.
/// - [`FftError::LengthMismatch`] — `out.len() != input.len() / 2 + 1`.
///
/// # Examples
///
/// ```
/// use resonant_fft::{rfft, Complex};
///
/// let input = [1.0_f32, 0.0, -1.0, 0.0];
/// let mut out = [Complex::new(0.0_f32, 0.0); 3]; // N/2 + 1 = 3
/// rfft(&input, &mut out).unwrap();
/// // DC bin: sum of all = 0
/// assert!(out[0].re.abs() < 1e-5);
/// ```
pub fn rfft(input: &[f32], out: &mut [Complex<f32>]) -> Result<(), FftError> {
    let n = input.len();
    if n == 0 {
        return Err(FftError::Empty);
    }
    if !n.is_power_of_two() {
        return Err(FftError::NotPowerOfTwo(n));
    }
    let m = n / 2;
    if out.len() != m + 1 {
        return Err(FftError::LengthMismatch {
            input: n,
            output: out.len(),
        });
    }

    // Pack: z[k] = x[2k] + j·x[2k+1]
    for k in 0..m {
        out[k] = Complex::new(input[2 * k], input[2 * k + 1]);
    }

    // M-point complex FFT in-place
    crate::radix2::fft(&mut out[..m])?;

    // Post-processing: recover N/2+1 bins from M-point output.
    //
    // For k = 0..M:
    //   A[k] = (Z[k] + Z*[M-k]) / 2
    //   B[k] = -j * (Z[k] - Z*[M-k]) / 2
    //   X[k] = A[k] + e^{-j·2π·k/N} · B[k]
    //
    // k=0 and k=M both reference Z[0]: handle first to avoid aliasing.
    // All other k are processed in pairs (k, M-k) so we read both Z values
    // before writing either output slot.

    let z0 = out[0]; // save before overwriting
    out[m] = Complex::new(z0.re - z0.im, 0.0); // X[N/2]: real-valued
    out[0] = Complex::new(z0.re + z0.im, 0.0); // X[0]:   real-valued

    // k = 1 .. floor(M/2), processing symmetric pairs simultaneously
    let half = m / 2;
    for k in 1..=half {
        let zk = out[k];
        let zmk = out[m - k];

        let theta = -2.0 * PI * k as f32 / n as f32;
        let w = Complex::new(theta.cos(), theta.sin()); // e^{-j·2πk/N}

        let a = (zk + zmk.conj()) * 0.5;
        let b = (zk - zmk.conj()) * Complex::new(0.0, -0.5); // -j * diff / 2

        out[k] = a + w * b;

        // Symmetric counterpart at k' = M-k (may equal k when M is even and k = M/2)
        if m - k != k {
            let theta_mk = -2.0 * PI * (m - k) as f32 / n as f32;
            let w_mk = Complex::new(theta_mk.cos(), theta_mk.sin());
            let a_mk = (zmk + zk.conj()) * 0.5;
            let b_mk = (zmk - zk.conj()) * Complex::new(0.0, -0.5);
            out[m - k] = a_mk + w_mk * b_mk;
        }
    }

    Ok(())
}

/// Computes the inverse real-valued FFT.
///
/// `input` must contain the N/2+1 complex bins produced by [`rfft`].
/// `out` must have length `(input.len() - 1) * 2`.
///
/// The output is scaled by `1/N` so that `irfft(rfft(x)) ≈ x` (round-trip
/// fidelity within floating-point precision).
///
/// The output buffer is temporarily reinterpreted as `Complex<f32>` scratch
/// to avoid any heap allocation.
///
/// # Errors
///
/// - [`FftError::Empty`] — fewer than 2 bins supplied.
/// - [`FftError::NotPowerOfTwo`] — implied output length is not a power of two.
/// - [`FftError::LengthMismatch`] — `out.len() != (input.len() - 1) * 2`.
///
/// # Examples
///
/// ```
/// use resonant_fft::{rfft, irfft, Complex};
///
/// let original = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let mut bins = [Complex::new(0.0_f32, 0.0); 5]; // N/2+1 = 5
/// rfft(&original, &mut bins).unwrap();
///
/// let mut recovered = [0.0_f32; 8];
/// irfft(&bins, &mut recovered).unwrap();
/// for (a, b) in original.iter().zip(recovered.iter()) {
///     assert!((a - b).abs() < 1e-4, "{a} vs {b}");
/// }
/// ```
pub fn irfft(input: &[Complex<f32>], out: &mut [f32]) -> Result<(), FftError> {
    if input.len() < 2 {
        return Err(FftError::Empty);
    }
    let m = input.len() - 1; // N/2
    let n = m * 2; // N
    if out.len() != n {
        return Err(FftError::LengthMismatch {
            input: input.len(),
            output: out.len(),
        });
    }
    if !n.is_power_of_two() {
        return Err(FftError::NotPowerOfTwo(n));
    }

    // Reinterpret `out` as Complex<f32> scratch.
    //
    // SAFETY: Complex<f32> is #[repr(C)] with layout [re: f32, im: f32],
    // identical alignment (4) and size (8 = 2×4) to [f32; 2]. `out` has
    // n = 2*m f32 elements, which maps to exactly m Complex<f32> elements.
    // We hold only `scratch` for the duration of the function and do not
    // alias `out` simultaneously.
    let scratch: &mut [Complex<f32>] =
        unsafe { core::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<Complex<f32>>(), m) };

    // Inverse post-processing: recover Z[k] from X[k] = input[k].
    //
    // For k = 0..M-1:
    //   Z[k] = (X[k] + X*[M-k]) / 2 + j · W_N^k · (X[k] - X*[M-k]) / 2
    // where W_N^k = e^{j·2πk/N}.
    //
    // k=0 is a special case: X[0] = Z0.re + Z0.im, X[M] = Z0.re - Z0.im (both real).
    let x0 = input[0].re;
    let xm = input[m].re;
    scratch[0] = Complex::new((x0 + xm) * 0.5, (x0 - xm) * 0.5);

    for k in 1..m {
        let xk = input[k];
        let xmk_conj = input[m - k].conj(); // X*[M-k]

        let theta = 2.0 * PI * k as f32 / n as f32; // W_N^k = e^{j·2πk/N}
        let w = Complex::new(theta.cos(), theta.sin());

        let sum = (xk + xmk_conj) * 0.5;
        let diff = (xk - xmk_conj) * 0.5;
        scratch[k] = sum + Complex::new(0.0, 1.0) * w * diff;
    }

    // M-point inverse FFT (applies 1/M scaling internally)
    crate::radix2::ifft(scratch)?;

    // After IFFT, scratch[k] = x[2k] + j·x[2k+1].
    // Because scratch and out share the same memory and Complex<f32> is
    // laid out as [re, im], out[2k] = x[2k] and out[2k+1] = x[2k+1] already.

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f32, im: f32) -> Complex<f32> {
        Complex::new(re, im)
    }


    #[test]
    fn rfft_empty_returns_error() {
        let mut out = [c(0.0, 0.0); 1];
        assert_eq!(rfft(&[], &mut out), Err(FftError::Empty));
    }

    #[test]
    fn rfft_non_power_of_two_returns_error() {
        let input = [0.0_f32; 6];
        let mut out = [c(0.0, 0.0); 4];
        assert_eq!(rfft(&input, &mut out), Err(FftError::NotPowerOfTwo(6)));
    }

    #[test]
    fn rfft_wrong_output_length_returns_error() {
        let input = [0.0_f32; 8];
        let mut out = [c(0.0, 0.0); 3]; // should be 5
        assert!(matches!(
            rfft(&input, &mut out),
            Err(FftError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn irfft_empty_returns_error() {
        let mut out = [0.0_f32; 0];
        assert_eq!(irfft(&[], &mut out), Err(FftError::Empty));
        assert_eq!(irfft(&[c(0.0, 0.0)], &mut out), Err(FftError::Empty));
    }

    #[test]
    fn irfft_wrong_output_length_returns_error() {
        let bins = [c(0.0, 0.0); 5]; // implies N=8
        let mut out = [0.0_f32; 6]; // should be 8
        assert!(matches!(
            irfft(&bins, &mut out),
            Err(FftError::LengthMismatch { .. })
        ));
    }


    #[test]
    fn roundtrip_8point() {
        let original = [0.1_f32, -0.3, 0.5, -0.7, 0.9, -0.2, 0.4, -0.6];
        let mut bins = [c(0.0, 0.0); 5];
        rfft(&original, &mut bins).unwrap();

        let mut recovered = [0.0_f32; 8];
        irfft(&bins, &mut recovered).unwrap();

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-4, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn roundtrip_4point() {
        let original = [1.0_f32, 2.0, 3.0, 4.0];
        let mut bins = [c(0.0, 0.0); 3];
        rfft(&original, &mut bins).unwrap();

        let mut recovered = [0.0_f32; 4];
        irfft(&bins, &mut recovered).unwrap();

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-4, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn roundtrip_2point() {
        let original = [3.0_f32, -1.0];
        let mut bins = [c(0.0, 0.0); 2];
        rfft(&original, &mut bins).unwrap();

        let mut recovered = [0.0_f32; 2];
        irfft(&bins, &mut recovered).unwrap();

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn roundtrip_256point() {
        let original: [f32; 256] = core::array::from_fn(|i| (i as f32 * 0.1).sin());
        let mut bins = [c(0.0, 0.0); 129];
        rfft(&original, &mut bins).unwrap();

        let mut recovered = [0.0_f32; 256];
        irfft(&bins, &mut recovered).unwrap();

        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-3, "mismatch: {a} vs {b}");
        }
    }


    #[test]
    fn rfft_matches_complex_fft_dc_bins() {
        // rfft output bins should match the first N/2+1 bins of a full complex FFT
        let input = [1.0_f32, -0.5, 0.3, -0.8, 0.6, -0.2, 0.9, -0.4];
        let n = input.len();
        let m = n / 2;

        // rfft
        let mut rfft_out = [c(0.0, 0.0); 5];
        rfft(&input, &mut rfft_out).unwrap();

        // full complex FFT
        let mut cfft_buf: [Complex<f32>; 8] = core::array::from_fn(|i| Complex::new(input[i], 0.0));
        crate::radix2::fft(&mut cfft_buf).unwrap();

        for k in 0..=m {
            let diff = (rfft_out[k] - cfft_buf[k]).norm();
            assert!(
                diff < 1e-4,
                "bin {k}: rfft={:?} cfft={:?}",
                rfft_out[k],
                cfft_buf[k]
            );
        }
    }

    #[test]
    fn rfft_dc_signal() {
        // DC: all ones → bin[0] = N, all others ≈ 0
        let input = [1.0_f32; 8];
        let mut out = [c(0.0, 0.0); 5];
        rfft(&input, &mut out).unwrap();

        assert!((out[0].re - 8.0).abs() < 1e-4, "DC bin: {}", out[0].re);
        assert!(out[0].im.abs() < 1e-4);
        for b in &out[1..] {
            assert!(b.norm() < 1e-4, "non-DC bin should be ~0: {b:?}");
        }
    }

    #[test]
    fn rfft_single_frequency() {
        // A pure cosine at bin k=1 should have energy only at bins 1 (and N-1 which
        // maps to M-1 in the half-spectrum). Bin 0, Nyquist, and others ≈ 0.
        let n = 8usize;
        let input: [f32; 8] =
            core::array::from_fn(|i| (2.0 * PI * 1.0 * i as f32 / n as f32).cos());
        let mut out = [c(0.0, 0.0); 5];
        rfft(&input, &mut out).unwrap();

        // bin 0 (DC) and bin 4 (Nyquist) should be ~0
        assert!(out[0].norm() < 1e-3, "DC should be ~0: {:?}", out[0]);
        assert!(out[4].norm() < 1e-3, "Nyquist should be ~0: {:?}", out[4]);
        // bin 1 should have the energy (magnitude ≈ N/2 = 4 for cosine)
        assert!(
            out[1].norm() > 3.0,
            "bin 1 should have energy: {:?}",
            out[1]
        );
        // bins 2 and 3 should be ~0
        assert!(out[2].norm() < 1e-3, "bin 2 should be ~0: {:?}", out[2]);
        assert!(out[3].norm() < 1e-3, "bin 3 should be ~0: {:?}", out[3]);
    }

    #[test]
    fn rfft_nyquist_bin_is_real() {
        // For real input the Nyquist bin (X[N/2]) is always real-valued
        let input = [0.1_f32, -0.3, 0.5, -0.7, 0.9, -0.2, 0.4, -0.6];
        let mut out = [c(0.0, 0.0); 5];
        rfft(&input, &mut out).unwrap();

        assert!(
            out[4].im.abs() < 1e-5,
            "Nyquist bin must be real: {:?}",
            out[4]
        );
        assert!(out[0].im.abs() < 1e-5, "DC bin must be real: {:?}", out[0]);
    }

    #[test]
    fn rfft_conjugate_symmetry_matches_full_fft() {
        // For real input: rfft bins should satisfy X[k] = conj(full_fft[N-k])
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = input.len();
        let m = n / 2;

        let mut rfft_out = [c(0.0, 0.0); 5];
        rfft(&input, &mut rfft_out).unwrap();

        let mut cfft: [Complex<f32>; 8] = core::array::from_fn(|i| Complex::new(input[i], 0.0));
        crate::radix2::fft(&mut cfft).unwrap();

        // Check X[k] = full_fft[k] for k = 0..=N/2
        for k in 0..=m {
            let diff = (rfft_out[k] - cfft[k]).norm();
            assert!(
                diff < 1e-4,
                "bin {k} mismatch: rfft={:?} full={:?}",
                rfft_out[k],
                cfft[k]
            );
        }

        // Check conjugate symmetry: full_fft[N-k] = conj(full_fft[k])
        for k in 1..m {
            let sym = (cfft[n - k] - cfft[k].conj()).norm();
            assert!(sym < 1e-4, "conjugate symmetry violated at k={k}");
        }
    }
}
