//! no_std smoke test for resonant-core, resonant-fft, and resonant-filters.
//!
//! This binary is compiled and *run* in CI under QEMU. A zero exit code means
//! all operations completed without panicking on a bare-metal Cortex-M4 target
//! (`thumbv7em-none-eabihf`). Any panic exits QEMU with code 1.
//!
//! ## Running locally with QEMU
//!
//! ```sh
//! cargo build --release --target thumbv7em-none-eabihf
//! qemu-system-arm -machine mps2-an386 -nographic \
//!   -semihosting-config enable=on,target=native \
//!   -kernel target/thumbv7em-none-eabihf/release/resonant-no-std-check
//! echo "Exit code: $?"
//! ```

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_semihosting as _;

use cortex_m_semihosting::debug;
use resonant_core::{signal::Signal, window, RingBuf};
use resonant_fft::radix2;
use resonant_filters::biquad::{Biquad, BiquadCoeffs};

#[entry]
fn main() -> ! {
    // --- resonant-core: RingBuf ---
    let mut ring: RingBuf<f32, 8> = RingBuf::new();
    ring.push(1.0);
    ring.push(0.5);

    // --- resonant-core: window functions ---
    let mut buf = [1.0_f32; 8];
    window::hann(&mut buf);

    // --- resonant-core: window::apply (SIMD dispatch, scalar on this target) ---
    let win = [0.5_f32; 8];
    window::apply(&mut buf, &win);

    // --- resonant-core: Signal type-state ---
    let sig = Signal::from_samples([0.0_f32, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]);

    // --- resonant-fft: radix-2 FFT (no_alloc path) ---
    use resonant_fft::Complex;
    let mut fft_buf = [
        Complex::new(sig.data()[0], 0.0),
        Complex::new(sig.data()[1], 0.0),
        Complex::new(sig.data()[2], 0.0),
        Complex::new(sig.data()[3], 0.0),
        Complex::new(sig.data()[4], 0.0),
        Complex::new(sig.data()[5], 0.0),
        Complex::new(sig.data()[6], 0.0),
        Complex::new(sig.data()[7], 0.0),
    ];
    let _ = radix2::fft(&mut fft_buf);

    // --- resonant-filters: Biquad (no_alloc) ---
    let coeffs = BiquadCoeffs { b0: 0.5, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 };
    let mut filter = Biquad::new(coeffs);
    let _ = filter.process_sample(1.0);

    // Signal success to QEMU. debug::exit() should not return under QEMU;
    // the loop below satisfies the `-> !` return type if it somehow does.
    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}
