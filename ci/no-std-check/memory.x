/* Memory layout for the ARM MPS2+ AN386 (Cortex-M4) as emulated by QEMU.
 *
 * QEMU invocation for local verification:
 *   cargo build --release --target thumbv7em-none-eabihf
 *   qemu-system-arm -machine mps2-an386 -nographic -semihosting \
 *     -kernel target/thumbv7em-none-eabihf/release/resonant-no-std-check
 *
 * Note: with panic-halt the program loops on panic; press Ctrl-A X to quit QEMU.
 * Swap panic-halt for panic-semihosting (with a semihosting exit call in main)
 * if you need a clean CI-friendly exit.
 */
MEMORY
{
    FLASH : ORIGIN = 0x00000000, LENGTH = 256M
    RAM   : ORIGIN = 0x20000000, LENGTH = 4M
}
