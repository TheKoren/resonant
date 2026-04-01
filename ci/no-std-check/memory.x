/* Memory layout for the ARM MPS2+ AN386 (Cortex-M4) as emulated by QEMU.
 *
 * QEMU invocation for local verification:
 *   cargo build --release --target thumbv7em-none-eabihf
 *   qemu-system-arm -machine mps2-an386 -nographic \
 *     -semihosting-config enable=on,target=native \
 *     -kernel target/thumbv7em-none-eabihf/release/resonant-no-std-check
 *   echo "Exit code: $?"
 *
 * The binary exits cleanly via semihosting (code 0 = success, 1 = panic).
 */
MEMORY
{
    FLASH : ORIGIN = 0x00000000, LENGTH = 256M
    RAM   : ORIGIN = 0x20000000, LENGTH = 4M
}
