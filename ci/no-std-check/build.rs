// Tell the linker where to find memory.x so that cortex-m-rt's link.x
// can INCLUDE it regardless of which directory cargo is invoked from.
fn main() {
    println!(
        "cargo:rustc-link-search={}",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    );
    println!("cargo:rerun-if-changed=memory.x");
}
