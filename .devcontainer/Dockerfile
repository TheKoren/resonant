FROM rust:1.86-bookworm

# System packages: QEMU for ARM, arm-none-eabi toolchain for objdump/nm,
# and build essentials.
RUN apt-get update && apt-get install -y --no-install-recommends \
        qemu-system-arm \
        gcc-arm-none-eabi \
        binutils-arm-none-eabi \
        libfontconfig1-dev \
        pkg-config \
        git \
    && rm -rf /var/lib/apt/lists/*

# Rust targets and components
RUN rustup target add thumbv7em-none-eabihf wasm32-unknown-unknown \
    && rustup component add clippy rustfmt llvm-tools

# cargo-binutils gives rust-objdump / rust-nm (uses the bundled LLVM toolchain)
RUN cargo install cargo-binutils --locked \
    && cargo install cargo-llvm-cov@0.6.21 --locked \
    && cargo install cargo-deny@0.18.3 --locked

WORKDIR /workspace
