#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
build="$root/runs/smartcore"
mkdir -p "$build/src"
cp "$root/benchmarks/single-model/smartcore.rs" "$build/src/main.rs"
cat > "$build/Cargo.toml" <<'TOML'
[package]
name = "terasweep-smartcore-comparison"
version = "0.1.0"
edition = "2021"
publish = false

[dependencies]
smartcore = "=0.6.14"
serde_json = "=1.0.154"

[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
TOML
RUSTFLAGS='-C target-cpu=native' cargo +1.100.0 build --release --manifest-path "$build/Cargo.toml"
