#!/usr/bin/env bash
#
# Build every native (C/C++) component of SABLE that is compiled with
# machine-specific flags (-march=native, or fixed AVX-512 flags).
#
# IMPORTANT: because of -march=native, binaries built by this script are
# tied to the CPU of the machine it runs on. The artifact image therefore
# ships without them: run this once inside the container, on the machine the
# benchmarks will run on, before benchmarking:
#
#   docker run --rm -it <image> bash
#   bash build_native.sh
#
set -euo pipefail

SABLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== CPU feature check ==="
if grep -q avx512f /proc/cpuinfo 2>/dev/null; then
    HAVE_AVX512=1
    echo "AVX-512 (avx512f/avx512vl): supported on this CPU."
else
    HAVE_AVX512=0
    echo "AVX-512: NOT supported on this CPU."
    echo "  -> The SpV8 (SpMV) and SpReg (SpMM) baselines require AVX-512, which the"
    echo "     artifact lists as a hardware requirement. They are not built here."
fi
echo

echo "=== [1/4] Building find-submatrices/ (block partitioner: partition_matrix) ==="
cmake -S "$SABLE_DIR/find-submatrices" -B "$SABLE_DIR/find-submatrices/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$SABLE_DIR/find-submatrices/build" --target partition_matrix -j"$(nproc)"
echo "  -> $SABLE_DIR/find-submatrices/build/partition_matrix"
echo

echo "=== [2/4] Building spv8-public/ (SpV8 SpMV baseline) ==="
if [ "$HAVE_AVX512" -eq 1 ]; then
    make -C "$SABLE_DIR/spv8-public"
    echo "  -> built"
else
    echo "  -> skipping build: spv8-public hard-codes AVX-512 flags"
    echo "     (-mavx512f -mavx512vl -mfma -mprfchw) and this CPU lacks AVX-512."
    echo "     The SPV8CSRSpmv kernel will not be usable in this environment."
fi
echo

echo "=== [3/4] Building sparse-register-tiling/ (SpReg SpMM baseline) ==="
# SABLE's SpReg kernel links spmm_nano_kernels/build/libspreg.a. Without it
# every generated SpReg program compiles the library's ~115 C++ files itself,
# so building it once here is what keeps SpMM benchmarking to a sane length.
# It needs no network access.
if [ "$HAVE_AVX512" -eq 1 ]; then
    ( cd "$SABLE_DIR/sparse-register-tiling/spmm_nano_kernels" && \
      python3 -m codegen.generate_ukernels && \
      make -j"$(nproc)" )
    echo "  -> $SABLE_DIR/sparse-register-tiling/spmm_nano_kernels/build/libspreg.a"
else
    echo "  -> skipping build: SABLE drives SpReg through its AVX-512 micro-kernels"
    echo "     (-DENABLE_AVX512 -mavx512f) and this CPU lacks AVX-512."
    echo "     The SPRegCSRSpmm kernel will not be usable in this environment."
fi
echo

echo "=== [4/4] Rust toolchain check (for the UZP baseline) ==="
if command -v rustup >/dev/null 2>&1; then
    echo "  rustup found at $(command -v rustup); UZP's z_polyhedrator will build lazily"
    echo "  on first use via uzp_prepare.sh. The 1.85.0 toolchain is already installed;"
    echo "  that first build fetches the crates pinned in Cargo.lock, so it needs network access."
else
    echo "  WARNING: rustup not found on PATH. The UZP baseline (UZPCSRSpmv) will try to"
    echo "  self-install rustup via 'curl https://sh.rustup.rs | sh' the first time it is"
    echo "  used, which requires network access at benchmark time."
fi
echo

echo "All native components built (or explicitly skipped with a reason above)."
echo "spf_aggregator (uzp-artifact/uzp-tuners) and z_polyhedrator are intentionally"
echo "NOT built here -- uzp_prepare.sh builds and caches both automatically, per"
echo "matrix, the first time a UZP kernel runs against that matrix."
echo

# NOTE: this script deliberately does NOT delete any .git directory. The SABLE
# clone keeps its .git on purpose: `git pull` inside the container is how new
# commits on the artifact branch are picked up.
