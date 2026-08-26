#!/usr/bin/env bash
#
# Build every native (C/C++) component of SABLE that is compiled with
# machine-specific flags (-march=native, or fixed AVX-512 flags).
#
# IMPORTANT: because of -march=native, binaries built by this script are
# tied to the CPU of the machine it runs on. The Dockerfile runs this once
# at `docker build` time, which is correct as long as you build and run the
# container on the same machine (the normal case for artifact evaluation).
# If you build on one machine and run on another, re-run this script inside
# the running container first:
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
    echo "  -> The SpV8 baseline (spv8-public) requires AVX-512 and will be built"
    echo "     but is NOT expected to run correctly here."
    echo "  -> sparse-register-tiling will be built with -DENABLE_AVX2=True instead"
    echo "     of -DENABLE_AVX512=True."
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
( cd "$SABLE_DIR/sparse-register-tiling/spmm_nano_kernels" && python3 -m codegen.generate_ukernels )
mkdir -p "$SABLE_DIR/sparse-register-tiling/build"
if [ "$HAVE_AVX512" -eq 1 ]; then
    AVX_FLAG="-DENABLE_AVX512=True"
else
    AVX_FLAG="-DENABLE_AVX2=True"
fi
# NOTE: this CMake project downloads two dependencies at *configure* time --
# rapidyaml (FetchContent git clone) and Google Benchmark v1.5.5
# (ExternalProject zip). They are fetched here rather than pre-staged in the
# image, so this step needs network access. Nothing else in the artifact does:
# the matrices and all sources already ship inside the image.
( cd "$SABLE_DIR/sparse-register-tiling/build" && \
  cmake .. -DCMAKE_BUILD_TYPE=Release "$AVX_FLAG" && \
  make -j"$(nproc)" SPMM_demo )
echo "  -> built with $AVX_FLAG"
echo

echo "=== [4/4] Rust toolchain check (for the UZP baseline) ==="
if command -v rustup >/dev/null 2>&1; then
    echo "  rustup found at $(command -v rustup); UZP's z_polyhedrator will build lazily"
    echo "  on first use via uzp_prepare.sh (no network access needed -- the 1.85.0"
    echo "  toolchain was pre-installed at image build time)."
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

# NOTE: this script deliberately does NOT delete any .git directory.
#
# An earlier version stripped them, to remove the stray checkout CMake's
# FetchContent leaves under sparse-register-tiling/build/_deps/ryml-src/.
# That is harmless, and deleting indiscriminately was not: the SABLE clone
# keeps its .git on purpose, and wiping it would break the `git pull` inside
# the container that picks up new commits on the artifact branch.
