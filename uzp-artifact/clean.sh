#!/bin/bash

pushd z_polyhedrator && RUSTUP_HOME="/opt/rustup" CARGO_HOME="/tmp/.cargo_pldi25_artifact" cargo clean && popd

rm -rf /opt/rustup
rm -rf /tmp/.cargo_pldi25_artifact

rm -rf /uzp-artifact/lib/scripts/__pycache__/
rm -rf /uzp-artifact/spmv-executors/__pycache__/

find /tmp -type f -name "*.uzp" -delete

make -C /uzp-artifact/spmv-executors/uzp-genex distclean clean
make -C /uzp-artifact/spmv-executors/spmv-mkl clean
make -C /uzp-artifact/spmv-executors/spmv-csr clean
make -C /uzp-artifact/spmv-executors/spmv-csr5/CSR5_avx2 clean
