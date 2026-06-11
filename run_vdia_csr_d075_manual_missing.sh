#!/bin/bash
# VDIA+CSR on the 4 manual d075 band matrices (Freescale1 excluded: SpMM OOMs,
# so it only has SpMV results and would leave the sets inconsistent).
# Fills in the CSR dispatches still missing besides naive (done) and spreg (excluded):
#   SpMV: VDIA=bandmkl,   CSR=mkl,spv8,uzp
#   SpMM: VDIA=bandnaive, CSR=mkl
# Baselines are measured in-run (these matrices have no entries in the old
# sable_<op>_mkl_<csr>.json baseline files).
# Suggested: nohup ./run_vdia_csr_d075_manual_missing.sh > vdia_csr_d075_manual_missing.log 2>&1 &
cd /home/min/a/das160/SABLE-main

MATRICES="circuit5M_dc memchip ohne2 CoupCons3D"

python3 bench_suitesparse.py \
    --operation spmv --csr-kernels mkl,spv8,uzp --vbr-kernels none --vdia-kernels bandmkl \
    --bands-results-dir find-submatrices/results_bands_075 \
    --output-dir results \
    $MATRICES

python3 bench_suitesparse.py \
    --operation spmm --csr-kernels mkl --vbr-kernels none --vdia-kernels bandnaive \
    --bands-results-dir find-submatrices/results_bands_075 \
    --output-dir results \
    $MATRICES
