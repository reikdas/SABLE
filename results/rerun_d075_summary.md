# SpMV VDIA+CSR delta=75% re-run (in-session baselines)

Session: 2026-06-11T18:22:45-04:00 on iapetus.ecn.purdue.edu, core 0, governor performance, freq 3400 MHz before / 3231 MHz after, turbo no_turbo=0, gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0, MKL /home/min/a/das160/intel/oneapi/mkl/2024.0, repo e720b231013f.

Protocol: 1 thread (OMP_NUM_THREADS=1, MKL_NUM_THREADS=1), taskset-pinned, 30 invocations x 30 internal iterations, strict interleaved baseline/composed alternation, decile outlier filtering, filtered means in microseconds. Baselines forced with --baseline-source run; band YAMLs verified against the published extraction before any run.

Exact commands (logs/commands.log):
```
[2026-06-11T02:49:41-04:00] run_all.sh start
[2026-06-11T02:49:41-04:00] /local/scratch/a/das160/rerun_d075/01_preflight.sh
[2026-06-11T02:49:42-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver TSC_OPF_1047)
[2026-06-11T14:32:32-04:00] run_all.sh start
[2026-06-11T14:32:32-04:00] /local/scratch/a/das160/rerun_d075/01_preflight.sh
[2026-06-11T14:32:33-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver TSC_OPF_1047)
[2026-06-11T14:33:58-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver bcsstk28)
[2026-06-11T14:34:12-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver bcsstk32)
[2026-06-11T14:35:12-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver cegb2802)
[2026-06-11T14:35:23-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver cegb2919)
[2026-06-11T14:35:34-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver gupta3)
[2026-06-11T14:39:08-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver heart1)
[2026-06-11T14:40:08-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver heart2)
[2026-06-11T14:40:40-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver heart3)
[2026-06-11T14:41:13-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver msc10848)
[2026-06-11T14:42:00-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver nd3k)
[2026-06-11T14:44:18-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver nemeth19)
[2026-06-11T16:27:19-04:00] run_all.sh start
[2026-06-11T16:27:19-04:00] /local/scratch/a/das160/rerun_d075/01_preflight.sh
[2026-06-11T16:27:27-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver nemeth20)
[2026-06-11T16:28:10-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver nemeth21)
[2026-06-11T16:29:08-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver nemeth22)
[2026-06-11T16:30:23-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver nemeth23)
[2026-06-11T16:31:26-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver nemeth24)
[2026-06-11T16:32:31-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver nemeth25)
[2026-06-11T16:33:35-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver nemeth26)
[2026-06-11T16:34:39-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver olafu)
[2026-06-11T16:35:22-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver opt1)
[2026-06-11T16:36:07-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver pkustk07)
[2026-06-11T16:37:02-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver pkustk08)
[2026-06-11T16:38:14-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive,mkl,spv8 --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver vsp_c-30_data_data)
[2026-06-11T16:38:23-04:00] (cd /local/scratch/a/das160/SABLE && SABLE_CODEGEN_DIR=/local/scratch/a/das160/rerun_d075/codegen SABLE_DENSE_TENSOR_DIR=/local/scratch/a/das160/rerun_d075/dense_tensors python3 bench_suitesparse.py --operation spmv --csr-kernels naive --vdia-kernels bandmkl --vbr-kernels none --baseline-source run --bench 30 --threads 1 --bands-results-dir /local/scratch/a/das160/rerun_d075/results_bands_075 --output-dir /local/scratch/a/das160/rerun_d075/results_driver ohne2)
[2026-06-11T18:22:44-04:00] run_all.sh start
[2026-06-11T18:22:44-04:00] /local/scratch/a/das160/rerun_d075/01_preflight.sh
[2026-06-11T18:52:53-04:00] /local/scratch/a/das160/rerun_d075/04_interleaved.sh
[2026-06-11T19:13:35-04:00] /local/scratch/a/das160/rerun_d075/01_preflight.sh record after
```

## Headline: best-vs-best over the 24 matrices
- geomean speedup: **0.9564x** over 24 matrices (1 wins, 23 losses, 0 ties)

## Per-matrix results (interleaved, filtered means, us)
| matrix | naive base | naive vdia | naive spd | mkl base | mkl vdia | mkl spd | spv8 base | spv8 vdia | spv8 spd | best-vs-best |
|---|---|---|---|---|---|---|---|---|---|---|
| TSC_OPF_1047 | 713.37 | 709.57 | 1.0054x | 710.54 | 740.98 | 0.9589x | 697.94 | 708.43 | 0.9852x | 0.9852x |
| bcsstk28 | 70.21 | 80.08 | 0.8767x | 71.75 | 83.16 | 0.8628x | 86.6 | 96.59 | 0.8966x | 0.8767x |
| bcsstk32 | 762.56 | 871.54 | 0.875x | 745.32 | 854.04 | 0.8727x | 784.48 | 893.19 | 0.8783x | 0.8727x |
| cegb2802 | 96.4 | 99.6 | 0.9679x | 98.31 | 103.0 | 0.9545x | 105.06 | 102.06 | 1.0294x | 0.9679x |
| cegb2919 | 114.83 | 117.0 | 0.9815x | 116.75 | 118.11 | 0.9885x | 119.75 | 117.95 | 1.0153x | 0.9815x |
| gupta3 | 7167.96 | 7261.99 | 0.9871x | 7351.64 | 7377.49 | 0.9965x | 7802.14 | 7819.97 | 0.9977x | 0.9871x |
| heart1 | 476.51 | 508.69 | 0.9367x | 500.01 | 546.72 | 0.9146x | 470.58 | 502.9 | 0.9357x | 0.9357x |
| heart2 | 235.37 | 249.86 | 0.942x | 248.84 | 278.85 | 0.8924x | 232.35 | 251.33 | 0.9245x | 0.9299x |
| heart3 | 235.24 | 249.48 | 0.9429x | 250.96 | 281.38 | 0.8919x | 232.32 | 250.43 | 0.9277x | 0.9312x |
| msc10848 | 427.15 | 433.76 | 0.9848x | 446.89 | 443.62 | 1.0074x | 431.19 | 433.57 | 0.9945x | 0.9852x |
| nd3k | 1304.2 | 1355.09 | 0.9624x | 1338.21 | 1379.71 | 0.9699x | 1318.9 | 1375.86 | 0.9586x | 0.9624x |
| nemeth19 | 283.33 | 338.98 | 0.8358x | 292.56 | 386.87 | 0.7562x | 288.35 | 342.06 | 0.843x | 0.8358x |
| nemeth20 | 336.2 | 339.12 | 0.9914x | 346.75 | 352.48 | 0.9837x | 340.8 | 342.86 | 0.994x | 0.9914x |
| nemeth21 | 407.81 | 436.65 | 0.934x | 417.87 | 463.45 | 0.9017x | 407.55 | 433.1 | 0.941x | 0.941x |
| nemeth22 | 470.16 | 532.4 | 0.8831x | 478.67 | 584.14 | 0.8194x | 470.56 | 540.42 | 0.8707x | 0.8831x |
| nemeth23 | 521.21 | 525.37 | 0.9921x | 527.35 | 528.11 | 0.9986x | 521.72 | 525.31 | 0.9932x | 0.9922x |
| nemeth24 | 518.26 | 522.77 | 0.9914x | 531.14 | 536.26 | 0.9905x | 522.88 | 523.43 | 0.9989x | 0.9914x |
| nemeth25 | 518.36 | 523.2 | 0.9907x | 531.7 | 537.25 | 0.9897x | 522.38 | 527.49 | 0.9903x | 0.9907x |
| nemeth26 | 518.95 | 528.08 | 0.9827x | 529.97 | 535.64 | 0.9894x | 522.22 | 528.01 | 0.989x | 0.9828x |
| olafu | 372.85 | 391.82 | 0.9516x | 365.45 | 388.27 | 0.9412x | 368.04 | 386.25 | 0.9529x | 0.9461x |
| opt1 | 670.6 | 674.27 | 0.9946x | 690.47 | 692.95 | 0.9964x | 684.75 | 687.54 | 0.9959x | 0.9946x |
| pkustk07 | 866.16 | 873.51 | 0.9916x | 875.85 | 880.01 | 0.9953x | 871.38 | 879.62 | 0.9906x | 0.9916x |
| pkustk08 | 1283.06 | 1299.91 | 0.987x | 1313.27 | 1313.45 | 0.9999x | 1326.65 | 1338.01 | 0.9915x | 0.987x |
| vsp_c-30_data_data | 62.67 | 74.76 | 0.8383x | 58.05 | 60.69 | 0.9565x | 47.25 | 45.43 | 1.0401x | 1.0401x |

## Driver same-session cross-check (codegen-phase, composed-then-baseline)
All driver baselines must say `measured_in_this_run`; any exception is listed under PROBLEMS above.

| matrix | kernel | driver baseline us | driver composed us |
|---|---|---|---|
| TSC_OPF_1047 | mkl | 693.91 | 706.1 |
| TSC_OPF_1047 | naive | 710.59 | 706.73 |
| TSC_OPF_1047 | spv8 | 694.14 | 710.56 |
| bcsstk28 | mkl | 69.42 | 78.82 |
| bcsstk28 | naive | 68.25 | 75.09 |
| bcsstk28 | spv8 | 81.73 | 90.03 |
| bcsstk32 | mkl | 736.0 | 828.12 |
| bcsstk32 | naive | 767.17 | 882.75 |
| bcsstk32 | spv8 | 780.69 | 893.93 |
| cegb2802 | mkl | 96.72 | 99.4 |
| cegb2802 | naive | 95.48 | 96.47 |
| cegb2802 | spv8 | 101.05 | 97.61 |
| cegb2919 | mkl | 114.1 | 115.12 |
| cegb2919 | naive | 110.89 | 113.58 |
| cegb2919 | spv8 | 117.63 | 116.13 |
| gupta3 | mkl | 7805.46 | 7543.97 |
| gupta3 | naive | 7522.04 | 7380.58 |
| gupta3 | spv8 | 7913.05 | 7853.56 |
| heart1 | mkl | 486.73 | 507.43 |
| heart1 | naive | 476.51 | 503.0 |
| heart1 | spv8 | 469.76 | 502.76 |
| heart2 | mkl | 238.55 | 252.51 |
| heart2 | naive | 235.23 | 250.62 |
| heart2 | spv8 | 232.06 | 252.09 |
| heart3 | mkl | 238.11 | 252.42 |
| heart3 | naive | 235.32 | 250.66 |
| heart3 | spv8 | 232.43 | 250.23 |
| msc10848 | mkl | 428.28 | 429.29 |
| msc10848 | naive | 427.36 | 432.99 |
| msc10848 | spv8 | 429.8 | 431.79 |
| nd3k | mkl | 1321.1 | 1350.81 |
| nd3k | naive | 1303.56 | 1351.81 |
| nd3k | spv8 | 1308.68 | 1332.68 |
| nemeth19 | mkl | 282.8 | 347.65 |
| nemeth19 | naive | 283.03 | 340.56 |
| nemeth19 | spv8 | 288.03 | 340.92 |
| nemeth20 | mkl | 334.83 | 335.66 |
| nemeth20 | naive | 336.08 | 340.29 |
| nemeth20 | spv8 | 340.12 | 342.99 |
| nemeth21 | mkl | 402.06 | 434.68 |
| nemeth21 | naive | 408.07 | 435.69 |
| nemeth21 | spv8 | 410.31 | 435.7 |
| nemeth22 | mkl | 464.69 | 538.88 |
| nemeth22 | naive | 472.95 | 541.84 |
| nemeth22 | spv8 | 470.1 | 536.47 |
| nemeth23 | mkl | 514.21 | 515.79 |
| nemeth23 | naive | 522.07 | 526.08 |
| nemeth23 | spv8 | 519.68 | 521.68 |
| nemeth24 | mkl | 514.48 | 514.91 |
| nemeth24 | naive | 518.11 | 522.91 |
| nemeth24 | spv8 | 519.24 | 523.24 |
| nemeth25 | mkl | 514.6 | 516.98 |
| nemeth25 | naive | 517.94 | 523.05 |
| nemeth25 | spv8 | 520.94 | 526.16 |
| nemeth26 | mkl | 515.05 | 518.12 |
| nemeth26 | naive | 518.02 | 523.74 |
| nemeth26 | spv8 | 521.86 | 526.08 |
| ohne2 | naive | 9833.84 | 5609.14 |
| olafu | mkl | 353.1 | 368.02 |
| olafu | naive | 373.02 | 390.8 |
| olafu | spv8 | 367.72 | 387.79 |
| opt1 | mkl | 672.33 | 674.39 |
| opt1 | naive | 673.73 | 680.33 |
| opt1 | spv8 | 677.52 | 684.94 |
| pkustk07 | mkl | 859.23 | 872.88 |
| pkustk07 | naive | 868.72 | 872.53 |
| pkustk07 | spv8 | 868.8 | 877.87 |
| pkustk08 | mkl | 1270.03 | 1369.56 |
| pkustk08 | naive | 1290.14 | 1308.24 |
| pkustk08 | spv8 | 1296.2 | 1282.45 |
| vsp_c-30_data_data | mkl | 53.3 | 54.7 |
| vsp_c-30_data_data | naive | 59.25 | 67.91 |
| vsp_c-30_data_data | spv8 | 44.43 | 40.77 |

## ohne2 (P0b): confirmation + cross-machine calibration point
ohne2: 181343x181343, nnz 11,063,545, band nnz 136,383 = 1.23272% in 424 segments, residual kernel: naive.

- **baseline (fully-sparse CSR naive): 9273.86 us** (filtered mean, 900 iterations)
- **composed (VDIA+CSR naive): 5580.64 us** (filtered mean, 900 iterations)
- **speedup: 1.6618x** (previously recorded 1.73x with in-session baseline)
- per-invocation means (us), baseline: [9560.37, 9322.71, 9314.23, 9278.7, 9259.13, 9310.22, 9272.74, 9286.76, 9336.18, 9297.3, 9370.36, 9258.16, 9250.15, 9269.97, 9255.71, 9299.02, 9295.5, 9267.15, 9310.58, 9294.9, 9311.81, 9275.25, 9298.83, 9270.13, 9257.93, 9314.99, 9256.07, 9294.89, 9281.19, 9947.97]
- per-invocation means (us), composed: [5940.52, 5950.03, 5870.92, 5908.85, 5861.21, 5880.53, 5888.7, 5876.99, 5895.41, 5871.46, 5874.05, 5895.59, 5943.07, 5895.31, 5889.91, 5864.62, 5913.12, 6102.22, 5885.88, 5871.77, 5892.56, 5906.83, 5875.05, 5873.74, 5863.51, 5877.61, 5927.96, 5912.78, 5876.68, 5993.02]
- driver cross-check: baseline 9833.84 us, composed 5609.14 us (measured_in_this_run)

## P1: provenance audit verdicts
- `spmv_vdia_vbr_csr_d075.json`: IN-SESSION (distinct from old sweep: only 0/75 baselines coincide with sable_spmv_mkl_mkl.json, sable_spmv_mkl_naive.json, sable_spmv_mkl_spv8.json)
- `spmm_vdia_vbr_csr_d075.json`: IN-SESSION (distinct from old sweep: only 0/54 baselines coincide with sable_spmm_mkl_mkl.json, sable_spmm_mkl_naive.json, sable_spmm_mkl_spreg.json); 16 baseline_us==0 entries (baseline unavailable at aggregation time)
- `spmv_vdia_csr_d075.json`: LOOKED UP from sable_spmv_mkl_mkl.json, sable_spmv_mkl_naive.json, sable_spmv_mkl_spv8.json (75/75 baselines bit-identical to the old sweep)
- `sable_spmv_mkl_naive.json`: IN-SESSION (old-format sweep: 117/117 populated entries record compile_time_sparse_s > 0, i.e. the fully-sparse baseline binary was compiled and measured in the same sweep). NOTE: this file is itself the lookup source used by later aggregations.
- `sable_spmv_mkl_mkl.json`: IN-SESSION (old-format sweep: 117/117 populated entries record compile_time_sparse_s > 0, i.e. the fully-sparse baseline binary was compiled and measured in the same sweep). NOTE: this file is itself the lookup source used by later aggregations.
- `sable_spmv_mkl_spv8.json`: IN-SESSION (old-format sweep: 117/117 populated entries record compile_time_sparse_s > 0, i.e. the fully-sparse baseline binary was compiled and measured in the same sweep). NOTE: this file is itself the lookup source used by later aggregations.
- `sable_spmv_mkl_uzp.json`: IN-SESSION (old-format sweep: 117/117 populated entries record compile_time_sparse_s > 0, i.e. the fully-sparse baseline binary was compiled and measured in the same sweep). NOTE: this file is itself the lookup source used by later aggregations.
- `sable_spmm_mkl_naive.json`: IN-SESSION (old-format sweep: 126/126 populated entries record compile_time_sparse_s > 0, i.e. the fully-sparse baseline binary was compiled and measured in the same sweep). NOTE: this file is itself the lookup source used by later aggregations.
- `sable_spmm_mkl_mkl.json`: IN-SESSION (old-format sweep: 126/126 populated entries record compile_time_sparse_s > 0, i.e. the fully-sparse baseline binary was compiled and measured in the same sweep). NOTE: this file is itself the lookup source used by later aggregations.
- `sable_spmm_mkl_spreg.json`: IN-SESSION (old-format sweep: 30/30 populated entries record compile_time_sparse_s > 0, i.e. the fully-sparse baseline binary was compiled and measured in the same sweep; 87 empty placeholder entries ignored). NOTE: this file is itself the lookup source used by later aggregations.

