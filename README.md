## Install dependencies

System dependencies:
- Python3
- GNU parallel
- Intel MKL for Partially Strided Codelet and Sparse Register Tiling

Install Python dependencies:
```
python3 -m pip install -r requirements.txt
```

## Perform the following operations in order

#### Generate VBR Matrices and their Matrix Market equivalent

```
./scripts/generate_vbr_matrices_w_all_mostly_dense.sh
./scripts/generate_vbr_matrices_w_some_mostly_sparse.sh
python3 gen.py -o vbr_to_mtx
```

The corresponding files will be generated in `Generated_VBR/`, `Generated_VBR_Sparse/`, `Generated_MMarket/` and `Generated_MMarket_Sparse/`.

### SABLE

#### Run SABLE to generate code to perform SpMV and SpMM over these Matrices

```
python3 gen.py -o vbr_to_spmv
python3 gen.py -o vbr_to_spmm
```

The corresponding files will be generated in `Generated_SpMV/` and `Generated_SpMM/`.

#### Benchmark SABLE

```
python3 bench.py
python3 bench_sparse.py
```

The corresponding files will be generated in `results/`.
