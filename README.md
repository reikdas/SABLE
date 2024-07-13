## Install dependencies

System dependencies:
- Python3
- GNU parallel

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
python3 gen.py -o vbr_to_mtx --dense-blocks-only
```

The corresponding files will be generated in `Generated_VBR/`, `Generated_VBR_Sparse/`, `Generated_MMarket/` and `Generated_MMarket_Sparse/`.

#### Generate VBR matrices from manually partitioned Suitesparse matrices

```
python3 scripts/convert_real_to_vbr.py
```

The corresponding files will be generated in `manual_vbr/`.

### SABLE

#### Run SABLE to generate CPU code to perform SpMV and SpMM over these generated Matrices

```
python3 gen.py -o vbr_to_spmv
python3 gen.py -o vbr_to_spmv --dense-blocks-only
python3 gen.py -o vbr_to_spmm
python3 gen.py -o vbr_to_spmm --dense-blocks-only
```

The corresponding files will be generated in `Generated_SpMV/`, `Generated_SpMV_Sparse/`, `Generated_SpMM/` and `Generated_SpMM_Sparse/`.

#### Run SABLE to generate CUDA code to perform SpMV and SpMM over these Matrices

```
python3 gen.py -o vbr_to_spmv --cuda
python3 gen.py -o vbr_to_spmv --cuda --dense-blocks-only
python3 gen.py -o vbr_to_spmm --cuda
python3 gen.py -o vbr_to_spmv --cuda --dense-blocks-only
```

The corresponding files will be generated in `Generated_SpMV_cuda/`, `Generated_SpMV_cuda_Sparse`, `Generated_SpMM_cuda` and `Generated_SpMM_cuda_Sparse/`.

#### Benchmark SABLE

```
python3 bench.py
python3 bench_sparse.py
python3 bench_manualvbr.py
python3 bench_cuda.py
python3 bench_sparse_cuda.py
```

The corresponding files will be generated in `results/`.

## Contributing guidelines

### Formatting files

Please run the following commands in order over files you have changed.

```
black <filename.py>
isort <filename.py>
```
