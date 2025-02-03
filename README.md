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
./synthesize_matrices/generate_vbr_matrices_w_all_mostly_dense.sh
./synthesize_matrices/generate_vbr_matrices_w_some_mostly_sparse.sh
python3 synthesize_matrices/gen.py -o vbr_to_mtx
python3 synthesize_matrices/gen.py -o vbr_to_mtx --dense-blocks-only
```

The corresponding files will be generated in `Generated_VBR/`, `Generated_VBR_Sparse/`, `Generated_MMarket/` and `Generated_MMarket_Sparse/`.

#### Generate VBR matrices from manually partitioned Suitesparse matrices

```
python3 utils/convert_real_to_vbr.py
```

The corresponding files will be generated in `manual_vbr/`.

### SABLE

#### Run SABLE to generate CPU code to perform SpMV and SpMM over these generated Matrices

```
python3 synthesize_matrices/gen.py -o vbr_to_spmv
python3 synthesize_matrices/gen.py -o vbr_to_spmv --dense-blocks-only
python3 synthesize_matrices/gen.py -o vbr_to_spmm
python3 synthesize_matrices/gen.py -o vbr_to_spmm --dense-blocks-only
```

The corresponding files will be generated in `Generated_SpMV/`, `Generated_SpMV_Sparse/`, `Generated_SpMM/` and `Generated_SpMM_Sparse/`.

#### Run SABLE to generate CUDA code to perform SpMV and SpMM over these Matrices

```
python3 synthesize_matrices/gen.py -o vbr_to_spmv --cuda
python3 synthesize_matrices/gen.py -o vbr_to_spmv --cuda --dense-blocks-only
python3 synthesize_matrices/gen.py -o vbr_to_spmm --cuda
python3 synthesize_matrices/gen.py -o vbr_to_spmv --cuda --dense-blocks-only
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


## Benchmarking the CSR SpMV version
``` bash
# to compile for multi-thread execution with OpenMP
mkdir -p tmp && g++ -DOPENMP -O3 src/csr-spmv.cpp -o tmp/csr-spmv -fopenmp

# to compile for single-thread execution
mkdir -p tmp && g++ -O3 src/csr-spmv.cpp -o tmp/csr-spmv -fopenmp

# execute to evaluate individual matrix
./tmp/csr-spmv <matrix> <threads>
./tmp/csr-spmv /local/scratch/a/Suitesparse/heart1/heart1.mtx 1
./tmp/csr-spmv /local/scratch/a/Suitesparse/heart1/heart1.mtx 2

# edit the files in eval and skip lists in the file before running
# benchmark_csr_spmv.py, if empty it will evaluate all matrices
# to eval specific files add it to eval [] array without .mtx
# to skip specific files add it to skip [] array without .mtx 
# before executing the python script
# this will save the execution times in benchmark_results/csr_spmv.csv file
python benchmark_scripts/benchmark_csr_spmv.py <threads>
python benchmark_scripts/benchmark_csr_spmv.py 1
python benchmark_scripts/benchmark_csr_spmv.py 2
```