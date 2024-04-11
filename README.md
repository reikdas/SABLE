## Setup Repository

```
git clone git@github.com:reikdas/SABLE.git --recurse-submodules
```

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
./scripts/generate_vbr_matrices.sh
python3 gen.py -o vbr_to_mtx
```

The corresponding files will be generated in `Generated_VBR/` and `Generated_MMarket/`.

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
```

The corresponding files will be generated in `results/`.

### Partially Strided Codelet

#### Build

```
cd partially-strided-codelet
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$INTEL_PARENT_DIR/intel/oneapi/mkl/latest/lib/intel64/;$INTEL_PARENT_DIR/intel/oneapi/mkl/latest/include/" ..
make
```

#### Benchmark Partially Strided Codelet for SpMV

```
cd $SABLE_PARENT_DIR/SABLE/partially-strided-codelet
python3 bench.py
```
The corresponding benchmark files will be generated in the root dir of `partially-strided-codelet/`.

### Sparse Register Tiling

#### Build

```
cd $SABLE_PARENT_DIR/SABLE
cd sparse-register-tiling
cd spmm_nano_kernels
python3 -m codegen.generate_ukernels
cd ..
mkdir release-build
cmake -Brelease-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$INTEL_PARENT_DIR/intel/oneapi/mkl/latest/lib/intel64/;$INTEL_PARENT_DIR/intel/oneapi/mkl/latest/include/" -DENABLE_AVX512=True .
make -Crelease-build SPMM_demo
```

#### Benchmark Sparse Register Tiling for SpMM

```
python3 bench.py
```
The corresponding benchmark files will be generated in `sparse-register-tiling/results/`.

### Generate plots used in paper

```
cd plots/
for f in *.py; do python3 "$f"; done
```

### Generate tables used in paper

```
cd tables/
for f in *.py; do python3 "$f"; done
```
