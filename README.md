## Install dependencies

```
python -m pip install -r requirements.txt
```

## File structure

* `src/vbr_matrices_gen.py` — generating synthetic VBR[^1] matrices.
  - Entry point: `vbr_matrix_gen`
  - Output: `./Generated_Data`
  - See “Generating VBR matrices” below for details.

All scripts below depend on the results from the script above.

* `src/mtx_matrices_gen.py` — generating Matrix market (`*.mtx`) files[^2].
  - Entry point: `convert_all_vbr_to_mtx()`.
  - Output: `./Generated_Matrix`
  - See “Converting VBR Matrices to MTX Format” below for details.

* `src/spmv_codegen.py` — generating C code that performs SpMV over synthesized VBR matrices.
  - Entry point: `spmv_codegen`.
  - Output: `./Generated_Code`

* `gen.py` — CLI for generators above.

* `bench.py` — benchmarks other than partially-strided codelets.

* `test.py` — test correctness of `spmv_codegen`.

[^1]: https://arxiv.org/abs/2005.12414
[^2]: https://math.nist.gov/MatrixMarket/formats.html

## Generating VBR matrices

The script that generates VBR matrices is `generate_vbr_matrices_w_all_mostly_dense.sh` and `generate_vbr_matrices_w_some_mostly_sparse.sh`. To generate all the default VBR matrices run:

```bash
nohup ./scripts/generate_vbr_matrices_w_all_mostly_dense.sh &> timed_log5000.txt &
nohup ./scripts/generate_vbr_matrices_w_some_mostly_sparse.sh &> timed_log5000.txt &
```

The individual VBR matrices can be generated using the command below:
```bash
# To help and view argument descriptions to the VBR matrix generation script
python gen.py --help

# To generate a VBR matrix with given arguments
python gen.py --dense-blocks-only --num-rows 1000 --num-cols 1000 --partition-type uniform --row-split 50 --col-split 50 --percentage-of-dense 20 --percentage-of-zeros 0 --percentage-of-zeros 50
```

## Converting VBR Matrices to MTX Format

Execute `scripts/convert_all_vbr_to_mtx.sh` to convert all the `.vbr` format matrices in the `Generated_Data` directory to the `.mtx` format matrices and store them in the `Generate_Matrix` directory.

```bash
nohup ./scripts/convert_all_vbr_to_mtx.sh &> timed_convert.txt &
python gen.py -o vbr_to_mtx
```

## Generating Code using VBR Matrices

Execute `scripts/generate_spmv_code.sh` to generate code for all the `.vbr` format matrices in the `Generated_Data` directory and save them in the `Generated_SpMV` directory.

```bash
nohup ./scripts/generate_spmv_code.sh &> timed_gen_code.txt &
python gen.py -o vbr_to_code
```

## Benchmarking the implementation

Execute the `bench.py`
```
nohup python bench.py &> bench.txt &
```
