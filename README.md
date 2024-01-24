# Install dependencies

```
python -m pip install -r requirements.txt
```

# File structure

* Synthesizing VBR matrices - `src/vbr_matrices_gen.py`'s `vbr_matrix_gen()` function. See Generating VBR matrices for commands to generate the matrices and save them in VBR format.
* Generating Matrix market files - `src/mtx_matrices_gen.py`'s `convert_all_vbr_to_mtx()` function (Depends on the previous step). See `Converting VBR Matrices to MTX Format` section in README for more details.
* Generating C code that performs SpMV over synthesized VBR matrices - `src/spmv_codegen.py`'s `spmv_codegen()` function (Depends on the first step of creating vbr matrices)
* `bench.py` - Benchmarks (other than partially-strided-codelets) (Run after `vbrgen.py`)
* `test.py` - Test correctness of `spmv_codegen` (Run after `vbrgen.py`)


# Generating VBR matrices

The script that generates VBR matrices is generate_vbr_matrices.sh. To generate all the default VBR matrices run the below command

```bash
nohup ./scripts/generate_vbr_matrices.sh &> timed_log5000.txt &
```

The individual VBR matrices can be generated using the below command
```bash
# To help and view argument descriptions to the VBR matrix generation script
python gen.py --help

# To generate a VBR matrix with given arguments
python gen.py --num-rows 1000 --num-cols 1000 --partition-type uniform --row-split 50 --col-split 50 --percentage-of-blocks 20 --percentage-of-zeros 50
```

# Converting VBR Matrices to MTX Format

Execute `scripts/convert_all_vbr_to_mtx.sh` to convert all the `.vbr` format matrices in the `Generated_Data` directory to the `.mtx` format matrices and store them in the `Generate_Matrix` directory.

```bash
nohup ./scripts/convert_all_vbr_to_mtx.sh &> timed_convert.txt &
python gen.py -o vbr_to_mtx
```

# Generating Code using VBR Matrices

Execute scripts/generate_spmv_code.sh script to generate code for all the `.vbr` format matrices in the `Generated_Data` directory and save them in the `Generated_SpMV` directory.

```bash
nohup ./scripts/generate_spmv_code.sh &> timed_gen_code.txt &
python gen.py -o vbr_to_code
```