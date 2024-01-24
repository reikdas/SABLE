# Install dependencies

```
python -m pip install -r requirements.txt
```

# File structure

* Synthesizing VBR matrices - `src/gen_vbr_matrices.py`'s `gen_vbr_matrix()` function. See Generating VBR matrices for commands to generate the matrices and save them in VBR format.
* Generating Matrix market files - `vbrgen.py`'s `mtx_gen()` function (Depends on `gen_data()` having been executed)
* Generating C code that performs SpMV over synthesized VBR matrices - `vbrgen.py`'s `spmv_codegen()` function (Depends on `gen_data()` having been executed)
* `bench.py` - Benchmarks (other than partially-strided-codelets) (Run after `vbrgen.py`)
* `test.py` - Test correctness of `spmv_codegen` (Run after `vbrgen.py`)


# Generating VBR matrices

The script that generates VBR matrices is generate_vbr_matrices.sh. To generate all the default VBR matrices run the below command

```bash
nohup ./scripts/generate_vbr_matrices.sh &> timed_log.txt &
```

The individual VBR matrices can be generated using the below command
```bash
# To help and view argument descriptions to the VBR matrix generation script
python gen.py --help

# To generate a VBR matrix with given arguments
python gen.py --num-rows 1000 --num-cols 1000 --partition-type uniform --row-split 50 --col-split 50 --percentage-of-blocks 20 --percentage-of-zeros 50
```

# Running tests
```bash
# run an entire file
pytest test/test_sample.py

# run all tests in a file with verbose
pytest test/test_sample.py -s

# run specific test with verbose
pytest test/test_sample.py::test_val_array_append -s
```
