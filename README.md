# Install dependencies

```
python -m pip install -r requirements.txt
```

# File structure

* Synthesizing VBR matrices - `vbrgen.py`'s `gen_data()` function
* Generating Matrix market files - `vbrgen.py`'s `mtx_gen()` function (Depends on `gen_data()` having been executed)
* Generating C code that performs SpMV over synthesized VBR matrices - `vbrgen.py`'s `spmv_codegen()` function (Depends on `gen_data()` having been executed)
* `bench.py` - Benchmarks (other than partially-strided-codelets) (Run after `vbrgen.py`)
* `test.py` - Test correctness of `spmv_codegen` (Run after `vbrgen.py`)
