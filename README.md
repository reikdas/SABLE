# File structure

* Synthesizing VBR matrices - `vbrgen.py`'s `gen_data()` function
* Generating Matrix market files - `vbrgen.py`'s `mtx_gen()` function (Depends on `gen_data()` having been executed)
* Generating C code that performs SpMV over synthesized VBR matrices - `vbrgen.py`'s `spmv_codegen()` function
* `bench.py` - Benchmarks (other than partially-strided-codelets)
* `test.py` - Test correctness of `spmv_codegen`
