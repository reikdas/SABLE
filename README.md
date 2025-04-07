## Install dependencies

System dependencies:
- Python3
- GNU parallel

Install Python dependencies:
```
python3 -m pip install -r requirements.txt
```

Setup Intel MKL

In `src/consts.py` set the correct paths.

To get the block choosing model:
- Run `studies/threshold.py`.
- Run `studies/find_threshold.py`.

In `bench_parallel_launcher.py`:
- set `mtx_dir` to the Suitesparse directory.
- the list `eval` is the list of matrices that you want to generate code for and compile.

In `eval.py`:
- the list `eval` is the list of matrices that you want to evaluate.
- Set the number of threads in the call to `eval_single_proc()`.
- Account for NUMA nodes and such while setting the cores to execute our programs on, in line 30.
- Evaluated values will be stored in `results/res_{}.csv`.

For MKL baseline, in `bench_mkl.py`:
- Set `mtx_dir` to the Suitesparse directory.
- the list `eval` is the list of matrices that you want to evaluate.
- Correctly set the value of `threads = []`.
- Evaluated values will be stored in `results/mkl-spmv-merged-results.csv`.