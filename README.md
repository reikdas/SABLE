System dependencies:
- Python3
- GNU parallel
- MPI

Install Intel MKL. Make sure `setvars.sh` has been executed in current working shell.

Install Python dependencies:
```
python3 -m pip install -r requirements.txt
```

In `src/consts.py` set the correct paths.

To benchmark, run the `bench_suitesparse_split_timings_c*.py` files.
