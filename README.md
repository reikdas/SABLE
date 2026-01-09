System dependencies:
- Python3
- MPI

Install Intel MKL. Make sure `setvars.sh` has been executed in current working shell.

Install Python dependencies:

Create a Python virtual environment and install dependencies:

```bash
python3 -m venv sable-env
source sable-env/bin/activate
pip install -r requirements.txt
```

In `src/consts.py` set the correct paths.

To benchmark, run the `bench_suitesparse_split_timings_c*.py` files.
