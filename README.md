## Setup Instructions

### 1. Clone Repository

Clone the repository and make sure to recursively download submodules:

```bash
git clone --recursive <repository-url>
```

### 2. System Dependencies

Install the following system dependencies:
- Python3
- MPI

### 3. Install Intel MKL

Install Intel MKL. Make sure `setvars.sh` has been executed in current working shell.

### 4. Build spv8-public/

Make sure to build `spv8-public/`:

```bash
cd spv8-public/
# Follow build instructions for spv8-public (just `make` in root dir should work)
cd ..
```

### 5. Python Environment Setup

Create a Python virtual environment and install dependencies:

```bash
python3 -m venv sable-env
source sable-env/bin/activate
pip install -r requirements.txt
```

### 6. Configuration

In `src/consts.py` set the correct paths.

### 7. Running Benchmarks

To benchmark, run the `bench_suitesparse_split_timings_c*.py` files.

## Testing

### Quick tests (no MKL / hardware dependencies)

These tests verify the benchmarking infrastructure itself — compiler flag
assembly and C code generation — without running any actual sparse
computations.  They run in seconds and need only Python + pytest:

```bash
# Compiler-flag unit tests (all Backend × DenseKernel combinations)
python3 -m pytest tests/test_compile_flags.py -v

# Codegen golden tests (compares generated C against stored references)
python3 -m pytest tests/test_codegen_golden.py -v

# Run both at once
python3 -m pytest tests/test_compile_flags.py tests/test_codegen_golden.py -v
```

After an **intentional** change to codegen output, regenerate the golden
files:

```bash
python3 -m pytest tests/test_codegen_golden.py --update-golden
```

### Full integration tests (require MKL, GCC, hardware)

The full test suite in `test.py` compiles and runs generated C code, so it
needs GCC, MKL, and the `spv8-public` / `sparse-register-tiling` submodules
built:

```bash
python3 -m pytest test.py -v
```
