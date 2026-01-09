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
