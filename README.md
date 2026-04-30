## Usage

The SABLE frontend lets you split a sparse matrix into dense blocks and a
sparse residual, dispatch each part to a different kernel, and compile
everything into a single timed C program.

The example below runs **SpMM** (sparse matrix × dense matrix).  It assumes
`matrix.mtx` (a Matrix Market file) and `rhs.matrix` (a flat text file of
dense values) already exist on disk.

```python
from sable import Matrix, Plan
from sable.extractors import BlockDetector, CSRConvertor
from sable.kernels import MixedVBRSpmm, NaiveCSRSpmm
from sable.tensor import DenseInput, DenseLayout

# ── 1. Wrap the sparse matrix ────────────────────────────────────────
matrix = Matrix("matrix.mtx", name="my_spmm")
N_RHS = 512  # number of right-hand-side columns

# ── 2. Create a Plan ─────────────────────────────────────────────────
plan = Plan(matrix, artifact_dir="./artifacts")

# Tell the plan where the dense RHS lives and its shape / layout.
plan.rhs(
    DenseInput.matrix(
        "rhs.matrix",
        shape=(matrix.ncols, N_RHS),
        layout=DenseLayout.ROW_MAJOR,
    )
)

# ── 3. Extract formats ───────────────────────────────────────────────
# BlockDetector (defined in sable/extractors/block_detector.py) runs
# the native C++ partitioner in find-submatrices/ to discover dense
# sub-blocks.  It returns a VBR format and shrinks the residual.
vbr = plan.extract(
    BlockDetector(min_density=0.5, min_area=2500, threads=4)
)

# CSRConvertor claims whatever non-zeros remain as plain CSR.
csr = plan.extract(CSRConvertor())

# ── 4. Dispatch kernels ──────────────────────────────────────────────
# MixedVBRSpmm uses a heuristic per dense block: blocks whose smallest
# dimension ≥ 8 and aspect ratio ≤ 100 are multiplied with MKL's
# cblas_dgemm; smaller or very thin blocks fall back to a naive loop.
plan.dispatch(vbr, MixedVBRSpmm())

# NaiveCSRSpmm multiplies the sparse residual with a triple-nested loop.
plan.dispatch(csr, NaiveCSRSpmm())

# ── 5. Compile, build, and run ───────────────────────────────────────
executor = plan.compile(filename="my_spmm", bench=5)
executor.build()
output = executor.run()
print(output)
```

The final three calls are separate so generated code can be inspected or
compiled independently:

- `plan.compile(...)` validates that every extracted format has a dispatched
  kernel, binds each `Rep` to a unique C symbol, writes the `.sabledata` file,
  and emits the C source.
- `executor.build()` invokes the compiler with flags requested by the
  dispatched kernels, for example MKL include and link flags.
- `executor.run()` executes the compiled binary, reads the dense RHS from disk,
  runs the generated sparse computation for `bench` iterations, and returns the
  program output.

### Writing an extractor

An extractor takes a `ResidualMatrix` and carves out a `Format`.  It must
expose an `extract` method that returns `(Format, ResidualMatrix)`.
Set `produces` so the `Plan` can type-check the result.

SABLE uses `Rep` for staged data: a `Rep` value names data that is known
while the Python frontend is generating code, but will be materialized as a C
array in the emitted program.

```python
from sable.formats import CSR, Rep
from sable.matrix import ResidualMatrix


class CSRConvertor:
    produces = CSR

    def extract(self, A: ResidualMatrix) -> tuple[CSR, ResidualMatrix]:
        csr = A.to_csr()
        fmt = CSR(
            nrows=A.nrows,
            ncols=A.ncols,
            nnz=int(csr.nnz),
            indptr=Rep(csr.indptr.astype(int).tolist(), label="csr_indptr"),
            indices=Rep(csr.indices.astype(int).tolist(), label="csr_indices"),
            values=Rep(csr.data.astype(float).tolist(), label="csr_val"),
        )
        return fmt, A.empty_residual()
```

`BlockDetector` in `sable/extractors/block_detector.py` is a more involved
example — it calls the C++ partitioner in `find-submatrices/` and returns
a `VBR` format.

### Writing a kernel

A kernel tells the compiler what C code to emit for a given format.  Set
`accepts` to the format type and `operation` to an operation type (`Operation.SPMV` or
`Operation.SPMM` -- the user can write their own operation).  The frontend derives the expected RHS rank from that
operation and checks that every dispatched kernel in a plan agrees.  The only
required method is `emit_call`, which returns a C code string.

```python
from sable import Operation
from sable.formats import CSR


class NaiveCSRSpmm:
    accepts = CSR
    operation = Operation.SPMM

    def emit_includes(self) -> list[str]:
        return []

    def emit_call(self, fmt: CSR, y: str, x: str, rhs) -> str:
        if fmt.nnz == 0:
            return ""
        nrhs = rhs.shape[1]
        return f"""\
for (int i = 0; i < {fmt.nrows}; i++) {{
    for (int p = {fmt.indptr}[i]; p < {fmt.indptr}[i + 1]; p++) {{
        int col = {fmt.indices}[p];
        double a = {fmt.values}[p];
        for (int j = 0; j < {nrhs}; j++) {{
            {y}[i * {nrhs} + j] += a * {x}[col * {nrhs} + j];
        }}
    }}
}}
"""

    def compile_flags(self) -> list[str]:
        return []

    def link_flags(self) -> list[str]:
        return []

    def runtime_env(self) -> dict[str, str]:
        return {}
```

Kernels that need external libraries (MKL, spv8, sparse-register-tiling)
return the appropriate flags from `compile_flags()` / `link_flags()` and
headers from `emit_includes()`.  See `sable/kernels/vbr.py` and
`sable/kernels/csr.py` for examples.

## Setup Instructions

### 1. Clone Repository

Clone the repository and make sure to recursively download submodules:

```bash
git clone --recursive <repository-url>
cd SABLE-main
git submodule update --init --recursive
```

### 2. System Dependencies

Install the following system dependencies:
- Python3
- GCC
- CMake

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Build find-submatrices/

`BlockDetector` uses the native C++ partitioner in `find-submatrices/`.
Build it before running frontend extraction tests or benchmarks:

```bash
cmake -S find-submatrices -B find-submatrices/build
cmake --build find-submatrices/build --target partition_matrix -j$(nproc)
```

### 5. Optional kernel dispatch dependencies

The following are only needed if you dispatch kernels that use them.

#### Intel MKL (for `MixedVBR*`, `MKLVBR*`, `MKLCSR*` kernels)

Install Intel MKL. Make sure `setvars.sh` has been executed in current working shell.

#### spv8-public/ (for `SPV8CSRSpmv` kernel)

```bash
cd spv8-public/
# Follow build instructions for spv8-public (just `make` in root dir should work)
cd ..
```

#### sparse-register-tiling/ (for `SPRegCSRSpmm` kernel)

First generate the micro-kernels, then build with CMake:

```bash
cd sparse-register-tiling/spmm_nano_kernels/
python3 -m codegen.generate_ukernels
cd ..
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_AVX512=True
make -j$(nproc) SPMM_demo
cd ../..
```

### 6. Configuration

Shared compiler flags and MKL discovery live in `sable/build_config.py`.
UZP, SPV8, MKL, and naive SpMV sparse dispatches are exposed through the frontend kernel API.

### 7. Running Benchmarks

To benchmark, run `bench_suitesparse.py`.

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
