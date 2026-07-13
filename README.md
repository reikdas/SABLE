## Usage

The SABLE frontend lets you extract matrix regions into formats such as VBR,
VDIA, and CSR, dispatch each format to a compatible kernel, and compile
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
# the native C++ partitioner in find-submatrices/ to discover block
# regions. It returns a VBR format and shrinks the residual.
vbr = plan.extract(
    BlockDetector(min_density=0.5, min_area=2500, threads=4)
)

# CSRConvertor claims whatever non-zeros remain as plain CSR.
csr = plan.extract(CSRConvertor())

# ── 4. Dispatch kernels ──────────────────────────────────────────────
# MixedVBRSpmm uses a heuristic per VBR block: blocks whose smallest
# dimension ≥ 8 and aspect ratio ≤ 100 are multiplied with MKL's
# cblas_dgemm; smaller or very thin blocks fall back to a naive loop.
plan.dispatch(vbr, MixedVBRSpmm())

# NaiveCSRSpmm multiplies the CSR residual with a triple-nested loop.
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
  runs the generated computation for `bench` iterations, and returns the
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
`accepts` to the format type and `operation` to an operation type
(`Operation.SPMV` or `Operation.SPMM`; users can add their own operation).
The frontend derives the expected RHS rank from that operation and checks that
every dispatched kernel in a plan agrees.

The required method is `emit_call`.  It may return a C code string, an
`out_of_line(...)` marker, or a list containing either kind.  Kernels can also
implement `emit_timed_calls`; when present, each returned snippet is timed as a
separate dispatch part.

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

#### Moving generated code out of line

Use `out_of_line(...)` when a generated snippet should be hoisted into a helper
function instead of emitted directly in `main`.  This is useful for loop-heavy
generated kernels where duplicating the loop body for every block or segment
would grow the C source.  Plain strings stay inline, so a kernel can mix inline
library calls with out-of-line loop bodies.

```python
from sable import out_of_line


def emit_timed_calls(self, fmt, y: str, x: str, rhs) -> list:
    calls = []
    nrhs = rhs.shape[1]
    for r0, r1, c0, c1, offset in self.blocks:
        if self.use_blas(r0, r1, c0, c1):
            calls.append(f"""\
cblas_dgemm(...,
    &{fmt.val}[{offset}], {r1} - {r0},
    &{x}[{c0} * {nrhs}], {nrhs},
    1.0,
    &{y}[{r0} * {nrhs}], {nrhs});
""")
        else:
            loop_body = f"""\
for (int i = r0; i < r1; i++) {{
    for (int j = c0; j < c1; j++) {{
        double a = {fmt.val}[offset + (j - c0) * (r1 - r0) + (i - r0)];
        for (int k = 0; k < nrhs; k++) {{
            {y}[i * nrhs + k] += a * {x}[j * nrhs + k];
        }}
    }}
}}
"""
            calls.append(
                out_of_line(
                    loop_body,
                    name=f"{fmt.val}_spmm_naive_block",
                    parameters=[
                        "int r0",
                        "int r1",
                        "int c0",
                        "int c1",
                        "int offset",
                        "int nrhs",
                    ],
                    arguments=[r0, r1, c0, c1, offset, nrhs],
                )
            )
    return calls
```

The compiler automatically adds `double *y`, `const double *x`, and any staged
`Rep` arrays referenced in the body to the helper signature.  `parameters` and
`arguments` are for scalar values that vary between calls, such as bounds,
bandwidths, and offsets.  Reusing the same `name` with the same body and
signature creates one helper function and multiple calls; reusing a name with
a different body or signature is rejected.

Built-in VBR and VDIA kernels use this mechanism by default.  VBR naive loop
blocks are dispatched through one parameterized helper per operation, while
VBR MKL/CBLAS calls remain inline.  VDIA segment loops are also dispatched
through one parameterized helper per operation.

Kernels that need external libraries (MKL, spv8, sparse-register-tiling)
return the appropriate flags from `compile_flags()` / `link_flags()` and
headers from `emit_includes()`.  See `sable/kernels/vbr.py` and
`sable/kernels/csr.py` for examples.

## Setup Instructions

### 1. Clone Repository

For a fresh checkout, clone the repository and its pinned submodule commits in
one command:

```bash
git clone --recurse-submodules <repository-url> SABLE
cd SABLE
```

To initialize or repair submodule configuration in an existing checkout, run:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Run `git submodule sync --recursive` after pulling a change to `.gitmodules` so
the URLs in the local Git configuration are refreshed.  The update command
checks out the exact submodule commits recorded by SABLE; it does not track the
latest commit on each submodule's default branch.  If a submodule contains
local changes, Git will preserve them and may refuse to update it.  Commit or
stash those changes before retrying.

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

Shared compiler flags, MKL discovery, and kernel-family enum names live in
`sable/build_config.py`.  UZP, SPV8, MKL, naive CSR/VBR/VDIA, and mixed VBR
kernels are exposed through the frontend kernel API.  Benchmark kernels are
grouped by format family: CSR, VBR, and VDIA.

### 7. Running Benchmarks

To benchmark, run `bench_suitesparse.py`.

## Testing

### Quick tests (no MKL / hardware dependencies)

These tests verify the benchmarking infrastructure itself — frontend compiler
assembly and C code generation — without running any actual sparse
computations.  They run in seconds and need only Python + pytest:

```bash
# Frontend compiler unit tests, including kernel build requirements
python3 -m pytest tests/test_frontend_compiler.py -v

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
