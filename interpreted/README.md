# Interpreted VDIA + VBR + CSR kernels

A standalone, *interpreted* counterpart to SABLE's compiled VDIA+VBR+CSR codegen,
with the mixed MKL/naive VBR block dispatch embedded.

In the compiled path (`sable/kernels/*.py` + `sable/compiler.py`), the structural
indirection arrays are walked **at code-generation time** and one hard-coded call
per packed block / diagonal segment is emitted into the C source (e.g.
`vbr_val_spmv_naive_block(y, x, vbr_val, 3, 6, 3, 6, 0);`).  Only the `Rep`
arrays (`vbr_val`, `vdia_val`, `vdia_idiag`, and the CSR arrays) end up in the
`.sabledata` file.

The kernels here instead walk the indirection arrays **at run time**: they read
every array (including the structural ones) from a `.sabledata` file, discover
the structure, and dispatch each block/segment, then apply the CSR residual.

VBR carries **only packed dense blocks**: `bpntrb` is a CSR-style row pointer
(block-row `a`'s blocks are `bindx[bpntrb[a] .. bpntrb[a+1])`, values at
`val + indx[k]`).  There is no `ublocks` list and no `bpntre` — non-dense
sub-blocks are carried entirely by the CSR residual, and an empty block-row is
just a zero-length range.

## Dispatch per format

- **VDIA** — SpMV uses MKL `mkl_ddiamv` (per segment); SpMM uses the naive
  per-segment diagonal loop.
- **VBR** — mixed MKL/naive, decided per block at run time by the same heuristic
  as `MixedVBRSpmv`/`MixedVBRSpmm`: a block of size `rows x cols` goes to MKL
  (`cblas_dgemv`/`cblas_dgemm`) iff `min(rows, cols) >= 8` **and**
  `max(rows, cols) / min(rows, cols) <= 100`; otherwise the naive column-major
  loop runs.
- **CSR** — selectable at run time via argv (`naive` or `mkl`, the latter using
  `mkl_sparse_d_mv`/`mkl_sparse_d_mm`).

## Files

| File | Purpose |
|------|---------|
| `make_data.py` | Builds the 14×14 fixture (a 3×3 VDIA band + two VBR blocks + CSR residual) and writes `fixture_vdia.sabledata`, `x.vector` (SpMV RHS) and `x.matrix` (SpMM RHS, nrhs=4). Exposes `build_vdia_matrix` / `write_vdia_sabledata` / `write_dense` for reuse by the tests and benchmark. |
| `interp_vdia_vbr_csr_spmv.c` | Interpreted VDIA(MKL)+VBR(mixed)+CSR SpMV. |
| `interp_vdia_vbr_csr_spmm.c` | Interpreted VDIA(naive)+VBR(mixed)+CSR SpMM. |

The VDIA `.sabledata` prepends the VDIA segment metadata and diagonals to the
VBR+CSR arrays, in the fixed order the C readers consume them:

```
seg_row_start, seg_nrows, seg_ndiags, seg_idiag_ptr, seg_val_ptr, vdia_idiag, vdia_val,
rpntr, cpntr, bpntrb, bindx, indx, vbr_val,
csr_indptr, csr_indices, csr_val
```

Array sizes are not baked in — the readers discover each array's length from the
`[...]` delimiters, and the matrix shape from the partition pointers
(`nrows = rpntr[-1]`, `ncols = cpntr[-1]`).

## Run

```bash
# from repo root, so the `sable` package is importable
PYTHONPATH=. python3 interpreted/make_data.py

cd interpreted
gcc -O2 -o interp_vdia_vbr_csr_spmv interp_vdia_vbr_csr_spmv.c \
    -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_rt
gcc -O2 -o interp_vdia_vbr_csr_spmm interp_vdia_vbr_csr_spmm.c \
    -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_rt

# argv: <data> <rhs> <csr_backend: naive|mkl> [y]
MKL_THREADING_LAYER=GNU ./interp_vdia_vbr_csr_spmv fixture_vdia.sabledata x.vector naive
MKL_THREADING_LAYER=GNU ./interp_vdia_vbr_csr_spmm fixture_vdia.sabledata x.matrix mkl
```

Each binary times the compute over `SABLE_BENCH` iterations (a plain `#define`
near the top of each source: 100 for SpMV, 30 for SpMM) and prints one
`Dispatch N:` line of per-iteration nanosecond timings per format — VDIA
(Dispatch 1), VBR (Dispatch 2), CSR (Dispatch 3).  Passing a trailing `y`
argument makes the binary print its result vector instead (used by the value
test below).

## Test

`tests/test_interpreted_vdia_values.py` compiles each kernel, runs it against the
fixture (both CSR backends; SpMM at several `nrhs`), and compares the result to a
NumPy reference (requires gcc + MKL).  These are value tests only — there are
intentionally no golden/text-snapshot tests for the hand-written kernels.

```bash
python3 -m pytest tests/test_interpreted_vdia_values.py -v
```
