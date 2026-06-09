"""Generate a self-contained VBR(+CSR) data set for the *interpreted* C kernels.

The compiled SABLE path consumes the VBR indirection arrays (rpntr, cpntr,
bpntrb, bindx, indx) at code-generation time and bakes one hard-coded block
call per packed block into the C source.  The interpreted
kernels in this directory instead traverse those indirection arrays at run
time, so this script writes *all* of them into the .sabledata file (the stock
SABLE writer only emits the `Rep` arrays: vbr_val and the CSR arrays).

It emits, into this directory:
  - fixture.sabledata : VBR + CSR arrays, one `label=[...]` line each, in the
                        fixed order the interpreted C readers expect.
  - x.vector          : dense RHS vector (all ones) for the SpMV kernel.
  - x.matrix          : dense RHS matrix (row-major) for the SpMM kernel.

It also prints the reference results so the C output can be checked.
"""

import os

import numpy
import scipy.sparse

from sable import Matrix, Plan
from sable.extractors import BandExtractorSkip, BlockDetectorSkip, CSRConvertor

HERE = os.path.dirname(os.path.abspath(__file__))

FIXTURE_SIZE = 14
BLOCK_REGIONS = [(3, 6, 3, 6), (6, 14, 6, 14)]
CSR_RESIDUAL_ENTRIES = [
    (0, 13, 5.0), (1, 12, 4.0), (2, 11, 3.0),
    (11, 2, 6.0), (12, 1, 1.0), (13, 0, 2.0),
]
# A VDIA band over rows 0..3, +/-2 diagonals (covers the 3x3 top-left corner).
BAND_REGIONS = [{"diag_offset": 0, "segments": [{"rows": [0, 3], "bandwidth": [2, 2]}]}]
NRHS = 4


def build_matrix():
    values = numpy.zeros((FIXTURE_SIZE, FIXTURE_SIZE), dtype=float)
    values[3:6, 3:6] = numpy.arange(101, 110, dtype=float).reshape(3, 3)
    values[6:14, 6:14] = numpy.arange(201, 265, dtype=float).reshape(8, 8)
    for r, c, v in CSR_RESIDUAL_ENTRIES:
        values[r, c] = v
    return scipy.sparse.csr_matrix(values)


def build_vdia_matrix():
    """build_matrix() plus a dense 3x3 band in the top-left corner, so the
    matrix decomposes into VDIA (rows 0..2) + VBR (two blocks) + CSR residual."""
    values = numpy.zeros((FIXTURE_SIZE, FIXTURE_SIZE), dtype=float)
    values[0:3, 0:3] = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    values[3:6, 3:6] = numpy.arange(101, 110, dtype=float).reshape(3, 3)
    values[6:14, 6:14] = numpy.arange(201, 265, dtype=float).reshape(8, 8)
    for r, c, v in CSR_RESIDUAL_ENTRIES:
        values[r, c] = v
    return scipy.sparse.csr_matrix(values)


def write_array(f, label, values):
    f.write(f"{label}=[")
    f.write(",".join(str(v) for v in values))
    f.write("]\n")


def write_sabledata(data_path, A=None):
    """Write the interpreted .sabledata for matrix ``A`` (default: fixture).

    Emits every VBR indirection array plus the CSR residual arrays, in the
    fixed order the interpreted C readers consume them.  Returns the matrix.
    """
    if A is None:
        A = build_matrix()
    plan = Plan(Matrix(A, name="fixture"), artifact_dir="/tmp/interp_art")
    vbr = plan.extract(BlockDetectorSkip(blocks=BLOCK_REGIONS))
    csr = plan.extract(CSRConvertor())

    with open(data_path, "w") as f:
        # ---- VBR indirection arrays (read in THIS order by the C kernels) ----
        write_array(f, "rpntr", vbr.rpntr)
        write_array(f, "cpntr", vbr.cpntr)
        write_array(f, "bpntrb", vbr.bpntrb)
        write_array(f, "bindx", vbr.bindx)
        write_array(f, "indx", vbr.indx)
        write_array(f, "vbr_val", vbr.val.values)
        # ---- CSR residual arrays ----
        write_array(f, "csr_indptr", csr.indptr.values)
        write_array(f, "csr_indices", csr.indices.values)
        write_array(f, "csr_val", csr.values.values)
    return A


def write_vdia_sabledata(data_path, A=None):
    """Write the interpreted .sabledata for the VDIA+VBR+CSR decomposition of
    ``A`` (default: the VDIA fixture), in the fixed order the C readers expect.
    Returns the matrix."""
    if A is None:
        A = build_vdia_matrix()
    plan = Plan(Matrix(A, name="fixture_vdia"), artifact_dir="/tmp/interp_art")
    vdia = plan.extract(BandExtractorSkip(bands=BAND_REGIONS))
    vbr = plan.extract(BlockDetectorSkip(blocks=BLOCK_REGIONS))
    csr = plan.extract(CSRConvertor())

    with open(data_path, "w") as f:
        # ---- VDIA segment metadata + diagonals/values ----
        write_array(f, "seg_row_start", vdia.seg_row_start)
        write_array(f, "seg_nrows", vdia.seg_nrows)
        write_array(f, "seg_ndiags", vdia.seg_ndiags)
        write_array(f, "seg_idiag_ptr", vdia.seg_idiag_ptr)
        write_array(f, "seg_val_ptr", vdia.seg_val_ptr)
        write_array(f, "vdia_idiag", vdia.idiag.values)
        write_array(f, "vdia_val", vdia.val.values)
        # ---- VBR indirection arrays ----
        write_array(f, "rpntr", vbr.rpntr)
        write_array(f, "cpntr", vbr.cpntr)
        write_array(f, "bpntrb", vbr.bpntrb)
        write_array(f, "bindx", vbr.bindx)
        write_array(f, "indx", vbr.indx)
        write_array(f, "vbr_val", vbr.val.values)
        # ---- CSR residual arrays ----
        write_array(f, "csr_indptr", csr.indptr.values)
        write_array(f, "csr_indices", csr.indices.values)
        write_array(f, "csr_val", csr.values.values)
    return A


def write_dense(path, values):
    """Write a flat, comma-separated dense RHS file."""
    with open(path, "w") as f:
        f.write(",".join(str(v) for v in numpy.asarray(values).reshape(-1)) + "\n")


def main():
    A = build_matrix()
    data_path = os.path.join(HERE, "fixture.sabledata")
    write_sabledata(data_path, A)

    # Dense RHS for SpMV: all ones.
    x_vec = numpy.ones(FIXTURE_SIZE)
    write_dense(os.path.join(HERE, "x.vector"), x_vec)

    # Dense RHS for SpMM: row-major arange so the result is non-trivial.
    X = numpy.arange(FIXTURE_SIZE * NRHS, dtype=float).reshape(FIXTURE_SIZE, NRHS)
    write_dense(os.path.join(HERE, "x.matrix"), X)

    # VDIA + VBR + CSR fixture (shares x.vector / x.matrix as RHS).
    write_vdia_sabledata(os.path.join(HERE, "fixture_vdia.sabledata"))

    print(f"wrote {data_path}")
    print(f"nrows={A.shape[0]} ncols={A.shape[1]} nrhs={NRHS}")
    print("reference y (spmv, x=ones):")
    print(list(A @ x_vec))
    print("reference Y (spmm, nrhs=4, row-major flat):")
    print(list((A @ X).reshape(-1)))


if __name__ == "__main__":
    main()
