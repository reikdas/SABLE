# https://github.com/SpRegTiling/sparse-register-tiling/blob/main/tools/smtx_to_mtx.py

import numpy  as np
from collections import namedtuple
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
import argparse
import os
import pathlib

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH, "..")

CSRPattern = namedtuple('CSRPattern',
                        ['nrows', 'ncols', 'nnz', 'row_ptrs', 'col_indices'])


# Read the custom sputnik data file structure, its a pattern only file
# format so contains no values
#   line 1: nrows, ncols, nnz
#   line 2: row_ptrs ... (space delimited)
#   line 3: col_indices ... (space delimited)
def read_pattern(filepath):
    with open(filepath) as file:
        lines = [file.readline() for _ in range(3)]
        nrows, ncols, nnz = [int(x) for x in lines[0].split(',')]
        return CSRPattern(nrows=nrows, ncols=ncols, nnz=nnz,
                          row_ptrs=np.fromstring(lines[1], dtype=int, sep=" "),
                          col_indices=np.fromstring(lines[2], dtype=int, sep=" ")
                          )
    return None


# Convert the pattern to a scipy CSR matrix with 1s for all the values to
# enable using scipy libraries/plotting/etc.
def pattern_to_scipy_csr(csr_pattern: CSRPattern):
    nnz = len(csr_pattern.col_indices)
    return csr_matrix(([1] * nnz, csr_pattern.col_indices, csr_pattern.row_ptrs),
                      (csr_pattern.nrows, csr_pattern.ncols))

if __name__ == "__main__":
    directory = f"{BASE_PATH}/dlmc/transformer/magnitude_pruning/0.7"
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".smtx"):
            smtx_file = os.path.join(directory, filename)
            mtx_file = os.path.join(BASE_PATH, "Real_mtx", filename[:-5] + ".mtx")
            mmwrite(mtx_file, pattern_to_scipy_csr(read_pattern(smtx_file)))
