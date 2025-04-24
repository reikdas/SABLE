import argparse
import os

import scipy
import pathlib

from src.autopartition import cut_indices2_fast, similarity2_numba
from utils.convert_real_to_vbr import convert_sparse_to_vbr

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform MTX to VBR.")
    parser.add_argument("mtx_file", type=str, help="Path to the MTX file.")
    args = parser.parse_args()

    mtx_file = args.mtx_file
    fname = pathlib.Path(mtx_file).resolve().stem
    dst_dir = os.path.join(BASE_PATH, "artifact_partition")
    A = scipy.io.mmread(mtx_file)
    A = scipy.sparse.csc_matrix(A, copy=False)
    cut_threshold = 0.2
    cpntr, rpntr = cut_indices2_fast(A, cut_threshold, similarity2_numba)
    convert_sparse_to_vbr(A, rpntr, cpntr, fname, dst_dir)
