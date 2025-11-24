import os
import pathlib
import sys

import psutil
from tqdm import tqdm

# Hack for imports - FIXME
FILEPATH=pathlib.Path(__file__).resolve().parent
sys.path.append(str(FILEPATH.parent))

from src.codegen import *
from src.consts import CFLAGS as CFLAGS
from src.consts import MKL_FLAGS as MKL_FLAGS
from utils.convert_real_to_vbr import vbrc_matrix_gen, _generate_vbrc_filename
from utils.fileio import write_dense_vector, write_dense_matrix

from bench_suitesparse_split_timings_c_spmm import eval_single_file_split_timings

FILEPATH=pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

BENCHMARK_FREQ = 5
EVAL_FREQ = 5

DENSE_CODEGEN_DIR_SPMV = os.path.join(FILEPATH, "..", "Generated_SpMV_threshold_Dense")
SPARSE_CODEGEN_DIR_SPMV = os.path.join(FILEPATH, "..", "Generated_SpMV_threshold_Sparse")
DENSE_CODEGEN_DIR_SPMM = os.path.join(FILEPATH, "..", "Generated_SpMM_threshold_Dense")
SPARSE_CODEGEN_DIR_SPMM = os.path.join(FILEPATH, "..", "Generated_SpMM_threshold_Sparse")

def calculate_threshold(op):
    dense_codegen_dir = eval(f"DENSE_CODEGEN_DIR_{op.upper()}")
    sparse_codegen_dir = eval(f"SPARSE_CODEGEN_DIR_{op.upper()}")
    dense_vbr_dir = os.path.join(FILEPATH, "..", "Generated_VBR_threshold_Dense")
    sparse_vbr_dir = os.path.join(FILEPATH, "..", "Generated_VBR_threshold_Sparse")
    codegen_func = eval(f"gen_single_threaded_{op.lower()}")
    perc_zeros_list = [0, 20, 40, 50, 75, 80, 85, 90, 95, 99]
    dims = [1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 625, 1000, 1250, 2000, 2500, 5000]
    with open(os.path.join(FILEPATH,f"threshold_results_{op.lower()}.csv"), "w") as f:
        f.write(f"dim1,dim2,perc_zeros,nnz,sparse,dense\n")
        for dim1 in tqdm(reversed(dims), desc="Processing dimensions"):
            for dim2 in tqdm(reversed(dims), desc=f"Dim {dim1}: Processing dimensions", leave=False):
                if op.lower() == "spmv":
                    write_dense_vector(1.0, dim2)
                elif op.lower() == "spmm":
                    write_dense_matrix(1.0, dim2, 512)
                else:
                    raise Exception("Unknown operation")
                for perc_zeros in tqdm(perc_zeros_list, desc=f"Dim {dim2}: Processing % zeros", leave=False):
                    nnz = (dim1*dim2*(100-perc_zeros))//100
                    fname = vbrc_matrix_gen(dim1, dim2, "uniform", 1, 1, 1, perc_zeros, 0, True, dense_vbr_dir, 0)
                    # fname = _generate_vbrc_filename(dim1, dim2, "uniform", 1, 1, 1, perc_zeros, 0, True)
                    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = read_vbrc(os.path.join(dense_vbr_dir,f"{fname}.vbrc"))
                    codegen_func(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, dense_codegen_dir, fname, dense_vbr_dir, bench=BENCHMARK_FREQ)
                    _, dense_time, _ = eval_single_file_split_timings(fname, dense_codegen_dir, EVAL_FREQ, False)
                    # print(f"Dim {dim1}x{dim2}, {perc_zeros}% zeros, nnz={nnz}, Dense time: {dense_time:.4f} ns")
                    
                    # Generate and test sparse variant
                    fname2 = vbrc_matrix_gen(dim1, dim2, "uniform", 1, 1, 1, perc_zeros, 0, True, sparse_vbr_dir, 100)
                    val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val = read_vbrc(os.path.join(sparse_vbr_dir,f"{fname2}.vbrc"))
                    codegen_func(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, sparse_codegen_dir, fname2, sparse_vbr_dir, bench=BENCHMARK_FREQ)
                    sparse_time, _, _ = eval_single_file_split_timings(fname2, sparse_codegen_dir, EVAL_FREQ, False)

                    f.write(",".join([str(dim1), str(dim2), str(perc_zeros), str(nnz), str(sparse_time), str(dense_time)]))
                    f.write("\n")
                    f.flush()
                    os.remove(os.path.join(dense_vbr_dir, fname+".vbrc"))
                    os.remove(os.path.join(sparse_vbr_dir, fname2+".vbrc"))

if __name__ == "__main__":
    calculate_threshold("spmv")
    # calculate_threshold("spmm")
