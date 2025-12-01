import subprocess
import os
import pathlib
import statistics

from scipy.io import mmread
import pandas as pd
import numpy as np
import scipy

from utils.utils import check_file_matches_parent_dir, set_ulimit, extract_mul_nums
from utils.fileio import write_dense_vector
from src.autopartition import cut_indices2_fast, similarity2_numba
from utils.convert_real_to_vbr import convert_sparse_to_vbrc
from src.consts import CFLAGS as CFLAGS
from src.consts import MKL_FLAGS as MKL_FLAGS
from src.codegen import gen_single_threaded_spmv_mkl
from bench_suitesparse_split_timings_c import eval_single_file_split_timings

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

BENCHMARK_FREQ = 100
COMPILE_TIMEOUT = 60 * 60 * 4

results_dir = os.path.join(BASE_PATH, "results")

cut_indices = cut_indices2_fast
similarity = similarity2_numba
cut_threshold = 0.2

if __name__ == "__main__":
    eval = [
    "eris1176",
    "lp_wood1p",
    "jendrec1",
    "lowThrust_5",
    "hangGlider_4",
    "brainpc2",
    "hangGlider_3",
    "lowThrust_7",
    "lowThrust_11",
    "lowThrust_3",
    "lowThrust_6",
    "lowThrust_12",
    "hangGlider_5",
    "Journals",
    "bloweybl",
    "heart1",
    "TSOPF_FS_b9_c6",
    "Sieber",
    "case9",
    "c-30",
    "c-32",
    "freeFlyingRobot_10",
    "freeFlyingRobot_11",
    "freeFlyingRobot_12",
    "lowThrust_10",
    "lowThrust_13",
    "lowThrust_4",
    "lowThrust_8",
    "lowThrust_9",
    "lp_fit2p",
    "std1_Jac2",
    "std1_Jac3",
    "nd12k",
    "vsp_c-30_data_data"]
    mtx_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse"))
    pid = os.getpid()
    cpu_affinity = os.sched_getaffinity(pid)
    # threads = [1, 2, 4, 8]
    threads = [1]
    vbr_dir = os.path.join(BASE_PATH, "Generated_VBR_Sparse")
    mkl_vbrc_dir = os.path.join(BASE_PATH, "MKL_VBRC")
    codegen_dir = os.path.join(BASE_PATH, "Generated_MKL")
    cores = list(cpu_affinity)
    if not os.path.exists(mkl_vbrc_dir):
        os.makedirs(mkl_vbrc_dir)
    for thread in threads:
        print(f"{thread} Threads")
        with open(os.path.join(BASE_PATH, "results", f"mkl-spmv-suitesparse_{thread}thrd.csv"), "w") as f:
            f.write("Matrix,Time(ns)\n")
            for file_path in mtx_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
                    fname = pathlib.Path(file_path).resolve().stem
                    if fname not in eval:
                        continue
                    print(f"Processing {fname}")
                    try:
                        A = scipy.io.mmread(file_path)
                    except:
                        print("Error reading file:", file_path)
                        continue
                    # Ensure the matrix supports indexing — convert read matrix (often COO) to CSC
                    A = scipy.sparse.csc_matrix(A, copy=False)
                    print(f"Processing {fname}")
                    
                    # Get matrix dimensions and non-zeros
                    matrix_rows = A.shape[0]
                    matrix_cols = A.shape[1]

                    # Iterate over matrix to count nnz and assert matrix_nnz correctness
                    matrix_nnz = 0
                    for i in range(matrix_rows):
                        for j in range(matrix_cols):
                            if A[i, j] != 0:
                                matrix_nnz += 1
                    cpntr, rpntr = cut_indices(A, cut_threshold, similarity)
                    write_dense_vector(1.0, cpntr[-1])
                    val_sparse, indx_sparse, bindx_sparse, bpntrb_sparse, bpntre_sparse, ublocks_sparse, indptr_sparse, indices_sparse, csr_val_sparse = convert_sparse_to_vbrc(A, rpntr, cpntr, fname, os.path.join(vbr_dir,fname), op="spmv", density=100)
                    gen_single_threaded_spmv_mkl(val_sparse, indx_sparse, bindx_sparse, rpntr, cpntr, bpntrb_sparse, bpntre_sparse, ublocks_sparse, indptr_sparse, indices_sparse, csr_val_sparse, codegen_dir, fname, os.path.join(vbr_dir, fname), bench=100)
                    subprocess.run(["gcc", f"{fname}.c", "-g", "-o", fname] + CFLAGS + MKL_FLAGS, cwd=codegen_dir, check=True, timeout=COMPILE_TIMEOUT)
                    sparse_avg_sparse_time, _, _ = eval_single_file_split_timings(fname, codegen_dir, 100)
                    f.write(f"{fname},{sparse_avg_sparse_time}\n")
                    f.flush()
    
    results = {}
    for thread in threads:
        result_file = os.path.join(results_dir, f"mkl-spmv-suitesparse_{thread}thrd.csv")
        if os.path.exists(result_file):
            df = pd.read_csv(result_file)
            for _, row in df.iterrows():
                fname, exec_time = row["Matrix"], row["Time(ns)"]
                if fname not in results:
                    results[fname] = {}
                results[fname][thread] = exec_time
    
    # Writing merged results
    merged_results_path = os.path.join(results_dir, "mkl-spmv-merged-results.csv")
    with open(merged_results_path, "w") as f:
        f.write("Matrix," + ",".join([f"{t}thread" for t in threads]) + "\n")
        for fname, times in results.items():
            f.write(f"{fname}," + ",".join([str(times.get(t, 'N/A')) for t in threads]) + "\n")
    print(f"Merged results saved to {merged_results_path}")
