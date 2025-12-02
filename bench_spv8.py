import os
import pathlib
import subprocess

import scipy

from bench_suitesparse import eval, eval_single_file
from src.autopartition import cut_indices2_fast, similarity2_numba
from src.codegen import gen_single_threaded_spmv_spv8
from src.consts import CFLAGS as CFLAGS
from utils.convert_real_to_vbr import convert_sparse_to_vbrc
from utils.utils import check_file_matches_parent_dir

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

COMPILE_TIMEOUT = 60 * 60 * 4

mtx_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse"))
codegen_dir = os.path.join(BASE_PATH, "Generated_SpMV_Python_sparse")
vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "Generated_VBR_Sparse"))

cut_indices = cut_indices2_fast
similarity = similarity2_numba
cut_threshold = 0.2

if __name__ == "__main__":
    with open(os.path.join(BASE_PATH, "results", "res_sparse_1.csv"), "w") as result_file:
        result_file.write("Filename,SABLE(ns)\n")
        for file_path in mtx_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
                fname = pathlib.Path(file_path).resolve().stem
                if fname not in eval:
                    continue
                try:
                    A = scipy.io.mmread(file_path)
                except:
                    print("Error reading file:", file_path)
                    continue
                print(f"Processing {fname}")
                relative_path = file_path.relative_to(mtx_dir)
                dest_path = vbr_dir / relative_path.with_suffix(".vbr")
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                A = scipy.sparse.csc_matrix(A, copy=False)
                cpntr, rpntr = cut_indices(A, cut_threshold, similarity)
                val, indx, bindx, bpntrb, bpntre, ublocks, indptr, indices, csr_val = convert_sparse_to_vbrc(A, rpntr, cpntr, fname, os.path.join(vbr_dir,fname), op="spmv", density=100)
                gen_single_threaded_spmv_spv8(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, codegen_dir, fname, os.path.join(vbr_dir, fname), bench=100)
                subprocess.run(["taskset", "-a", "-c", "0", "gcc", f"{fname}.c", "-o", fname] + CFLAGS, cwd=codegen_dir, check=True, timeout=COMPILE_TIMEOUT)
                # Evaluate the generated program immediately
                print(f"Evaluating {fname}...")
                median_exec_time_unroll = eval_single_file(fname, codegen_dir)
                result_file.write(f"{fname},{median_exec_time_unroll}\n")
                result_file.flush()
                print(f"Evaluation complete for {fname}")