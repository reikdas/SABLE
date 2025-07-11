import os
import pathlib
import statistics
import subprocess
import time

import numpy as np
import scipy

from src.autopartition import cut_indices2_fast, similarity2_numba
from src.codegen import gen_single_threaded_spmv_python
from src.consts import CFLAGS as CFLAGS
from src.consts import MKL_FLAGS as MKL_FLAGS
from utils.convert_real_to_vbr import convert_sparse_to_vbrc
from utils.fileio import write_dense_vector
from utils.utils import (check_file_matches_parent_dir, extract_mul_nums,
                         set_ulimit)

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

COMPILE_TIMEOUT = 60 * 60 * 4

mtx_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse"))
codegen_dir = os.path.join(BASE_PATH, "Generated_SpMV_Python")
vbr_dir = pathlib.Path(os.path.join(BASE_PATH, "Generated_VBR"))

cut_indices = cut_indices2_fast
similarity = similarity2_numba
cut_threshold = 0.2

def remove_outliers_deciles(data):
    if len(data) < 10:  # Ensure enough data points for deciles
        return data
    
    D1 = np.percentile(data, 10)  # 10th percentile
    D9 = np.percentile(data, 90)  # 90th percentile

    return [x for x in data if D1 <= x <= D9]

def eval_single_file(fname, codegen_dir):
    pid = os.getpid()
    cpu_affinity = os.sched_getaffinity(pid)

    l = []
    for _ in range(100):
        output = subprocess.check_output(["taskset", "-a", "-c", ",".join([str(x) for x in cpu_affinity]), "python3", f"{fname}.py"], cwd=codegen_dir, preexec_fn=set_ulimit).decode("utf-8").split("\n")
        if "warning" in output[0].lower():
            output = output[1]
        else:
            output = output[0]
        output = extract_mul_nums(output)
        median_exec_time_unroll = statistics.median([float(x) for x in output])
        l.append(median_exec_time_unroll)
    l = remove_outliers_deciles(l)
    print(l)
    median_exec_time_unroll = statistics.mean(l)
    print(median_exec_time_unroll)
    return median_exec_time_unroll

eval = ["eris1176",
    "std1_Jac3",
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
    "nd12k",
    "std1_Jac2",
    "vsp_c-30_data_data"]

if __name__ == "__main__":
    
    with open(os.path.join(BASE_PATH, "results", "res_1.csv"), "w") as result_file:
        result_file.write("Filename,SABLE(ns)\n")
        with open(os.path.join(BASE_PATH, "results", "suitesparse_inspect.csv"), "w") as f:
            f.write("Filename,Codegen(ms)\n")
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
                    time1 = time.time_ns() // 1_000_000
                    cpntr, rpntr = cut_indices(A, cut_threshold, similarity)
                    time2 = time.time_ns() // 1_000_000
                    partition_time = time2-time1
                    time1 = time.time_ns() // 1_000_000
                    val, indx, bindx, bpntrb, bpntre, ublocks, indptr, indices, csr_val = convert_sparse_to_vbrc(A, rpntr, cpntr, fname, os.path.join(vbr_dir,fname))
                    time2 = time.time_ns() // 1_000_000
                    compress_time = time2-time1
                    write_dense_vector(1.0, cpntr[-1])
                    
                    codegen_time = gen_single_threaded_spmv_python(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, codegen_dir, fname, os.path.join(vbr_dir, fname), bench=100)

                    f.write(f"{fname},{partition_time + compress_time + codegen_time},\n")
                    f.flush()
                    print(f"Done {fname}")
                    
                    # Evaluate the generated program immediately
                    print(f"Evaluating {fname}...")
                    median_exec_time_unroll = eval_single_file(fname, codegen_dir)
                    result_file.write(f"{fname},{median_exec_time_unroll}\n")
                    result_file.flush()
                    print(f"Evaluation complete for {fname}")