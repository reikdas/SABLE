import subprocess
import os
import pathlib
import statistics

from scipy.io import mmread
import pandas as pd
import numpy as np

from utils.utils import check_file_matches_parent_dir, set_ulimit, extract_mul_nums
from utils.fileio import write_dense_vector, read_vbr
from utils.convert_real_to_vbr import convert_vbr_to_compressed
from src.consts import CFLAGS as CFLAGS
from src.consts import MKL_FLAGS as MKL_FLAGS
from src.codegen import gen_single_threaded_spmv, gen_multi_threaded_spmv

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

BENCHMARK_FREQ = 100
COMPILE_TIMEOUT = 60 * 60 * 4

results_dir = os.path.join(BASE_PATH, "results")

def remove_outliers_deciles(data):
    if len(data) < 10:  # Ensure enough data points for deciles
        return data
    
    D1 = np.percentile(data, 10)  # 10th percentile
    D9 = np.percentile(data, 90)  # 90th percentile

    return [x for x in data if D1 <= x <= D9]

if __name__ == "__main__":
    mtx_dir = pathlib.Path(os.path.join(BASE_PATH, "Suitesparse"))
    pid = os.getpid()
    cpu_affinity = os.sched_getaffinity(pid)
    #threads = [1, 2, 4, 8]
    threads = [1]
    eval = ["heart1", "std1_Jac2", "lowThrust_11", "lowThrust_12", "hangGlider_4", "hangGlider_5", "freeFlyingRobot_9"]
    vbr_dir = os.path.join(BASE_PATH, "Generated_VBR")
    mkl_vbrc_dir = os.path.join(BASE_PATH, "MKL_VBRC")
    codegen_dir = os.path.join(BASE_PATH, "Generated_MKL")
    if not os.path.exists(mkl_vbrc_dir):
        os.makedirs(mkl_vbrc_dir)
    for thread in threads:
        with open(os.path.join(BASE_PATH, "results", f"mkl-spmv-suitesparse_{thread}thrd.csv"), "w") as f:
            f.write("Matrix,Time(ns)\n")
            for file_path in mtx_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
                    fname = pathlib.Path(file_path).resolve().stem
                    if fname not in eval:
                        continue
                    print(f"Processing {fname}")
                    mtx = mmread(file_path)
                    cols = mtx.get_shape()[1]
                    # write_dense_vector(1.0, cols)
                    val,indx, bindx, rpntr, cpntr, bpntrb, bpntre = read_vbr(os.path.join(vbr_dir, fname, fname+".vbr"))
                    val, indx, bindx, bpntrb, bpntre, ublocks, indptr, indices, csr_val = convert_vbr_to_compressed(val, rpntr, cpntr, indx, bindx, bpntrb, bpntre, fname, mkl_vbrc_dir, 100)
                    for thread in threads:
                        if thread == 1:
                            gen_single_threaded_spmv(val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, codegen_dir+"_"+str(thread), fname, mkl_vbrc_dir, bench=100)
                        else:
                            gen_multi_threaded_spmv(thread, val, indx, bindx, rpntr, cpntr, bpntrb, bpntre, ublocks, indptr, indices, csr_val, codegen_dir+"_"+str(thread), fname, os.path.join(vbr_dir, fname))
                    subprocess.run(["gcc", f"{fname}.c", "-o", fname] + CFLAGS + MKL_FLAGS, cwd=codegen_dir+"_"+str(thread), check=True, timeout=COMPILE_TIMEOUT)
                    l = []
                    for _ in range(100):
                        output = subprocess.check_output(["taskset", "-a", "-c", "0", f"./{fname}"], cwd=codegen_dir+"_"+str(thread), preexec_fn=set_ulimit).decode("utf-8").split("\n")
                        if "warning" in output[0].lower():
                            output = output[1]
                        else:
                            output = output[0]
                        output = extract_mul_nums(output)
                        median_exec_time_unroll = statistics.median([float(x) for x in output])
                        l.append(median_exec_time_unroll)
                    l = remove_outliers_deciles(l)
                    csr_spmv_exec_time = statistics.mean(l)
                    f.write(f"{fname},{csr_spmv_exec_time}\n")
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
