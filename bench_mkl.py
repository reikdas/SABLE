import subprocess
import os
import pathlib
import statistics

from scipy.io import mmread
import pandas as pd
import numpy as np

from utils.utils import check_file_matches_parent_dir, set_ulimit
from utils.fileio import write_dense_vector
from src.consts import CFLAGS as CFLAGS
from src.consts import MKL_FLAGS as MKL_FLAGS

FILEPATH = pathlib.Path(__file__).resolve().parent
BASE_PATH = os.path.join(FILEPATH)

BENCHMARK_FREQ = 100

results_dir = os.path.join(BASE_PATH, "results")

def remove_outliers_deciles(data):
    if len(data) < 10:  # Ensure enough data points for deciles
        return data
    
    D1 = np.percentile(data, 10)  # 10th percentile
    D9 = np.percentile(data, 90)  # 90th percentile

    return [x for x in data if D1 <= x <= D9]

if __name__ == "__main__":
    mtx_dir = pathlib.Path(os.path.join("/local", "scratch", "a", "Suitesparse"))
    pid = os.getpid()
    cpu_affinity = os.sched_getaffinity(pid)
    subprocess.check_output(["taskset", "-a", "-c", ",".join([str(x) for x in cpu_affinity]), "g++", "-o", "mkl-spmv", "mkl-spmv.cpp"] + CFLAGS + MKL_FLAGS, cwd=os.path.join(BASE_PATH, "src"))
    threads = [1, 2, 4, 8]
    for thread in threads:
        with open(os.path.join(BASE_PATH, "results", f"mkl-spmv-suitesparse_{thread}thrd.csv"), "w") as f:
            f.write("Matrix,Time(ns)\n")
            for file_path in mtx_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix == ".mtx" and check_file_matches_parent_dir(file_path):
                    fname = pathlib.Path(file_path).resolve().stem
                    print(f"Processing {fname}")
                    mtx = mmread(file_path)
                    cols = mtx.get_shape()[1]
                    write_dense_vector(1.0, cols)
                    l = []
                    for _ in range(100):
                        output = subprocess.run(["taskset", "-a", "-c", ",".join([str(x) for x in cpu_affinity]), f"{BASE_PATH}/src/mkl-spmv", str(file_path), str(thread), str(BENCHMARK_FREQ), os.path.join(BASE_PATH, "Generated_dense_tensors", f"generated_vector_{cols}.vector")], capture_output=True, check=True, text=True, preexec_fn=set_ulimit).stdout.split("\n")
                        if "warning" in output[0].lower():
                            output = output[1]
                        else:
                            output = output[0]
                        csr_spmv_exec_time = float(output.split(" ")[1])
                        l.append(csr_spmv_exec_time)
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