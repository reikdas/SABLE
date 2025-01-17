import subprocess
import pathlib
import sys
import statistics

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Hack for imports - FIXME
FILEPATH=pathlib.Path(__file__).resolve().parent
sys.path.append(str(FILEPATH.parent))

from src.codegen import *
from src.baseline import *
from synthesize_matrices.vbr_matrices_gen import vbr_matrix_gen
from utils.fileio import write_dense_matrix, write_dense_vector
from utils.mtx_matrices_gen import vbr_to_mtx

FILEPATH=pathlib.Path(__file__).resolve().parent

BENCHMARK_FREQ = 5
COMPILE_TIMEOUT = 60 * 10

CODEGEN_DIR = "Generated_SpMV_threshold"
MTX_DIR = "Generated_MMarket_threshold"
VBR_DIR = "Generated_VBR_threshold"
BASELINE = "nonzeros"
# BASELINE = "psc"
BASELINE_CODEGEN_DIR = f"Generated_SpMV_{BASELINE}"
BASELINE_CODEGEN_FUNC = only_nonzeros_spmv


def draw_heatmap():
    data = pd.read_csv(os.path.join(FILEPATH,"threshold_results.csv"))
    data = data[data['sable_time'] != 0]
    # Calculate the ratio of nonzeros_time to sable_time
    data["time_ratio"] = data[f"{BASELINE}_time"] / data["sable_time"]
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    heatmap_data = data.pivot(index="perc_zeros", columns="dim", values="time_ratio")
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": f"Speedup of SABLE over {BASELINE}"}
    )
    plt.xlabel("Side of nnz_block")
    plt.ylabel("Percentage of Zeros per nnz_block")
    plt.savefig(os.path.join(FILEPATH,"heatmap.png"))

def calculate_threshold():
    perc_zeros_list = [0, 10, 20, 30, 40, 50, 75, 99]
    dims = [10, 20, 50, 100, 400]
    assert (10000%dim == 0 for dim in dims)
    write_dense_vector(1.0, 10000)
    with open(os.path.join(FILEPATH,"threshold_results.csv"), "w") as f:
        f.write(f"dim,perc_zeros,nnz,{BASELINE}_time,sable_time\n")
        for dim in tqdm(dims, desc="Processing dimensions"):
            split = 10000//dim
            for perc_zeros in tqdm(perc_zeros_list, desc=f"Dim {dim}: Processing % zeros", leave=False):
                nnz = (dim*dim*(100-perc_zeros))//100
                if nnz>100000:
                    continue
                fname = vbr_matrix_gen(10000, 10000, "uniform", split, split, 1, perc_zeros, 0, True, VBR_DIR)
                vbr_to_mtx(fname+".vbr", dir_name=MTX_DIR, vbr_dir=VBR_DIR)
                print(fname)
                BASELINE_CODEGEN_FUNC(fname, dir_name=BASELINE_CODEGEN_DIR, vbr_dir="Generated_VBR_threshold")
                try:
                    subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx", "-o", fname, fname+".c"], cwd=BASELINE_CODEGEN_DIR, timeout=COMPILE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    print("Nonzeros: Compilation failed for ", fname)
                    continue
                output = subprocess.run(["./"+fname], capture_output=True, cwd=BASELINE_CODEGEN_DIR)
                vbr_spmv_codegen(fname, density=0, dir_name=CODEGEN_DIR, vbr_dir=VBR_DIR, threads=1)
                try:
                    # subprocess.run(["gcc", "-O3", "-pthread", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx", "-o", fname, fname+".c"], cwd=LIBXSMM_CODEGEN, timeout=COMPILE_TIMEOUT)
                    subprocess.run(["gcc", "-o", fname, fname+".c", "-O3", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx"], cwd=CODEGEN_DIR, timeout=COMPILE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    print("SABLE: Compilation failed for ", fname)
                    continue
                output = subprocess.run(["./"+fname], capture_output=True, cwd=CODEGEN_DIR)
                baseline_times = []
                sable_times = []
                for i in range(BENCHMARK_FREQ):
                    print(f"Benchmarking iteration ", i+1, flush=True)
                    output = subprocess.run(["./"+fname], capture_output=True, cwd=BASELINE_CODEGEN_DIR)
                    execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                    baseline_times.append(execution_time)
                    output = subprocess.run(["./"+fname], capture_output=True, cwd=CODEGEN_DIR)
                    execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                    sable_times.append(execution_time)
                if len(baseline_times) == 0 or len(sable_times) == 0:
                    continue
                print("SABLE: ", sable_times)
                print(f"{BASELINE}: ", baseline_times)
                f.write(",".join([str(dim), str(perc_zeros), str(nnz), str(statistics.median([float(time) for time in baseline_times])), str(statistics.median([float(time) for time in sable_times]))]))
                f.write("\n")
                f.flush()

if __name__ == "__main__":
    calculate_threshold()
    draw_heatmap()
