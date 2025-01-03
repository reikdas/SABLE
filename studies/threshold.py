import subprocess
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Hack for imports - FIXME
FILEPATH=pathlib.Path(__file__).resolve().parent
sys.path.append(str(FILEPATH.parent))

from src.codegen import *
from synthesize_matrices.vbr_matrices_gen import vbr_matrix_gen
from utils.fileio import write_dense_matrix

BENCHMARK_FREQ = 5
COMPILE_TIMEOUT = 60 * 10

LIBXSMM_CODEGEN_DIR = "Generated_SpMM_libxsmm"

def draw_heatmap():
    data = pd.read_csv("threshold_results.csv")
    # Calculate the ratio of nonzeros_time to sable_time
    data["time_ratio"] = data["nonzeros_time"] / data["sable_time"]
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    heatmap_data = data.pivot(index="perc_zeros", columns="dim", values="time_ratio")
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": "Speedup of SABLE over only iterating over nnzs"}
    )
    plt.xlabel("Side of nnz_block")
    plt.ylabel("Percentage of Zeros per nnz_block")
    plt.savefig("heatmap.png")

def calculate_threshold():
    perc_zeros_list = [0, 10, 20, 30, 40, 50, 75, 99]
    dims = [10, 20, 50, 100, 400]
    assert (10000%dim == 0 for dim in dims)
    write_dense_matrix(1.0, 10000, 512)
    with open("threshold_results.csv", "w") as f:
        f.write("dim,perc_zeros,nnz,nonzeros_time,sable_time\n")
        for dim in dims:
            split = 10000//dim
            for perc_zeros in perc_zeros_list:
                nnz = (dim*dim*(100-perc_zeros))//100
                if nnz>100000:
                    continue
                fname = vbr_matrix_gen(10000, 10000, "uniform", split, split, 1, perc_zeros, 0, True, "Generated_VBR_threshold")
                print(fname)
                only_nonzeros(fname, dir_name="Generated_SpMM_nonzeros", vbr_dir="Generated_VBR_threshold")
                try:
                    subprocess.run(["gcc", "-O3", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx", "-o", fname, fname+".c"], cwd="Generated_SpMM_nonzeros", timeout=COMPILE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    print("Nonzeros: Compilation failed for ", fname)
                    continue
                output = subprocess.run(["./"+fname], capture_output=True, cwd="Generated_SpMM_nonzeros")
                # vbr_spmm_codegen(fname, density=0, dir_name="Generated_SpMM", vbr_dir="Generated_VBR_threshold", threads=1)
                vbr_spmm_codegen_libxsmm(fname, density=0, dir_name=LIBXSMM_CODEGEN_DIR, vbr_dir="Generated_VBR_threshold", threads=1)
                try:
                    # subprocess.run(["gcc", "-O3", "-pthread", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-mavx", "-o", fname, fname+".c"], cwd=LIBXSMM_CODEGEN, timeout=COMPILE_TIMEOUT)
                    subprocess.run(["gcc", "-o", fname, fname+".c", "-O3", "-march=native", "-funroll-all-loops", "-mprefer-vector-width=512", "-I", "/local/scratch/a/das160/libxsmm/include","-L", "/local/scratch/a/das160/libxsmm/lib","-lblas","-lm","-mavx"], cwd=LIBXSMM_CODEGEN_DIR, timeout=COMPILE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    print("SABLE: Compilation failed for ", fname)
                    continue
                output = subprocess.run(["./"+fname], capture_output=True, cwd=LIBXSMM_CODEGEN_DIR)
                nonzeros_times = []
                sable_times = []
                for i in range(BENCHMARK_FREQ):
                    print(f"Benchmarking iteration ", i+1, flush=True)
                    output = subprocess.run(["./"+fname], capture_output=True, cwd="Generated_SpMM_nonzeros")
                    execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                    nonzeros_times.append(execution_time)
                    output = subprocess.run(["./"+fname], capture_output=True, cwd=LIBXSMM_CODEGEN_DIR)
                    execution_time = output.stdout.decode("utf-8").split("\n")[0].split(" = ")[1]
                    sable_times.append(execution_time)
                print("SABLE: ", sable_times)
                print("Nonzeros: ", nonzeros_times)
                f.write(",".join([str(dim), str(perc_zeros), str(nnz), str(sum([float(time) for time in nonzeros_times])/BENCHMARK_FREQ), str(sum([float(time) for time in sable_times])/BENCHMARK_FREQ)]))
                f.write("\n")
                f.flush()

if __name__ == "__main__":
    # calculate_threshold()
    draw_heatmap()
